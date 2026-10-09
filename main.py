import os
import json
import csv
import time
import argparse
import signal
from datetime import datetime
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Import local modules
from src.recognition.face_recognizer import FaceRecognizer
from src.database.db_manager import DatabaseManager

# ============================================================
# INTELLIGENT FACE TRACKER - REAL-TIME PIPELINE
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

# Configuration mapping
MODEL_PATH = os.path.join(ROOT_DIR, config.get("models", {}).get("yolo_face", "models/yolo/yolov8n-face.pt"))
INPUT_SOURCE = config.get("input", {}).get("source", "video")
SOURCE_TYPE = config.get("input", {}).get("source_type", "folder")
RTSP_URL = config.get("input", {}).get("rtsp_url", "")
OUTPUT_DIR = os.path.join(ROOT_DIR, config.get("output", {}).get("directory", "output/tracking"))
LOG_DIR = os.path.join(ROOT_DIR, config.get("logging", {}).get("directory", "logs"))

CONFIDENCE = float(config.get("detection", {}).get("confidence", 0.5))
FRAME_SKIP = int(config.get("detection", {}).get("frame_skip", 1))
INFERENCE_SIZE = int(config.get("detection", {}).get("imgsz", 1280))
TRACKER = config.get("tracking", {}).get("tracker", "bytetrack.yaml")
EXIT_TIMEOUT_FRAMES = int(config.get("tracking", {}).get("exit_timeout_frames", 30))
SIMILARITY_THRESHOLD = float(config.get("recognition", {}).get("similarity_threshold", 0.40))
RETRY_FRAMES = int(config.get("recognition", {}).get("retry_frames", 5))
CROP_PADDING_RATIO = float(config.get("recognition", {}).get("crop_padding_ratio", 0.25))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
for crop_category in ("entries", "exits", "registrations"):
    os.makedirs(os.path.join(LOG_DIR, crop_category), exist_ok=True)

# System Event Logger setup
EVENT_LOG_PATH = os.path.join(LOG_DIR, "events.log")
event_logger = open(EVENT_LOG_PATH, "a", encoding="utf-8")
stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    log_system_event("Stop requested; flushing active tracks before shutdown.")

def log_system_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    event_logger.write(log_line)
    event_logger.flush()
    print(log_line.strip())

log_system_event("=== SYSTEM INITIALIZED: INTELLIGENT FACE TRACKER ===")
log_system_event(
    f"Compute Device: {DEVICE} | Confidence: {CONFIDENCE} | "
    f"Inference Size: {INFERENCE_SIZE} | Frame Skip: {FRAME_SKIP}"
)

# Date-partitioned crop storage helper
def get_crop_dir(event_type):
    today = datetime.now().strftime("%Y-%m-%d")
    folder = {
        "ENTRY": "entries",
        "EXIT": "exits",
        "REGISTRATION": "registrations",
    }.get(event_type.upper(), "registrations")
    target_path = os.path.join(LOG_DIR, folder, today)
    os.makedirs(target_path, exist_ok=True)
    return target_path

# Initialize Database & Recognition Engines
db_manager = DatabaseManager()
recognizer = FaceRecognizer()

# Load existing visitors from DB for persistent re-identification
db_manager.repair_missing_registration_crops()
registered_visitors = db_manager.load_registered_visitors()
log_system_event(f"Loaded {len(registered_visitors)} pre-registered visitors from database.")

def extract_padded_face(frame, x1, y1, x2, y2, padding_ratio=CROP_PADDING_RATIO):
    """Pads face bounding box safely within image boundaries for higher quality embedding extraction."""
    h, w, _ = frame.shape
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)

    px1 = max(0, x1 - pad_w)
    py1 = max(0, y1 - pad_h)
    px2 = min(w, x2 + pad_w)
    py2 = min(h, y2 + pad_h)

    crop = frame[py1:py2, px1:px2]
    return crop

def match_or_register_visitor(face_crop):
    """
    Matches face against known visitors using cosine similarity or registers a new visitor.
    Returns (visitor_id, embedding, is_new)
    """
    if face_crop is None or face_crop.size == 0:
        return None, None, False

    embedding = recognizer.extract_embedding(face_crop)
    if embedding is None:
        return None, None, False
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        log_system_event("[RECOGNITION] Ignored non-finite face embedding.")
        return None, None, False

    log_system_event(f"[EMBEDDING] Generated 512-D face embedding vector (norm: {float(np.linalg.norm(embedding)):.2f})")

    best_match_id = None
    max_sim = -1.0

    for visitor in registered_visitors:
        sim = recognizer.compute_similarity(embedding, visitor['embedding'])
        if sim > max_sim:
            max_sim = sim
            best_match_id = visitor['visitor_id']

    if max_sim >= SIMILARITY_THRESHOLD:
        log_system_event(f"[RE-ID] Recognized visitor {best_match_id} (Cosine Similarity: {max_sim:.3f} >= {SIMILARITY_THRESHOLD})")
        registration_path = db_manager.get_registration_crop_path(best_match_id)
        if registration_path:
            registration_absolute_path = os.path.join(ROOT_DIR, registration_path.replace("/", os.sep))
        else:
            registration_absolute_path = os.path.join(
                get_crop_dir("REGISTRATION"),
                f"{best_match_id}_registered.jpg",
            )

        if not os.path.isfile(registration_absolute_path):
            os.makedirs(os.path.dirname(registration_absolute_path), exist_ok=True)
            if face_crop.size > 0 and cv2.imwrite(registration_absolute_path, face_crop):
                stored_path = os.path.relpath(registration_absolute_path, ROOT_DIR)
                db_manager.update_registration_crop_path(best_match_id, stored_path)
                log_system_event(
                    f"[REGISTRATION] Restored registration face crop for {best_match_id} "
                    f"to: {stored_path.replace(os.sep, '/')}"
                )
            else:
                log_system_event(
                    f"[ERROR] Could not restore registration crop for {best_match_id}."
                )
        return best_match_id, embedding, False

    # Auto-register new unique visitor
    visitor_numbers = [
        int(visitor["visitor_id"].rsplit("_", 1)[1])
        for visitor in registered_visitors
        if visitor.get("visitor_id", "").startswith("VISITOR_")
        and visitor["visitor_id"].rsplit("_", 1)[-1].isdigit()
    ]
    next_number = max(visitor_numbers, default=0) + 1
    new_visitor_id = f"VISITOR_{next_number:04d}"
    today_crops = get_crop_dir("REGISTRATION")
    crop_filename = f"{new_visitor_id}_registered.jpg"
    crop_path = os.path.join(today_crops, crop_filename)
    if not cv2.imwrite(crop_path, face_crop) or not os.path.isfile(crop_path):
        log_system_event(
            f"[ERROR] Registration crop could not be saved for {new_visitor_id}; "
            "visitor was not registered."
        )
        return None, None, False

    log_system_event(f"[REGISTRATION] New face detected (Max Sim: {max_sim:.3f}). Assigning unique ID: {new_visitor_id}")
    if not db_manager.register_visitor(new_visitor_id, embedding, crop_path):
        log_system_event(
            f"[ERROR] Database registration failed for {new_visitor_id}; "
            "visitor was not added to the in-memory registry."
        )
        return None, None, False
    registered_visitors.append({'visitor_id': new_visitor_id, 'embedding': embedding})
    log_system_event(
        f"[REGISTRATION] Saved registration face crop to: "
        f"{os.path.relpath(crop_path, ROOT_DIR).replace(os.sep, '/')}"
    )
    return new_visitor_id, embedding, True

# Load YOLO face detection model
yolo_model = YOLO(MODEL_PATH)
run_seen_visitor_ids = set()

def process_stream(source_path, source_name="stream", is_live=False, max_frames=None):
    """Processes video file or RTSP stream with detection, ByteTrack tracking, re-ID, and entry/exit logging."""
    global stop_requested
    stop_requested = False
    # Do not carry ByteTrack identities from one independent video into another.
    yolo_model.predictor = None
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        log_system_event(f"ERROR: Unable to open input stream: {source_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1
    if max_frames and total_frames > max_frames:
        total_frames = max_frames

    output_filename = f"{os.path.splitext(source_name)[0]}_pipeline.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_to_visitor = {}
    recognition_attempts = {}
    active_tracks = {}
    frame_number = 0
    start_time = time.time()
    total_detections = 0
    total_entries = 0
    total_exits = 0
    stream_seen_visitor_ids = set()
    active_visitor_ids = set()

    log_system_event(f"Starting processing for '{source_name}' ({width}x{height} @ {fps:.1f} FPS)")

    while True:
        if stop_requested:
            break
        if max_frames and frame_number >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            if is_live:
                time.sleep(0.05)
                continue
            break

        frame_number += 1

        timestamp = round(frame_number / fps, 3)

        # Run detection/tracking on the configured cadence. A value of 1
        # preserves full-rate processing; larger values process every Nth frame.
        results = []
        if frame_number % max(1, FRAME_SKIP) == 0:
            results = yolo_model.track(
                frame,
                conf=CONFIDENCE,
                imgsz=INFERENCE_SIZE,
                device=DEVICE,
                persist=True,
                tracker=TRACKER,
                verbose=False
            )

        boxes = results[0].boxes if results else None
        if boxes is not None and len(boxes) > 0:
            total_detections += len(boxes)
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                track_id = int(boxes.id[i].item()) if boxes.id is not None else -1

                if track_id == -1:
                    continue

                face_crop = extract_padded_face(frame, x1, y1, x2, y2)

                # Track-to-visitor identification. Retry low-quality crops instead
                # of creating an event with a synthetic TRACK_* identifier.
                visitor_id = track_to_visitor.get(track_id)
                last_attempt = recognition_attempts.get(track_id, -RETRY_FRAMES)
                if visitor_id is None and frame_number - last_attempt >= RETRY_FRAMES:
                    recognition_attempts[track_id] = frame_number
                    visitor_id, _, is_new = match_or_register_visitor(face_crop)
                    if visitor_id:
                        track_to_visitor[track_id] = visitor_id
                        first_stream_observation = visitor_id not in active_visitor_ids
                        stream_seen_visitor_ids.add(visitor_id)
                        run_seen_visitor_ids.add(visitor_id)
                        active_visitor_ids.add(visitor_id)
                        if first_stream_observation:
                            total_entries += 1

                        if first_stream_observation:
                            # Save Entry Crop in logs/entries/YYYY-MM-DD/
                            entry_dir = get_crop_dir("ENTRY")
                            entry_crop_name = f"{visitor_id}_entry.jpg"
                            entry_crop_path = os.path.join(entry_dir, entry_crop_name)
                            if face_crop.size > 0 and cv2.imwrite(entry_crop_path, face_crop):
                                persisted = db_manager.log_event(
                                    source_name,
                                    frame_number,
                                    timestamp,
                                    visitor_id,
                                    "ENTRY",
                                    entry_crop_path,
                                )
                                if persisted:
                                    log_system_event(
                                        f"[ENTRY EVENT] {visitor_id} entered frame at "
                                        f"{timestamp}s (Track ID: {track_id})"
                                    )
                            else:
                                log_system_event(
                                    f"[ERROR] Could not save ENTRY crop for {visitor_id}; "
                                    "ENTRY event was not persisted."
                                )

                if visitor_id is None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        frame,
                        f"Recognizing (ID:{track_id})",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )
                    continue

                active_tracks[track_id] = {
                    'visitor_id': visitor_id,
                    'last_frame': frame_number,
                    'last_crop': face_crop,
                    'last_timestamp': timestamp
                }

                # Visual bounding box and label
                color = (0, 255, 0) if "VISITOR" in str(visitor_id) else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{visitor_id} (ID:{track_id})"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Detect track disappearance (Exit Event)
        disappeared_ids = [
            tid for tid, data in active_tracks.items()
            if (frame_number - data['last_frame']) > EXIT_TIMEOUT_FRAMES
        ]

        for tid in disappeared_ids:
            data = active_tracks.pop(tid)
            v_id = data['visitor_id']
            if any(
                other_data['visitor_id'] == v_id
                for other_data in active_tracks.values()
            ):
                continue
            active_visitor_ids.discard(v_id)
            total_exits += 1

            exit_dir = get_crop_dir("EXIT")
            exit_crop_name = f"{v_id}_exit.jpg"
            exit_crop_path = os.path.join(exit_dir, exit_crop_name)
            crop_saved = (
                data['last_crop'] is not None
                and data['last_crop'].size > 0
                and cv2.imwrite(exit_crop_path, data['last_crop'])
            )
            if crop_saved and db_manager.log_event(
                source_name, data['last_frame'], data['last_timestamp'], v_id, "EXIT", exit_crop_path
            ):
                log_system_event(f"[EXIT EVENT] {v_id} exited frame at {data['last_timestamp']}s after inactivity")
            else:
                log_system_event(f"[ERROR] Could not persist EXIT event for {v_id} after inactivity")

        # Overlay Pipeline Telemetry
        unique_count = len(stream_seen_visitor_ids)
        overlay_text = f"Unique Visitors: {unique_count} | Active: {len(active_tracks)} | Entries: {total_entries} | Exits: {total_exits}"
        cv2.rectangle(frame, (10, 10), (620, 45), (0, 0, 0), -1)
        cv2.putText(frame, overlay_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        writer.write(frame)

        if frame_number % 150 == 0:
            elapsed = time.time() - start_time
            proc_fps = frame_number / elapsed if elapsed > 0 else 0
            pct = f"({(frame_number / total_frames * 100):.1f}%)" if total_frames > 0 else ""
            log_system_event(
                f"[{source_name}] Frame: {frame_number}{pct} | Speed: {proc_fps:.1f} FPS | "
                f"Unique Visitors In Video: {unique_count}"
            )

    # Flush remaining active tracks on video completion as exits
    stream_end_exited_ids = set()
    for tid, data in list(active_tracks.items()):
        v_id = data['visitor_id']
        if v_id in stream_end_exited_ids:
            continue
        stream_end_exited_ids.add(v_id)
        active_visitor_ids.discard(v_id)
        total_exits += 1
        exit_dir = get_crop_dir("EXIT")
        exit_crop_name = f"{v_id}_exit.jpg"
        exit_crop_path = os.path.join(exit_dir, exit_crop_name)
        crop_saved = (
            data['last_crop'] is not None
            and data['last_crop'].size > 0
            and cv2.imwrite(exit_crop_path, data['last_crop'])
        )
        if crop_saved and db_manager.log_event(
            source_name, data['last_frame'], data['last_timestamp'], v_id, "EXIT", exit_crop_path
        ):
            log_system_event(f"[EXIT EVENT (STREAM END)] {v_id} logged exit at stream conclusion")
        else:
            log_system_event(f"[ERROR] Could not persist final EXIT event for {v_id}")

    cap.release()
    writer.release()
    total_time = time.time() - start_time
    avg_fps = frame_number / total_time if total_time > 0 else 0
    log_system_event(f"Completed {source_name}: {frame_number} frames in {total_time:.2f}s (Avg {avg_fps:.1f} FPS)")
    log_system_event(
        f"Summary for {source_name}: Entries={total_entries}, Exits={total_exits}, "
        f"Detections={total_detections}, Unique Visitors In Video={len(stream_seen_visitor_ids)}"
    )

def main():
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Path to single video or folder (overrides config)")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP Stream URL")
    parser.add_argument("--webcam", action="store_true", help="Use default webcam (Device 0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process per stream")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, request_stop)

    if args.rtsp or (SOURCE_TYPE == "rtsp" and RTSP_URL):
        stream_url = args.rtsp if args.rtsp else RTSP_URL
        log_system_event(f"Connecting to live RTSP camera stream: {stream_url}")
        process_stream(stream_url, source_name="rtsp_live_stream", is_live=True, max_frames=args.max_frames)
    elif args.webcam or SOURCE_TYPE == "webcam":
        log_system_event("Opening default webcam (Camera 0)...")
        process_stream(0, source_name="webcam_live", is_live=True, max_frames=args.max_frames)
    else:
        # Determine source
        target_source = args.source if args.source else INPUT_SOURCE
        target_path = os.path.join(ROOT_DIR, target_source) if not os.path.isabs(target_source) else target_source

        if os.path.isfile(target_path):
            log_system_event(f"Processing single video file: {target_path}")
            v_name = os.path.basename(target_path)
            process_stream(target_path, source_name=v_name, is_live=False, max_frames=args.max_frames)
        elif os.path.isdir(target_path):
            video_files = sorted([f for f in os.listdir(target_path) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])
            log_system_event(f"Found {len(video_files)} video file(s) in '{target_path}'")

            for idx, v_name in enumerate(video_files, 1):
                log_system_event(f"--- Processing Video ({idx}/{len(video_files)}): {v_name} ---")
                v_path = os.path.join(target_path, v_name)
                process_stream(v_path, source_name=v_name, is_live=False, max_frames=args.max_frames)
                if stop_requested:
                    break
        else:
            log_system_event(f"ERROR: Video source path does not exist: {target_path}")
            return

    total_visitors = len(registered_visitors)
    log_system_event("============================================================")
    log_system_event(
        f"PIPELINE RUN FINISHED. UNIQUE VISITORS SEEN IN THIS RUN: "
        f"{len(run_seen_visitor_ids)}"
    )
    log_system_event(
        f"TOTAL REGISTERED VISITORS IN DATABASE: {total_visitors}"
    )
    log_system_event(f"Database Record Count: {db_manager.get_visitor_count()}")
    log_system_event(f"============================================================")
    event_logger.close()

if __name__ == "__main__":
    main()