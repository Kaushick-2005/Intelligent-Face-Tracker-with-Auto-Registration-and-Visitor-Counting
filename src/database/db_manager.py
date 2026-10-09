import os
import json
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# Project root & config setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
else:
    config = {}

DB_CONFIG = config.get("database", {})
MONGODB_URI = DB_CONFIG.get("mongodb_uri", "mongodb://localhost:27017/")
DB_NAME = DB_CONFIG.get("database_name", "intelligent_face_tracker")
SQLITE_PATH = os.path.join(ROOT_DIR, DB_CONFIG.get("sqlite_path", "database/intelligent_face_tracker.sqlite3"))


class DatabaseManager:
    def __init__(self, uri=MONGODB_URI, db_name=DB_NAME, sqlite_path=SQLITE_PATH):
        """
        Initializes DatabaseManager with dual engine support:
        1. MongoDB (Atlas or Local)
        2. SQLite3 (Local persistent fallback & guaranteed offline operations)
        """
        self.client = None
        self.mongo_db = None
        self.visitors_col = None
        self.events_col = None
        self.sqlite_path = sqlite_path

        # Initialize SQLite first to guarantee local resilience
        self._init_sqlite()
        self._normalize_sqlite_paths()

        # Connect to MongoDB
        try:
            import pymongo
            self.client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
            self.client.server_info()  # Validate connection
            self.mongo_db = self.client[db_name]
            self.visitors_col = self.mongo_db["visitors"]
            self.events_col = self.mongo_db["events"]
            self._ensure_mongo_collections()
            self._normalize_mongo_paths()
            print(f"[INFO] Connected successfully to MongoDB: '{db_name}'")
        except Exception as e:
            print(f"[WARNING] MongoDB connection failed ({e}). Using SQLite3 local engine.")
            self.client = None
            self.mongo_db = None
            self.visitors_col = None
            self.events_col = None

    def _ensure_mongo_collections(self):
        """Ensure both persistence collections exist before the first event."""
        existing = set(self.mongo_db.list_collection_names())
        for collection_name in ("visitors", "events"):
            if collection_name not in existing:
                self.mongo_db.create_collection(collection_name)

        self.visitors_col.create_index("visitor_id", unique=True)
        self.events_col.create_index(
            [("video", 1), ("frame", 1), ("visitor_id", 1), ("event_type", 1)],
            unique=True,
        )

    def _init_sqlite(self):
        """Creates SQLite tables if they do not exist."""
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    visitor_id TEXT PRIMARY KEY,
                    registered_at TEXT,
                    crop_image_path TEXT,
                    embedding TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video TEXT,
                    frame INTEGER,
                    timestamp REAL,
                    visitor_id TEXT,
                    event_type TEXT,
                    crop_path TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def _normalize_sqlite_paths(self):
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT id, crop_path FROM events WHERE crop_path IS NOT NULL"
            ).fetchall()
            for event_id, path in rows:
                normalized = self._portable_path(path)
                if normalized != path:
                    conn.execute(
                        "UPDATE events SET crop_path = ? WHERE id = ?",
                        (normalized, event_id),
                    )

            rows = conn.execute(
                "SELECT visitor_id, crop_image_path FROM visitors "
                "WHERE crop_image_path IS NOT NULL"
            ).fetchall()
            for visitor_id, path in rows:
                normalized = self._portable_path(path)
                if normalized != path:
                    conn.execute(
                        "UPDATE visitors SET crop_image_path = ? WHERE visitor_id = ?",
                        (normalized, visitor_id),
                    )
            conn.commit()

    def register_visitor(self, visitor_id, embedding, crop_path):
        """
        Registers a new unique visitor with facial embedding and crop image path.
        Synchronizes to both MongoDB (if available) and SQLite.
        """
        emb_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        crop_path = self._portable_path(crop_path)
        crop_absolute_path = os.path.join(ROOT_DIR, crop_path)
        if not os.path.isfile(crop_absolute_path):
            print(f"[ERROR] Registration crop does not exist: {crop_absolute_path}")
            return False
        now_iso = datetime.now().isoformat()
        sqlite_written = False
        mongo_written = False

        # 1. Save to SQLite
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO visitors (visitor_id, registered_at, crop_image_path, embedding)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(visitor_id) DO UPDATE SET
                        registered_at = excluded.registered_at,
                        crop_image_path = excluded.crop_image_path,
                        embedding = excluded.embedding
                """, (visitor_id, now_iso, crop_path, json.dumps(emb_list)))
                conn.commit()
                sqlite_written = True
        except Exception as err:
            print(f"[ERROR] SQLite register_visitor failed: {err}")

        # 2. Save to MongoDB
        if self.visitors_col is not None:
            try:
                doc = {
                    "visitor_id": visitor_id,
                    "registered_at": now_iso,
                    "crop_image_path": crop_path,
                    "embedding": emb_list
                }
                self.visitors_col.replace_one({"visitor_id": visitor_id}, doc, upsert=True)
                print(f"[DB:MongoDB] Registered visitor: {visitor_id}")
                mongo_written = True
            except Exception as e:
                print(f"[WARNING] MongoDB write failed: {e}")

        print(f"[DB] Visitor successfully stored: {visitor_id}")
        return sqlite_written or mongo_written

    def get_registration_crop_path(self, visitor_id):
        """Return the stored registration crop path for a visitor, if available."""
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute(
                "SELECT crop_image_path FROM visitors WHERE visitor_id = ?",
                (visitor_id,),
            ).fetchone()
        if row and row[0]:
            return self._portable_path(row[0])

        if self.visitors_col is not None:
            try:
                document = self.visitors_col.find_one(
                    {"visitor_id": visitor_id},
                    {"crop_image_path": 1},
                )
                if document and document.get("crop_image_path"):
                    return self._portable_path(document["crop_image_path"])
            except Exception as err:
                print(f"[WARNING] Could not read registration crop path for {visitor_id}: {err}")
        return None

    def update_registration_crop_path(self, visitor_id, crop_path):
        """Persist a corrected registration crop path in the available stores."""
        portable_path = self._portable_path(crop_path)
        sqlite_written = False
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute(
                    "UPDATE visitors SET crop_image_path = ? WHERE visitor_id = ?",
                    (portable_path, visitor_id),
                )
                conn.commit()
                sqlite_written = cursor.rowcount > 0
        except Exception as err:
            print(f"[ERROR] SQLite registration path update failed: {err}")

        if self.visitors_col is not None:
            try:
                self.visitors_col.update_one(
                    {"visitor_id": visitor_id},
                    {"$set": {"crop_image_path": portable_path}},
                )
                sqlite_written = True
            except Exception as err:
                print(f"[WARNING] MongoDB registration path update failed: {err}")
        return sqlite_written

    def repair_missing_registration_crops(self):
        """Restore missing registration files from available entry crops."""
        visitor_paths = {}
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT visitor_id, crop_image_path FROM visitors "
                "WHERE crop_image_path IS NOT NULL"
            ).fetchall()
        visitor_paths.update(rows)

        if self.visitors_col is not None:
            try:
                mongo_rows = self.visitors_col.find(
                    {"visitor_id": {"$type": "string"}, "crop_image_path": {"$type": "string"}},
                    {"visitor_id": 1, "crop_image_path": 1},
                )
                visitor_paths.update(
                    (doc["visitor_id"], doc["crop_image_path"]) for doc in mongo_rows
                )
            except Exception as err:
                print(f"[WARNING] Could not inspect MongoDB registration crops: {err}")

        repaired = 0
        for visitor_id, crop_image_path in visitor_paths.items():
            portable_path = self._portable_path(crop_image_path)
            expected_path = os.path.join(ROOT_DIR, portable_path.replace("/", os.sep))
            if os.path.isfile(expected_path):
                continue

            entry_candidates = sorted(
                Path(ROOT_DIR).glob(f"logs/entries/*/{visitor_id}_entry.jpg")
            )
            if not entry_candidates:
                continue

            target_dir = os.path.join(ROOT_DIR, "logs", "registrations", datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, f"{visitor_id}_registered.jpg")
            try:
                shutil.copyfile(entry_candidates[-1], target_path)
                portable_target = self._portable_path(target_path)
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute(
                        "UPDATE visitors SET crop_image_path = ? WHERE visitor_id = ?",
                        (portable_target, visitor_id),
                    )
                    conn.commit()
                if self.visitors_col is not None:
                    try:
                        self.visitors_col.update_one(
                            {"visitor_id": visitor_id},
                            {"$set": {"crop_image_path": portable_target}},
                        )
                    except Exception as err:
                        print(
                            f"[WARNING] Could not update MongoDB crop path for "
                            f"{visitor_id}: {err}"
                        )
                repaired += 1
            except OSError as err:
                print(f"[WARNING] Could not repair registration crop for {visitor_id}: {err}")

        if repaired:
            print(f"[DB] Repaired {repaired} missing registration crop(s).")
        return repaired

    def log_event(self, video_name, frame, timestamp, visitor_id, event_type, crop_path):
        """
        Logs an ENTRY or EXIT event with timestamp and crop path into DB.
        """
        crop_path = self._portable_path(crop_path)
        now_iso = datetime.now().isoformat()
        sqlite_written = False
        mongo_written = False

        # 1. Save to SQLite
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (video, frame, timestamp, visitor_id, event_type, crop_path, created_at)
                    SELECT ?, ?, ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM events
                        WHERE video = ? AND frame = ? AND visitor_id = ? AND event_type = ?
                    )
                """, (
                    video_name,
                    frame,
                    timestamp,
                    visitor_id,
                    event_type,
                    crop_path,
                    now_iso,
                    video_name,
                    frame,
                    visitor_id,
                    event_type,
                ))
                conn.commit()
                sqlite_written = True
        except Exception as err:
            print(f"[ERROR] SQLite log_event failed: {err}")

        # 2. Save to MongoDB
        if self.events_col is not None:
            try:
                doc = {
                    "video": video_name,
                    "frame": frame,
                    "timestamp": timestamp,
                    "visitor_id": visitor_id,
                    "event_type": event_type,
                    "crop_path": crop_path,
                    "created_at": now_iso
                }
                result = self.events_col.update_one(
                    {
                        "video": video_name,
                        "frame": frame,
                        "visitor_id": visitor_id,
                        "event_type": event_type,
                    },
                    {"$setOnInsert": doc},
                    upsert=True,
                )
                if result.upserted_id is not None:
                    print(f"[DB:MongoDB] Logged {event_type} for {visitor_id} at frame {frame}")
                mongo_written = True
            except Exception as e:
                print(f"[WARNING] MongoDB log_event failed: {e}")

        if sqlite_written or mongo_written:
            print(f"[DB] Logged {event_type} event for {visitor_id} in {video_name} (frame {frame})")
        else:
            print(f"[ERROR] Could not persist {event_type} event for {visitor_id} in {video_name}")
        return sqlite_written or mongo_written

    def load_registered_visitors(self):
        """
        Retrieves all registered visitors and embeddings for face re-identification across runs.
        Attempts MongoDB first, falling back to SQLite.
        Returns list of dicts: [{'visitor_id': id, 'embedding': np.ndarray}]
        """
        visitors = []
        # Try loading from MongoDB Atlas first
        if self.visitors_col is not None:
            try:
                for doc in self.visitors_col.find():
                    v_id = doc.get("visitor_id")
                    emb_raw = doc.get("embedding")
                    if v_id and emb_raw:
                        visitors.append({
                            "visitor_id": v_id,
                            "embedding": np.array(emb_raw, dtype=np.float32)
                        })
                if visitors:
                    print(f"[DB:MongoDB] Loaded {len(visitors)} visitors from MongoDB.")
                    return visitors
            except Exception as e:
                print(f"[WARNING] MongoDB load visitors failed: {e}. Falling back to SQLite.")

        # Fallback to local SQLite
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT visitor_id, embedding FROM visitors")
                rows = cursor.fetchall()
                for v_id, emb_json in rows:
                    if emb_json:
                        emb = np.array(json.loads(emb_json), dtype=np.float32)
                        visitors.append({"visitor_id": v_id, "embedding": emb})
            print(f"[DB:SQLite] Loaded {len(visitors)} visitors from SQLite.")
            self._sync_visitors_to_mongo()
        except Exception as e:
            print(f"[ERROR] Failed to load visitors from SQLite: {e}")
        return visitors

    @staticmethod
    def _portable_path(path):
        """Store paths relative to the project instead of machine-specific absolute paths."""
        if not path:
            return path
        if not os.path.isabs(path):
            return path.replace("\\", "/")
        try:
            relative = os.path.relpath(path, ROOT_DIR)
            if not relative.startswith(".."):
                return relative.replace(os.sep, "/")
        except (TypeError, ValueError):
            pass
        return path

    def _sync_visitors_to_mongo(self):
        """Backfill MongoDB from SQLite when MongoDB was initially empty."""
        if self.visitors_col is None:
            return

        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT visitor_id, registered_at, crop_image_path, embedding FROM visitors"
            ).fetchall()

        for visitor_id, registered_at, crop_image_path, embedding_json in rows:
            if not embedding_json:
                continue
            self.visitors_col.update_one(
                {"visitor_id": visitor_id},
                {
                    "$setOnInsert": {
                        "visitor_id": visitor_id,
                        "registered_at": registered_at,
                        "crop_image_path": self._portable_path(crop_image_path),
                        "embedding": json.loads(embedding_json),
                    }
                },
                upsert=True,
            )

    def _normalize_mongo_paths(self):
        if self.visitors_col is None or self.events_col is None:
            return

        for collection, path_field in (
            (self.visitors_col, "crop_image_path"),
            (self.events_col, "crop_path"),
        ):
            for document in collection.find(
                {path_field: {"$type": "string"}}, {"_id": 1, path_field: 1}
            ):
                path = document.get(path_field)
                normalized = self._portable_path(path)
                if normalized != path:
                    collection.update_one(
                        {"_id": document["_id"]},
                        {"$set": {path_field: normalized}},
                    )

    def get_visitor_count(self):
        """Returns total unique visitor count from SQLite or MongoDB."""
        count = 0
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM visitors")
                count = cursor.fetchone()[0]
        except Exception:
            count = 0

        if count == 0 and self.visitors_col is not None:
            try:
                count = self.visitors_col.count_documents({})
            except Exception:
                pass
        return count


if __name__ == "__main__":
    db_mgr = DatabaseManager()
    print("Database Manager test completed. Unique visitors in DB:", db_mgr.get_visitor_count())