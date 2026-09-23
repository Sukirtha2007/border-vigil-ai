"""
Event Logger — persists all detection events (crossings, suspicious activities)
to a SQLite database with alert snapshots saved as JPEG images.

Features:
  - Timestamped event logging with full metadata
  - Auto-save frame snapshot at alert time
  - Query API for event history
  - Thread-safe writes
"""

import sqlite3
import os
import json
import time
import cv2
import numpy as np
from datetime import datetime
import threading


DB_NAME = "ibvap_events.db"
SNAPSHOT_DIR = "alerts"


class EventLogger:
    def __init__(self, db_path: str = DB_NAME, snapshot_dir: str = SNAPSHOT_DIR):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        self._lock = threading.Lock()

        # Create snapshot directory
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize database
        self._init_db()

        # Counters (in-memory for fast access)
        self.total_events = 0

    def _init_db(self):
        """Create the events table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime_str TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    track_id INTEGER,
                    class_name TEXT,
                    location_x REAL,
                    location_y REAL,
                    direction TEXT,
                    details TEXT,
                    duration REAL DEFAULT 0,
                    snapshot_path TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_severity ON events(severity)
            """)
            conn.commit()

    def _save_snapshot(self, frame: np.ndarray, event_type: str,
                       track_id: int, timestamp: float) -> str:
        """Save a JPEG snapshot of the frame at alert time."""
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{event_type}_{track_id}_{dt_str}.jpg"
        filepath = os.path.join(self.snapshot_dir, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filepath

    def log_crossing(self, event, frame: np.ndarray | None = None):
        """Log a boundary crossing event."""
        snapshot_path = ""
        if frame is not None:
            snapshot_path = self._save_snapshot(
                frame, event.direction, event.track_id, event.timestamp
            )

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO events
                    (timestamp, datetime_str, event_type, severity, track_id, class_name,
                     location_x, location_y, direction, details, snapshot_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp,
                    datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                    "BOUNDARY_CROSSING",
                    "CRITICAL",
                    event.track_id,
                    event.class_name,
                    event.location[0],
                    event.location[1],
                    event.direction,
                    f"{event.class_name} #{event.track_id} — {event.direction}",
                    snapshot_path
                ))
                conn.commit()
            self.total_events += 1

    def log_suspicious(self, event, frame: np.ndarray | None = None):
        """Log a suspicious activity event."""
        snapshot_path = ""
        if frame is not None:
            snapshot_path = self._save_snapshot(
                frame, event.event_type, event.track_id, event.timestamp
            )

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO events
                    (timestamp, datetime_str, event_type, severity, track_id, class_name,
                     location_x, location_y, details, duration, snapshot_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp,
                    datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                    event.event_type,
                    event.severity,
                    event.track_id,
                    event.class_name,
                    event.location[0],
                    event.location[1],
                    event.details,
                    event.duration,
                    snapshot_path
                ))
                conn.commit()
            self.total_events += 1

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get the most recent events."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_events_by_type(self, event_type: str, limit: int = 50) -> list[dict]:
        """Get events filtered by type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            total = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type"
            )
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            cursor = conn.execute(
                "SELECT severity, COUNT(*) as count FROM events GROUP BY severity"
            )
            by_severity = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "total_events": total,
            "by_type": by_type,
            "by_severity": by_severity,
        }
