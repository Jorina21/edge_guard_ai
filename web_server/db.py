# db.py – SQLite simple wrapper

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "events.db")


def init_db():
    """Creates DB + table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS motion_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            person_count INTEGER,
            confidence REAL,
            fps REAL
        )
        """
    )

    conn.commit()
    conn.close()


def log_event(person_count, confidence, fps):
    """Insert new event into DB."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        """
        INSERT INTO motion_events (timestamp, person_count, confidence, fps)
        VALUES (?, ?, ?, ?)
        """,
        (ts, person_count, confidence, fps),
    )

    conn.commit()
    conn.close()


def get_latest_events(limit=20):
    """Return latest N events."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT timestamp, person_count, confidence, fps
        FROM motion_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()
    return rows
