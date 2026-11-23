"""Vehicle register (SQLite): one row per car visit + event log + parking slots."""

import csv
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT, plate_conf REAL,
    car_name TEXT DEFAULT '',
    status TEXT,              -- SERVICE / STAFF / PARKING / EXITED
    -- main gate entry
    entry_time TEXT, entry_cam TEXT, entry_photo TEXT, entry_plate_photo TEXT,
    -- staff line
    staff_time TEXT,
    -- parking
    parking_time TEXT, parking_slot TEXT, parking_photo TEXT, parking_plate_photo TEXT,
    -- exit
    exit_time TEXT, exit_photo TEXT, exit_plate_photo TEXT, stay_seconds REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT, camera TEXT, event TEXT, entry_id INTEGER, detail TEXT
);
CREATE TABLE IF NOT EXISTS slots (
    name TEXT PRIMARY KEY, camera TEXT,
    occupied INTEGER, since TEXT, plate TEXT, photo TEXT, entry_id INTEGER
);
"""


def plate_distance(a: str, b: str) -> int:
    """Levenshtein distance between two plate strings."""
    if not a or not b:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Registry:
    def __init__(self, db_path: Path, fresh=True):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and db_path.exists():
            db_path.unlink()
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        ver = self.db.execute("PRAGMA user_version").fetchone()[0]
        if ver != SCHEMA_VERSION:              # old test database -> rebuild
            self.db.executescript(
                "DROP TABLE IF EXISTS entries; DROP TABLE IF EXISTS events; "
                "DROP TABLE IF EXISTS slots;")
            self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.db.executescript(SCHEMA)

    # ---------- entries ----------
    def add_entry(self, status, plate, conf, **fields):
        fields.update(status=status, plate=plate, plate_conf=conf)
        cols = ", ".join(fields)
        q = ", ".join("?" * len(fields))
        cur = self.db.execute(f"INSERT INTO entries ({cols}) VALUES ({q})",
                              tuple(fields.values()))
        self.db.commit()
        return cur.lastrowid

    def update(self, entry_id, **fields):
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE entries SET {sets} WHERE id=?",
                        (*fields.values(), entry_id))
        self.db.commit()

    def find_plate(self, plate, max_dist=2, active_only=True):
        """Most recent matching entry (tolerates 1-2 OCR character errors)."""
        if not plate or plate == "UNREAD":
            return None
        best = None
        for row in self.entries():
            if active_only and row["status"] == "EXITED":
                continue
            d = plate_distance(plate, row["plate"])
            if d <= max_dist and (best is None or d < best[0] or
                                  (d == best[0] and row["id"] > best[1]["id"])):
                best = (d, row)
        return best[1] if best else None

    def entries(self, status=None):
        q = "SELECT * FROM entries" + (" WHERE status=?" if status else "") + " ORDER BY id"
        return [dict(r) for r in self.db.execute(q, (status,) if status else ())]

    def get(self, entry_id):
        r = self.db.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return dict(r) if r else None

    # ---------- events ----------
    def log(self, time, camera, event, entry_id=None, detail=""):
        self.db.execute(
            "INSERT INTO events (time, camera, event, entry_id, detail) VALUES (?,?,?,?,?)",
            (time, camera, event, entry_id, detail))
        self.db.commit()

    def activity(self):
        """entry_id -> list of (time, camera, event) for every followed car."""
        out = {}
        for r in self.db.execute(
                "SELECT time, camera, event, entry_id FROM events "
                "WHERE entry_id IS NOT NULL ORDER BY id"):
            out.setdefault(r["entry_id"], []).append((r["time"], r["camera"], r["event"]))
        return out

    # ---------- parking slots ----------
    def set_slot(self, name, camera, occupied, since="", plate="", photo="", entry_id=None):
        self.db.execute(
            "INSERT OR REPLACE INTO slots (name, camera, occupied, since, plate, photo, entry_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (name, camera, int(occupied), since, plate, photo, entry_id))
        self.db.commit()

    def slots(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM slots ORDER BY name")]

    # ---------- export ----------
    def export_csv(self, path: Path):
        rows = self.entries()
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    def close(self):
        self.db.close()
