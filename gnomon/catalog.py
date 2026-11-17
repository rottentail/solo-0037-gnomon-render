import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .errors import require


def canonical(payload):
    return json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


class Catalog:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root/"gnomon.sqlite3",check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL, details TEXT NOT NULL, created TEXT NOT NULL)")
        self.db.commit()

    def create(self, kind, payload, state="active", details=None):
        raw = canonical(payload)
        identity = kind+"_"+uuid.uuid4().hex[:16]
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self.lock, self.db:
            self.db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?)",(identity,kind,state,raw,digest,canonical(details or {}),datetime.now(timezone.utc).isoformat()))
        return self.get(identity,kind)

    def get(self, identity, kind=None):
        with self.lock:
            row = self.db.execute("SELECT * FROM records WHERE id=?",(identity,)).fetchone()
        require(row is not None, "not_found", "resource does not exist",404)
        result = dict(row)
        require(kind is None or result["kind"] == kind,"resource_kind","resource has a different kind")
        result["payload"] = json.loads(result["payload"])
        result["details"] = json.loads(result["details"])
        return result

    def list(self, kind):
        with self.lock:
            rows = self.db.execute("SELECT id FROM records WHERE kind=? ORDER BY created,id",(kind,)).fetchall()
            return [self.get(row[0],kind) for row in rows]

    def transition(self, identity, expected, target, details=None):
        with self.lock, self.db:
            record = self.get(identity)
            require(record["state"] in expected,"state_conflict",f"cannot change {record['state']} to {target}",409)
            combined = {**record["details"], **(details or {})}
            self.db.execute("UPDATE records SET state=?,details=? WHERE id=?",(target,canonical(combined),identity))
        return self.get(identity)

    def recover_interrupted(self):
        with self.lock, self.db:
            rows = self.db.execute("SELECT id FROM records WHERE state IN ('queued','running','building','measuring','packing')").fetchall()
            for row in rows:
                self.transition(row[0],{"queued","running","building","measuring","packing"},"failed",{"error":{"code":"interrupted","detail":"process ended before completion; submit a new operation"}})
        return len(rows)

    def close(self):
        with self.lock:
            self.db.close()
