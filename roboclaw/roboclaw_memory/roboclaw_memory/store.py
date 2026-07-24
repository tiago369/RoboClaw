"""Backend SQLite para memória episódica robótica."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is not None: return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception: pass
    return _MODEL

@dataclass
class Episode:
    id: int
    subtask: str
    outcome: str
    env_state: dict
    timestamp: float
    task_id: str
    attempt: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_context_string(self) -> str:
        return f"[{self.subtask}] outcome={self.outcome} attempt={self.attempt} env={json.dumps(self.env_state, ensure_ascii=False)}"

class EpisodeStore:
    def __init__(self, db_path: str = "~/.roboclaw/memory.db", embedding_model=None):
        if db_path == ":memory:":
            self._path = ":memory:"
        else:
            p = Path(db_path).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(p)
        self._model = embedding_model if embedding_model is not None else _get_model()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subtask TEXT NOT NULL, outcome TEXT NOT NULL,
                env_state TEXT NOT NULL, timestamp REAL NOT NULL,
                task_id TEXT NOT NULL DEFAULT '', attempt INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS embeddings (
                episode_id INTEGER PRIMARY KEY REFERENCES episodes(id), vector BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_task_id ON episodes(task_id);
            CREATE INDEX IF NOT EXISTS idx_subtask  ON episodes(subtask);
        """); self._conn.commit()

    def insert(self, subtask, outcome, env_state, task_id="", attempt=1) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes (subtask,outcome,env_state,timestamp,task_id,attempt) VALUES (?,?,?,?,?,?)",
            (subtask, outcome, json.dumps(env_state), time.time(), task_id, attempt))
        eid = cur.lastrowid; self._conn.commit()
        if self._model is not None:
            vec = self._model.encode(f"{subtask} {outcome} {json.dumps(env_state)}", convert_to_numpy=True).astype(np.float32)
            self._conn.execute("INSERT INTO embeddings (episode_id,vector) VALUES (?,?)", (eid, vec.tobytes())); self._conn.commit()
        return eid

    def get(self, eid) -> Optional[Episode]:
        row = self._conn.execute("SELECT * FROM episodes WHERE id=?", (eid,)).fetchone()
        return self._row(row) if row else None

    def get_by_task(self, task_id) -> list[Episode]:
        return [self._row(r) for r in self._conn.execute("SELECT * FROM episodes WHERE task_id=? ORDER BY timestamp", (task_id,)).fetchall()]

    def get_recent(self, limit=20) -> list[Episode]:
        return [self._row(r) for r in self._conn.execute("SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()]

    def delete(self, eid):
        self._conn.execute("DELETE FROM embeddings WHERE episode_id=?", (eid,))
        self._conn.execute("DELETE FROM episodes WHERE id=?", (eid,)); self._conn.commit()

    def search(self, query, top_k=3) -> list[tuple[Episode, float]]:
        if self._model is None: return self._fallback(query, top_k)
        qv = self._model.encode(query, convert_to_numpy=True).astype(np.float32)
        rows = self._conn.execute("SELECT e.*,b.vector FROM episodes e JOIN embeddings b ON b.episode_id=e.id").fetchall()
        if not rows: return []
        scored = []
        for r in rows:
            v = np.frombuffer(r["vector"], dtype=np.float32)
            s = float(np.dot(qv,v)/(np.linalg.norm(qv)*np.linalg.norm(v)+1e-9))
            scored.append((self._row(r), s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _fallback(self, query, top_k):
        words = [w for w in query.lower().split() if len(w)>3] or [query]
        seen, res = set(), []
        for w in words:
            for r in self._conn.execute("SELECT * FROM episodes WHERE lower(subtask) LIKE ? LIMIT ?", (f"%{w}%", top_k)).fetchall():
                if r["id"] not in seen:
                    seen.add(r["id"]); res.append((self._row(r), 1.0))
            if len(res)>=top_k: break
        return res[:top_k]

    def _row(self, row) -> Episode:
        return Episode(id=row["id"], subtask=row["subtask"], outcome=row["outcome"],
                       env_state=json.loads(row["env_state"]), timestamp=row["timestamp"],
                       task_id=row["task_id"], attempt=row["attempt"])

    def close(self): self._conn.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
