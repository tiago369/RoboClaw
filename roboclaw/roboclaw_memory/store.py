"""
store.py
Backend SQLite for the RoboClaw episodic memory.
 
Tables:
  episodes   — each subtask executed with result and context
  embeddings — float32 vector of the episode description
 
Use standalone:
    store = EpisodeStore()
    eid = store.insert("grasp", "failed", {"reason": "slip"}, task_id="t1")
    results = store.search("grasp failure", top_k=3)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from debian import timestamp
import numpy as np

_MODEL = None  # Placeholder for the embedding model, e.g., OpenAI's text-embedding-3-small

def _get_default_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer('sentence_transformers/all-MiniLM-L6-v2')
    except Exception:
        raise ImportError("Please install sentence-transformers to use the default embedding model.")
    return _MODEL

# ---------------------------------------------------------------------------
# Data Structure
# ---------------------------------------------------------------------------

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
        """Convert episode data to a string for embedding."""
        state_str = json.dumps(self.env_state, ensure_ascii=False)
        return (
            f"[{self.subtask}] outcome={self.outcome} "
            f"attempt={self.attempt} env={state_str}"
            )
    
# ---------------------------------------------------------------------------
# Principal Store Class
# ---------------------------------------------------------------------------

class EpisodeStore:
    """
    Manages the persistance and the semantic search of episodic memory for RoboClaw.
    
    Args:
        db_path (str): Path to the SQLite database file. Defaults to 'roboclaw_memory.db'. 
        ":memory:" for temporary in-memory database.
        embedding_model: Optional embedding model. If None, uses a default model from sentence-transformers.

    """

    def __init__(
            self,
            db_path: str = "~/.roboclaw/memory.db",
            embedding_model=None
    ):
        if self.db_path == ":memory:":
            self._path = ":memory:"
        else:
            path = Path(db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(path)
        
        self._model = (
            embedding_model if embedding_model is not None
            else _get_default_model()
        )

        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        # ------------------------------------------------------------------
        # Schema
        # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subtask TEXT NOT NULL,
                outcome TEXT NOT NULL,
                env_state TEXT NOT NULL,
                timestamp REAL NOT NULL,
                task_id TEXT NOT NULL DEFAULT '',
                attempt INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                episode_id INTEGER PRIMARY KEY REFERENCES episodes(id),
                vector BLOB NOT NULL
            );
        
            CREATE INDEX IF NOT EXISTS idx_task_id ON episodes(task_id);
            CREATE INDEX IF NOT EXISTS idx_subtask  ON episodes(subtask);
            CREATE INDEX IF NOT EXISTS idx_outcome  ON episodes(outcome);
        """)
        self._conn.commit()

    def insert(
            self,
            subtask: str,
            outcome: str,
            env_state: dict,
            task_id: str = "",
            attempt: int = 1
    ) -> int:
        """
        Insert a new episode into the store.

        Args:
            subtask (str): The name of the subtask executed.
            outcome (str): The result of the subtask (e.g., "success", "failure").
            env_state (dict): The environment state at the time of execution.
            task_id (str): Optional identifier for the overarching task.
            attempt (int): The attempt number for this subtask.
        """
        cur = self._conn.execute(
            """
            INSERT INTO episodes
                (subtask, outcome, env_state, timestamp, task_id, attempt)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (subtask, outcome, json.dumps(env_state),
             time.time(), task_id, attempt)
        )
        episode_id = cur.lastrowid
        self._conn.commit()

        if self._model is not None:
            text = f"{subtask} {outcome} {json.dumps(env_state)}"
            vec = self._model.encode(
                text, convert_to_numpy=True
            ).astype(np.float32)
            self._conn.execute(
                "INSERT INTO embeddings (episode_id, vector) VALUES (?, ?)",
                (episode_id, vec.tobytes())
            )
            self._conn.commit()
        
        return episode_id
    
    def delete(self, episode_id: int) -> None:
        """
        Delete an episode and its embedding from the store.

        Args:
            episode_id (int): The ID of the episode to delete.
        """
        self._conn.execute("DELETE FROM embeddings WHERE episode_id = ?", (episode_id,))
        self._conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        self._conn.commit()
    
    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get(self, episode_id: int) -> Optional[Episode]:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return self._row_to_episode(row) if row else None
    
    def get_by_task(self, task_id: str) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE task_id = ?", (task_id,),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]
    
    def get_recent(self, limit: int = 10) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]
    
    # ------------------------------------------------------------------
    # Semantic Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[tuple[Episode, float]]:
        """
        Perform a semantic search for episodes similar to the query.
        with embeddings: by cosine similarity.
        without embeddings: by substring fallback in the query words.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
        
        Returns:
            list of tuples (Episode, similarity_score) by descending score.
        """
        if self._model is None:
            return self._search_fallback(query, top_k)
        
        query_vec = self._model.encode(
            query, convert_to_numpy=True
        ).astype(np.float32)

        rows = self._conn.execute(
            """
            SELECT e.*, b.vector
            FROM episodes e
            JOIN embeddings b ON b.episode_id = e.id
            """
        ).fetchall()

        if not rows:
            return []
        
        scored = []
        for row in rows:
            vec = np.frombuffer(row["vector"], dtype=np.float32)
            score = float(
                np.dot(query_vec, vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-9)
            )
            scored.append((self._row_to_episode(row), score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def _search_fallback(
            self, query: str, top_k: int
        ) -> list[tuple[Episode, float]]:
        """Fallback without embeddings: substring search in subtask and outcome."""
        words = [w for w in query.lower().split() if len(w) > 3]
        if not words:
            words = [query]
        
        seen_ids: set[int] = set()
        results: list[tuple[Episode, float]] = []
        for word in words:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE lower(subtask) LIKE ? LIMIT ?",
                (f"%{word}%", top_k),
            ).fetchall()
            for row in rows:
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    results.append((self._row_to_episode(row), 1.0))
            if len(results) >= top_k:
                break
        
        return results[:top_k]
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            subtask=row["subtask"],
            outcome=row["outcome"],
            env_state=json.loads(row["env_state"]),
            timestamp=row["timestamp"],
            task_id=row["task_id"],
            attempt=row["attempt"]
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
    
    def __enter__(self) -> EpisodeStore:
        return self
    
    def __exit__(self, *_):
        self.close()
