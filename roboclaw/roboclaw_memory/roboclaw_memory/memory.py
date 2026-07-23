"""API pública da memória episódica robótica do RoboClaw."""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from .store import Episode, EpisodeStore

@dataclass
class WorkingMemory:
    task_id: str
    active_skill: str = ""
    tool_call_history: list[dict] = field(default_factory=list)
    pending_episodes: list[Episode] = field(default_factory=list)

    def add_tool_call(self, tool, result):
        self.tool_call_history.append({"tool": tool, "result": result, "ts": time.time()})

    def clear(self):
        self.active_skill = ""
        self.tool_call_history.clear()
        self.pending_episodes.clear()

    def to_context_string(self) -> str:
        lines = [f"active_skill: {self.active_skill}"]
        if self.tool_call_history:
            lines.append(f"recent_tools: {' → '.join(c['tool'] for c in self.tool_call_history[-5:])}")
        if self.pending_episodes:
            lines.append("pending_episodes:")
            for ep in self.pending_episodes[-3:]:
                lines.append(f"  {ep.to_context_string()}")
        return "\n".join(lines)

class RoboClawMemory:
    """Memória episódica robótica — complementa o MemoryStore de conversação."""

    def __init__(self, db_path: str | Path = "~/.roboclaw/memory/episodes.db",
                 task_id: Optional[str] = None, embedding_model=None):
        self._store = EpisodeStore(db_path=str(db_path), embedding_model=embedding_model)
        self._task_id = task_id or str(uuid.uuid4())[:8]
        self._attempt_counter: dict[str, int] = {}
        self.working = WorkingMemory(task_id=self._task_id)

    def store(self, subtask: str, outcome: str, env_state: dict) -> Episode:
        attempt = self._attempt_counter.get(subtask, 0) + 1
        self._attempt_counter[subtask] = attempt
        eid = self._store.insert(subtask=subtask, outcome=outcome, env_state=env_state,
                                  task_id=self._task_id, attempt=attempt)
        ep = self._store.get(eid)
        self.working.pending_episodes.append(ep)
        return ep

    def retrieve(self, query: str, top_k: int = 3, as_context_string: bool = False):
        results = self._store.search(query, top_k=top_k)
        episodes = [ep for ep, _ in results]
        if not as_context_string: return episodes
        if not episodes: return "No relevant past robotic experience found."
        lines = ["## Robotic episode memory (past experience)"]
        for i, ep in enumerate(episodes, 1):
            lines.append(f"  {i}. {ep.to_context_string()}")
        return "\n".join(lines)

    def consolidate(self) -> dict:
        episodes = self._store.get_by_task(self._task_id)
        if not episodes:
            return {"task_id": self._task_id, "total": 0, "message": "nothing to consolidate"}
        total = len(episodes)
        succ  = sum(1 for e in episodes if e.outcome == "success")
        fail  = sum(1 for e in episodes if e.outcome == "failed")
        rec   = sum(1 for e in episodes if e.outcome == "recovered")
        prob  = [s for s,c in self._attempt_counter.items() if c>1]
        summary = {"task_id": self._task_id, "total_episodes": total,
                   "successes": succ, "failures": fail, "recoveries": rec,
                   "success_rate": round(succ/total,2) if total else 0.0,
                   "problematic_subtasks": prob, "consolidated_at": time.time()}
        self.working.clear(); self._attempt_counter.clear()
        self._task_id = str(uuid.uuid4())[:8]
        self.working.task_id = self._task_id
        return summary

    def set_active_skill(self, skill_name: str): self.working.active_skill = skill_name
    def log_tool_call(self, tool: str, result: str): self.working.add_tool_call(tool, result)
    def get_working_context(self) -> str: return self.working.to_context_string()
    def get_task_history(self) -> list[Episode]: return self._store.get_by_task(self._task_id)

    @property
    def task_id(self) -> str: return self._task_id

    def close(self): self._store.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()