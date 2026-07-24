"""API pública da memória episódica robótica do RoboClaw."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .store import Episode, EpisodeStore

_DISTILL_PROMPT = """Summarize the following robotic task episodes into a concise \
natural-language briefing (under 300 words) that an assistant can use as background \
context for future tasks. Focus on what worked, what failed and why, and any \
recurring problems. Do not invent details that are not present below.

Stats: {stats}

Episodes (oldest to newest):
{episodes}"""


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
        self._distilled_summary = ""
        self._distilled_at_count = 0

    def store(self, subtask: str, outcome: str, env_state: dict, kind: str = "task") -> Episode:
        attempt = self._attempt_counter.get(subtask, 0) + 1
        self._attempt_counter[subtask] = attempt
        eid = self._store.insert(subtask=subtask, outcome=outcome, env_state=env_state,
                                  task_id=self._task_id, attempt=attempt, kind=kind)
        ep = self._store.get(eid)
        self.working.pending_episodes.append(ep)
        return ep

    def retrieve(self, query: str, top_k: int = 3, as_context_string: bool = False,
                 kind: Optional[str] = "task"):
        results = self._store.search(query, top_k=top_k, kind=kind)
        episodes = [ep for ep, _ in results]
        if not as_context_string: return episodes
        if not episodes: return ""
        lines = ["## Robotic episode memory (past experience)"]
        for i, ep in enumerate(episodes, 1):
            lines.append(f"  {i}. {ep.to_context_string()}")
        return "\n".join(lines)

    def stats(self, limit: Optional[int] = None, kind: str = "task") -> dict:
        """Read-only success/failure/recovery breakdown over recent episodes."""
        episodes = self._store.get_recent(limit=limit or 10_000, kind=kind)
        if not episodes:
            return {"total_episodes": 0, "message": "nothing recorded yet"}
        total = len(episodes)
        succ = sum(1 for e in episodes if e.outcome == "success")
        fail = sum(1 for e in episodes if e.outcome == "failed")
        rec = sum(1 for e in episodes if e.outcome == "recovered")
        counts: dict[str, int] = {}
        for e in episodes:
            counts[e.subtask] = counts.get(e.subtask, 0) + 1
        return {
            "total_episodes": total, "successes": succ, "failures": fail, "recoveries": rec,
            "success_rate": round(succ / total, 2) if total else 0.0,
            "problematic_subtasks": [s for s, c in counts.items() if c > 1],
        }

    async def distill(self, provider: Any, model: str, max_episodes: int = 15) -> str:
        """LLM-based summary of recent task episodes, replacing raw-episode injection."""
        episodes = self._store.get_recent(limit=max_episodes, kind="task")
        if not episodes:
            return self._distilled_summary
        episodes_text = "\n".join(ep.to_context_string() for ep in reversed(episodes))
        prompt = _DISTILL_PROMPT.format(
            stats=json.dumps(self.stats(limit=max_episodes)), episodes=episodes_text,
        )
        try:
            response = await provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}], model=model,
            )
            summary = (response.content or "").strip()
        except Exception:
            logger.warning("Episode distillation failed, keeping previous summary")
            return self._distilled_summary
        if summary:
            self._distilled_summary = summary
            self._distilled_at_count = self._store.count(kind="task")
        return self._distilled_summary

    async def maybe_distill(self, provider: Any, model: str, every: int = 5,
                             max_episodes: int = 15) -> None:
        """Re-distill only after `every` new task episodes since the last pass."""
        total = self._store.count(kind="task")
        if total == 0 or (self._distilled_summary and total - self._distilled_at_count < every):
            return
        await self.distill(provider, model, max_episodes=max_episodes)

    def get_distilled_summary(self) -> str:
        return self._distilled_summary

    def set_active_skill(self, skill_name: str): self.working.active_skill = skill_name
    def log_tool_call(self, tool: str, result: str): self.working.add_tool_call(tool, result)
    def get_working_context(self) -> str: return self.working.to_context_string()
    def get_task_history(self) -> list[Episode]: return self._store.get_by_task(self._task_id)

    @property
    def task_id(self) -> str: return self._task_id

    def close(self): self._store.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
