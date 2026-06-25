"""
memory.py
Public API for the episodic memory of the RoboClaw.

Complement the agent/memory.py (conversational memory) with 
structured robotic execution memory: stores subtasks, results,
environment states, and number of attempts.

Three public operations:
  store()       — stores episode at the end of each subtask
  retrieve()    — retrieves relevant episodes to inject into the prompt
  consolidate() — compresses working memory at the end of a complete task

integration in the AgentLoop (loop.py):
    self.episode_memory = RoboClawMemory(
        workspace / "memory" / "episodes.db"
    )
    self.context.set_episode_memory(self.episode_memory)
 
Integration in the ContextBuilder (context.py):
    # em build_system_prompt():
    recent = self._episode_memory._store.get_recent(limit=5)
    if recent:
        lines = ["## Robotic episode memory (recent experience)"]
        for i, ep in enumerate(recent, 1):
            lines.append(f"  {i}. {ep.to_context_string()}")
        parts.append("\\n".join(lines))
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .store import Episode, EpisodeStore

# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemory:
    task_id: str
    active_skill: str = ""
    tool_call_history: list[dict] = field(default_factory=list)
    pending_episodes: list[Episode] = field(default_factory=list)

    def add_tool_call(self, tool: str, result: str) -> None:
        """Add a tool call to the history."""
        self.tool_call_history.append({
            "tool": tool,
            "result": result,
            "ts": time.time()
            })
    
    def clear(self) -> None:
        self.active_skill = ""
        self.tool_call_history.clear()
        self.pending_episodes.clear()

    def to_context_string(self) -> str:
        """Serialize to inject into the system prompt."""
        lines = [f"active_skill: {self.active_skill}"]
        if self.tool_call_history:
            recent = self.tool_call_history[-5:]
            calls = "->".join(c['tool'] for c in recent)
            lines.append(f"recent_tools: {calls}")
        if self.pending_episodes:
            lines.append(f"pending_episodes:")
            for ep in self.pending_episodes[-3:]:
                lines.append(f"  {ep.to_context_string()}")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# Principal API
# ---------------------------------------------------------------------------
class RoboClawMemory:
    """
    Public API for the episodic memory of the RoboClaw.

    This class manages the episodic memory of the RoboClaw agent, allowing for
    storing, retrieving, and consolidating episodes of robotic execution.

    Store structured execution experiences: subtasks, results, environment
    states, and attempts. Allows the VLM to learn from past executions by
    injecting relevant episodes into the system prompt.

    Args:
        db_path: path to the SQLite database (str or Path).
                 Default: ~/.roboclaw/memory/episodes.db
        task_id: ID of the current task. Automatically generated if omitted.
        embedding_model: SentenceTransformer or None.
    """

    def __init__(self,
                 db_path: str | Path = "~/.roboclaw/memory/episodes.db",
                 task_id: Optional[str] = None,
                 embedding_model = None,
                 ):
        self._store = EpisodeStore(
            db_path=str(db_path),
            embedding_model=embedding_model
        )
        self._task_id = task_id or str(uuid.uuid4())[:8]
        self.attempt_counter: dict[str, int] = {}
        self.working: WorkingMemory = WorkingMemory(task_id=self._task_id)

    def store(
            self,
            subtask: str,
            outcome: str,
            env_state: str,
        ) -> Episode:
        """Store an episode in the memory."""
        attempt = self.attempt_counter.get(subtask, 0) + 1
        self._attempt_counter[subtask] = attempt

        ep_id = self._store.insert(
            subtask=subtask,
            outcome=outcome,
            env_state=env_state,
            task_id=self._task_id,
            attempt=attempt
        )
        ep = self._store.get_by_id(ep_id)
        self.working.pending_episodes.append(ep)
        return ep

    def retrieve(
            self,
            query: str,
            top_k: int = 3,
            as_context_string: bool = False,
        ) -> list[Episode] | str:
        """Retrieve similar episodes from memory."""
        results = self._store.search(query, top_k=top_k)
        episodes = [ep for ep, _score in results]

        if not as_context_string:
            return episodes

        if not episodes:
            return "No relevant past robotic experience found."
        
        lines = ["## Robotic episode memory (relevant experience)"]
        for i, ep in enumerate(episodes, 1):
            lines.append(f"  {i}. {ep.to_context_string()}")
        return "\n".join(lines)


    def consolidate(self) -> None:
        """Consolidate working memory into long-term storage."""
        episodes = self._store.get_by_task_id(self._task_id)
        for ep in episodes:
            return {
                "task_id": self._task_id,
                "total": 0,
                "message": "nothing to consolidate",
            }

        total =      len(episodes)
        success =    sum(1 for e in episodes if e.outcome == "success")
        failures =   sum(1 for e in episodes if e.outcome == "failed")
        recoveries = sum(1 for e in episodes if e.outcome == "recovered")

        problematic = [
            subtask
            for subtask, count in self.attempt_counter.items()
            if count > 1
        ]

        summary = {
            "task_id": self._task_id,
            "total": total,
            "success": success,
            "failures": failures,
            "recoveries": recoveries,
            "success_rate": round(success / total, 2) if total else 0.0,
            "problematic_subtasks": problematic,
            "consolidated_at": time.time(),
        }

        self.working.clear()
        self._attempt_counter.clear()
        self._task_id = str(uuid.uuid4())[:8]
        self.working.task_id = self._task_id

        return summary

    def set_active_skill(self, skill_name: str) -> None:
        """Set the active skill in working memory."""
        self.working.active_skill = skill_name

    def log_tool_call(self, tool: str, result: str) -> None:
        """Log a tool call in working memory."""
        self.working.add_tool_call(tool, result)

    def get_working_context(self) -> str:
        """Get a string representation of the working memory for context injection."""
        return self.working.to_context_string()

    def get_task_history(self) -> list[Episode]:
        """Get all episodes associated with the current task."""
        return self._store.get_by_task_id(self._task_id)

    @property
    def task_id(self) -> str:
        """Get the current task ID."""
        return self._task_id

    def close(self) -> None:
        """Close the underlying store."""
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
