"""Tests for roboclaw_memory: EpisodeStore/RoboClawMemory kind-tagging, stats,
and LLM-based distillation. No coverage existed for this package before."""
from __future__ import annotations

import numpy as np
import pytest

from roboclaw.roboclaw_memory.roboclaw_memory import RoboClawMemory
from roboclaw.roboclaw_memory.roboclaw_memory.store import EpisodeStore


class _FakeEmbedder:
    """Deterministic fake embedding model — keeps tests hermetic and fast."""

    def encode(self, text, convert_to_numpy=True):
        vec = np.zeros(16, dtype=np.float32)
        for word in text.lower().split():
            vec[hash(word) % 16] += 1.0
        return vec


class _FakeProvider:
    def __init__(self, content: str | None = "Summary of past attempts.", raise_error: bool = False):
        self.content = content
        self.raise_error = raise_error
        self.calls: list[dict] = []

    async def chat_with_retry(self, messages, model=None, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        if self.raise_error:
            raise RuntimeError("provider unavailable")
        return _FakeResponse(self.content)


class _FakeResponse:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def store() -> EpisodeStore:
    return EpisodeStore(db_path=":memory:", embedding_model=_FakeEmbedder())


@pytest.fixture
def memory() -> RoboClawMemory:
    mem = RoboClawMemory(db_path=":memory:", embedding_model=_FakeEmbedder())
    return mem


# ---------------------------------------------------------------------------
# EpisodeStore: kind tagging
# ---------------------------------------------------------------------------

def test_insert_defaults_to_task_kind(store):
    eid = store.insert(subtask="move_forward", outcome="success", env_state={})
    ep = store.get(eid)
    assert ep.kind == "task"


def test_insert_with_explicit_kind(store):
    eid = store.insert(subtask="eap_reset", outcome="success", env_state={}, kind="eap_reset")
    ep = store.get(eid)
    assert ep.kind == "eap_reset"


def test_get_recent_filters_by_kind(store):
    store.insert(subtask="move_forward", outcome="success", env_state={}, kind="task")
    store.insert(subtask="eap_reset", outcome="success", env_state={}, kind="eap_reset")
    store.insert(subtask="rotate", outcome="failed", env_state={}, kind="task")

    task_only = store.get_recent(limit=10, kind="task")
    assert {e.subtask for e in task_only} == {"move_forward", "rotate"}

    everything = store.get_recent(limit=10)
    assert len(everything) == 3


def test_count_filters_by_kind(store):
    store.insert(subtask="a", outcome="success", env_state={}, kind="task")
    store.insert(subtask="b", outcome="success", env_state={}, kind="eap_reset")
    assert store.count(kind="task") == 1
    assert store.count(kind="eap_reset") == 1
    assert store.count() == 2


def test_search_excludes_other_kind(store):
    store.insert(subtask="move forward fast", outcome="success", env_state={}, kind="task")
    store.insert(subtask="move forward fast", outcome="success", env_state={}, kind="eap_reset")

    results = store.search("move forward fast", top_k=5, kind="task")
    assert len(results) == 1
    assert results[0][0].kind == "task"


# ---------------------------------------------------------------------------
# RoboClawMemory: store/retrieve pass-through, stats, distillation
# ---------------------------------------------------------------------------

def test_memory_store_and_retrieve_default_kind(memory):
    memory.store(subtask="move_forward", outcome="success", env_state={"distance_m": 1.0})
    ctx = memory.retrieve("move forward", as_context_string=True)
    assert "move_forward" in ctx


def test_memory_retrieve_empty_returns_empty_string(memory):
    assert memory.retrieve("anything", as_context_string=True) == ""


def test_memory_retrieve_excludes_eap_episodes_by_default(memory):
    memory.store(subtask="eap_reset", outcome="success", env_state={}, kind="eap_reset")
    assert memory.retrieve("eap_reset", as_context_string=True) == ""


def test_stats_is_read_only(memory):
    memory.store(subtask="move_forward", outcome="success", env_state={})
    memory.store(subtask="move_forward", outcome="failed", env_state={})
    task_id_before = memory.task_id

    stats = memory.stats()

    assert stats["total_episodes"] == 2
    assert stats["successes"] == 1
    assert stats["failures"] == 1
    assert "move_forward" in stats["problematic_subtasks"]
    assert memory.task_id == task_id_before  # no mutation, unlike the old consolidate()


@pytest.mark.asyncio
async def test_distill_calls_provider_and_caches_summary(memory):
    memory.store(subtask="move_forward", outcome="success", env_state={})
    provider = _FakeProvider(content="Robot moved forward successfully.")

    summary = await memory.distill(provider, model="test-model")

    assert summary == "Robot moved forward successfully."
    assert memory.get_distilled_summary() == summary
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == "test-model"


@pytest.mark.asyncio
async def test_distill_degrades_gracefully_on_provider_error(memory):
    memory.store(subtask="move_forward", outcome="success", env_state={})
    provider = _FakeProvider(raise_error=True)

    summary = await memory.distill(provider, model="test-model")

    assert summary == ""  # no crash, just no summary yet


@pytest.mark.asyncio
async def test_distill_noop_with_no_episodes(memory):
    provider = _FakeProvider()
    summary = await memory.distill(provider, model="test-model")
    assert summary == ""
    assert provider.calls == []


@pytest.mark.asyncio
async def test_maybe_distill_respects_every_threshold(memory):
    provider = _FakeProvider(content="first summary")
    memory.store(subtask="move_forward", outcome="success", env_state={})

    await memory.maybe_distill(provider, model="test-model", every=5)
    assert len(provider.calls) == 1  # first call always distills

    memory.store(subtask="move_backward", outcome="success", env_state={})
    await memory.maybe_distill(provider, model="test-model", every=5)
    assert len(provider.calls) == 1  # only 1 new episode, threshold not reached

    for _ in range(5):
        memory.store(subtask="rotate", outcome="success", env_state={})
    await memory.maybe_distill(provider, model="test-model", every=5)
    assert len(provider.calls) == 2  # threshold reached, re-distilled
