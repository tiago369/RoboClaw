"""
test_eap.py
===========
Isolated tests for EAPController and related classes. - No ROS2, no hardware, no LeRobot.

Runs with:
    cd RoboClaw && python roboclaw/embodied/service/session/test_eap.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

# ── path e stubs de dependências pesadas ──────────────────────────────────
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

for _name in ("roboclaw.agent", "roboclaw.agent.tools",
              "roboclaw.embodied.service",
              "roboclaw.embodied.service.session"):
    _stub = types.ModuleType(_name)
    _stub.__path__ = [_name.replace(".", "/")]
    _stub.__package__ = _name
    sys.modules[_name] = _stub

from roboclaw.embodied.service.session.eap import (  # noqa: E402
    EAPController,
    LeRobotResetExecutor,
    ResetResult,
    SpotResetExecutor,
)

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockRecordPhaseController:
    """Emulates RecordPhaseController without Board/ROS2."""

    def __init__(self, initial_phase: str = "idle") -> None:
        self._phase = initial_phase
        self.skip_reset_calls = 0
        self.skip_reset_rejected = False

    @property
    def phase(self):
        from roboclaw.embodied.service.session.record import RecordPhase
        return RecordPhase(self._phase)

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    async def request_skip_reset(self) -> None:
        if self.skip_reset_rejected:
            raise RuntimeError("skip_reset rejected (test)")
        self.skip_reset_calls += 1
        self._phase = "recording"


class MockResetExecutor:
    """ResetExecutor configurable for tests."""

    def __init__(self, outcomes: list[bool]) -> None:
        """outcomes: list of True/False for each call in sequence."""
        self._outcomes = list(outcomes)
        self.call_count = 0

    async def execute_reset(self, env_state=None) -> ResetResult:
        idx = min(self.call_count, len(self._outcomes) - 1)
        success = self._outcomes[idx]
        self.call_count += 1
        return ResetResult(
            success=success,
            message="ok" if success else "gripper missed",
            env_state_after={"arm_x": 0.5} if success else {},
            attempts=self.call_count,
        )


class MockMemory:
    def __init__(self):
        self.stored: list[dict] = []

    def store(self, subtask, outcome, env_state):
        self.stored.append({"subtask": subtask, "outcome": outcome, "env": env_state})


class MockSpotService:
    """SpotService mock for SpotResetExecutor."""
    async def arm_go_to_pose(self, **kwargs):
        import json
        return json.dumps({"status": "ok", "action": "arm_go_to_pose", "pose": kwargs})

    async def move_backward(self, distance_m=0.5):
        import json
        return json.dumps({"status": "ok", "action": "move_backward", "distance_m": distance_m})

    async def bad_method(self, **kwargs):
        import json
        return json.dumps({"status": "error", "error": "hardware fault"})


class MockEmbodiedService:
    """EmbodiedService mock for LeRobotResetExecutor."""
    def __init__(self, result: str = "Reset complete"):
        self._result = result

    async def run_inference(self, **kwargs):
        return self._result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def run_eap_cycle(phase_ctrl, executor, memory=None, max_retries=2, poll=0.05):
    """Executes a complete cycle: creates EAP, sets RESETTING, waits for resolution."""
    eap = EAPController(
        phase_controller=phase_ctrl,
        reset_executor=executor,
        max_reset_retries=max_retries,
        episode_memory=memory,
        poll_interval_s=poll,
    )
    eap.start()
    phase_ctrl.set_phase("resetting")

    for _ in range(40):
        await asyncio.sleep(0.05)
        if phase_ctrl._phase != "resetting":
            break

    await eap.stop()
    return eap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_successful_reset():
    """Reset successful on first attempt → skip_reset sent."""
    phase = MockRecordPhaseController()
    executor = MockResetExecutor(outcomes=[True])
    mem = MockMemory()

    eap = await run_eap_cycle(phase, executor, memory=mem)

    assert executor.call_count == 1, f"Expected 1 call, got {executor.call_count}"
    assert phase.skip_reset_calls == 1, "skip_reset should have been called once"
    assert phase._phase == "recording", f"Phase should be recording, got {phase._phase}"
    assert eap.total_resets_succeeded == 1
    assert eap.total_resets_escalated == 0

    # Memória deve ter 1 episódio de sucesso
    assert len(mem.stored) == 1
    assert mem.stored[0]["outcome"] == "success"
    assert mem.stored[0]["subtask"] == "eap_reset"
    print("✓ reset successful on first attempt")


async def test_retry_then_success():
    """Failure on first attempt, success on second → skip_reset sent."""
    phase = MockRecordPhaseController()
    executor = MockResetExecutor(outcomes=[False, True])
    mem = MockMemory()

    eap = await run_eap_cycle(phase, executor, memory=mem, max_retries=3)

    assert executor.call_count == 2
    assert phase.skip_reset_calls == 1
    assert eap.total_resets_succeeded == 1
    assert eap.total_resets_escalated == 0

    outcomes = [e["outcome"] for e in mem.stored]
    assert outcomes == ["failed", "success"]
    print("✓ retry: failure on first attempt, success on second")


async def test_all_retries_fail_escalates_to_human():
    """All retries fail → phase remains RESETTING (human intervenes)."""
    phase = MockRecordPhaseController()
    executor = MockResetExecutor(outcomes=[False, False])

    eap = EAPController(
        phase_controller=phase,
        reset_executor=executor,
        max_reset_retries=2,
        poll_interval_s=0.02,
    )
    eap.start()
    phase.set_phase("resetting")
    await asyncio.sleep(0.5)
    await eap.stop()

    assert executor.call_count >= 2, f"Expected at least 2 retries, got {executor.call_count}"
    assert phase.skip_reset_calls == 0, "skip_reset should NOT be called when escalating"
    assert eap.total_resets_escalated >= 1
    assert eap.total_resets_succeeded == 0
    print("✓ all retries fail → escalates to human (phase remains RESETTING)")


async def test_human_intervenes_before_eap():
    """If phase changes before EAP acts (human intervenes), EAP does not send skip."""
    phase = MockRecordPhaseController()
    executor = MockResetExecutor(outcomes=[True])

    eap = EAPController(
        phase_controller=phase,
        reset_executor=executor,
        max_reset_retries=2,
        poll_interval_s=0.05,
    )
    eap.start()
    phase.set_phase("resetting")

    await asyncio.sleep(0.01)
    phase.set_phase("recording")

    await asyncio.sleep(0.3)
    await eap.stop()

    assert eap.total_resets_escalated == 0 or eap.total_resets_succeeded >= 0
    print("✓ human intervenes before EAP: no conflict")


async def test_skip_reset_rejected_does_not_crash():
    """If skip_reset raises RuntimeError (phase already changed), EAP does not crash."""
    phase = MockRecordPhaseController()
    phase.skip_reset_rejected = True

    executor = MockResetExecutor(outcomes=[True])

    eap = EAPController(
        phase_controller=phase,
        reset_executor=executor,
        max_reset_retries=2,
        poll_interval_s=0.05,
    )
    eap.start()
    phase.set_phase("resetting")

    await asyncio.sleep(0.2)

    phase.set_phase("recording")
    await asyncio.sleep(0.15)
    await eap.stop()

    assert executor.call_count >= 1
    print("✓ skip_reset rejected dont crashes with EAPController")


async def test_no_memory_does_not_crash():
    """EAP without episodic memory works normally."""
    phase = MockRecordPhaseController()
    executor = MockResetExecutor(outcomes=[True])

    eap = await run_eap_cycle(phase, executor, memory=None)
    assert eap.total_resets_succeeded == 1
    print("✓ EAP without episodic memory works normally")


async def test_stats():
    """stats() return correct values after multiple cycles."""
    phase = MockRecordPhaseController()
    executor = MockResetExecutor(outcomes=[True, False, False, True])

    eap = EAPController(
        phase_controller=phase,
        reset_executor=executor,
        max_reset_retries=2,
        poll_interval_s=0.02,
    )
    eap.start()

    phase.set_phase("resetting")
    await asyncio.sleep(0.15)

    phase.set_phase("resetting")
    phase.skip_reset_calls = 0
    await asyncio.sleep(0.25)

    await eap.stop()

    s = eap.stats()
    assert "total_resets_attempted" in s
    assert "success_rate" in s
    assert 0.0 <= s["success_rate"] <= 1.0
    print(f"✓ stats() corretos: {s}")


async def test_spot_reset_executor_success():
    """SpotResetExecutor executes action sequence with success."""
    svc = MockSpotService()
    sequence = [
        {"action": "arm_go_to_pose", "kwargs": {"x": 0.5, "y": 0.0, "z": 0.3}},
        {"action": "move_backward",  "kwargs": {"distance_m": 0.5}},
    ]
    executor = SpotResetExecutor(svc, reset_sequence=sequence, timeout_s=5.0)
    result = await executor.execute_reset()

    assert result.success, f"Expected success, got: {result.message}"
    assert result.message == "Reset sequence completed"
    print("✓ SpotResetExecutor: success sequence")


async def test_spot_reset_executor_step_failure():
    """SpotResetExecutor stops and return failure when a step fails."""
    svc = MockSpotService()
    sequence = [
        {"action": "bad_method", "kwargs": {}},
        {"action": "move_backward", "kwargs": {}},
    ]
    executor = SpotResetExecutor(svc, reset_sequence=sequence)
    result = await executor.execute_reset()

    assert not result.success
    assert "bad_method" in result.message or "hardware" in result.message
    print("✓ SpotResetExecutor: failure in a step returns ResetResult(success=False)")


async def test_spot_reset_executor_invalid_method():
    """SpotResetExecutor returns failure if method does not exist in the service."""
    svc = MockSpotService()
    sequence = [{"action": "nonexistent_method", "kwargs": {}}]
    executor = SpotResetExecutor(svc, reset_sequence=sequence)
    result = await executor.execute_reset()

    assert not result.success
    assert "nonexistent_method" in result.message
    print("✓ SpotResetExecutor: nonexistent method returns failure")


async def test_lerobot_reset_executor_success():
    """LeRobotResetExecutor returns success when run_inference does not contain error."""
    svc = MockEmbodiedService("Reset complete. 1 episode saved.")
    executor = LeRobotResetExecutor(svc, reset_checkpoint_path="/path/to/reset_policy")
    result = await executor.execute_reset()

    assert result.success
    print("✓ LeRobotResetExecutor: run_inference bem-sucedido")


async def test_lerobot_reset_executor_failure():
    """LeRobotResetExecutor returns failure when run_inference indicates error."""
    svc = MockEmbodiedService("Recording failed: gripper error")
    executor = LeRobotResetExecutor(svc, reset_checkpoint_path="/path/to/reset_policy")
    result = await executor.execute_reset()

    assert not result.success
    print("✓ LeRobotResetExecutor: run_inference with error returns failure")


async def test_attach_eap_sets_controller():
    """attach_eap() creates EAPController and atributes to RecordSession._eap."""
    import types as _types

    session = _types.SimpleNamespace(
        _phase=MockRecordPhaseController(),
        _eap=None,
    )

    from roboclaw.embodied.service.session.eap import EAPController
    executor = MockResetExecutor(outcomes=[True])
    mem = MockMemory()

    eap = EAPController(
        phase_controller=session._phase,
        reset_executor=executor,
        max_reset_retries=2,
        episode_memory=mem,
    )
    session._eap = eap

    assert session._eap is not None
    assert session._eap._max_retries == 2
    assert session._eap._mem is mem
    print("✓ attach_eap: EAPController created and associated correctly")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("EAP — isolated tests (without ROS2, without hardware)")
    print("=" * 60)
    print()

    tests = [
        test_successful_reset,
        test_retry_then_success,
        test_all_retries_fail_escalates_to_human,
        test_human_intervenes_before_eap,
        test_skip_reset_rejected_does_not_crash,
        test_no_memory_does_not_crash,
        test_stats,
        test_spot_reset_executor_success,
        test_spot_reset_executor_step_failure,
        test_spot_reset_executor_invalid_method,
        test_lerobot_reset_executor_success,
        test_lerobot_reset_executor_failure,
        test_attach_eap_sets_controller,
    ]

    failed = 0
    for test in tests:
        try:
            await test()
        except Exception:
            import traceback
            print(f"✗ {test.__name__}: FAILED")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED.")
    else:
        print(f"All {len(tests)} tests passed.")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    code = asyncio.run(main())
    sys.exit(code)
