"""
eap.py
======
Entangled Action Pairs (EAP) — autonomous collection of data without manual reset.

The EAP intercepts the RESETTING phase of the RecordSession and executes
the inverse policy (π←) automatically, returning the environment to its
initial state without human intervention.

Flow with EAP:
    LeRobot emits "Reset the environment"
        → RecordPhaseController → RESETTING
        → EAPController.on_reset_detected()
        → executa π← via SpotService (or qualquer serviço de inferência)
        → when π← ends → sends SKIP_RESET to continue recording
 
Without EAP (original behavior):
    RESETTING → waits for human to press key or manually reset
 
Usage:
    eap = EAPController(
        phase_controller=record_session._phase,
        reset_executor=SpotResetExecutor(spot_svc, checkpoint_path),
    )
    # Pass eap to RecordOutputConsumer via RecordSession.__init__
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class ResetExecutor(Protocol):
    """
    Interface for executors of the reset policy.
 
    Any object that implements execute_reset() can be used as
    the inverse policy in EAP. Allows swapping executors without modifying
    the EAPController — e.g., SpotResetExecutor, LeRobotResetExecutor, MockResetExecutor.
    """

    async def execute_reset(self, env_state: dict | None = None) -> "ResetResult":
        """
        Execute the reset policy (π←).
 
        Args:
            env_state: current state of the environment (optional, for logging and memory)
 
        Returns:
            ResetResult with success, message and env_state_after
        """
        ...

# ---------------------------------------------------------------------------
# Reset Result
# ---------------------------------------------------------------------------

class ResetResult:
    """Result of a reset policy execution."""

    def __init__(
        self,
        success: bool,
        message: str = "",
        env_state_after: dict | None = None,
        attempts: int = 1,
    ) -> None:
        self.success = success
        self.message = message
        self.env_state_after = env_state_after or {}
        self.attempts = attempts

    def __repr__(self) -> str:
        return (
            f"ResetResult(success={self.success}, "
            f"attempts={self.attempts}, msg={self.message!r})"
        )

# ---------------------------------------------------------------------------
# EAP Controller
# ---------------------------------------------------------------------------

class EAPController:
    """
    Controls the EAP loop within a recording session.
 
    Monitors a phase of the RecordPhaseController. When detects RESETTING,
    executes the ResetExecutor and, in case of success, sends SKIP_RESET
    to continue recording automatically.
 
    Args:
        phase_controller: RecordPhaseController da s in the recording session
        reset_executor:   implements the ResetExecutor (política π←)
        max_reset_retries: tentatives before escalating to human
        episode_memory:   RoboClawMemory optional for recording episodes
        poll_interval_s:  polling interval of the phase (seconds)
    """


    def __init__(
        self,
        phase_controller: Any,
        reset_executor: ResetExecutor,
        max_reset_retries: int = 2,
        episode_memory: Any = None,
        poll_interval_s: float = 0.5,
    ) -> None:
        self._phase = phase_controller
        self._executor = reset_executor
        self._max_retries = max_reset_retries
        self._mem = episode_memory
        self._poll = poll_interval_s
        self._running = False
        self._task: asyncio.Task | None = None

        # Session statistics
        self.total_resets_attempted = 0
        self.total_resets_succeeded = 0
        self.total_resets_escalated = 0  # Needs a human

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

    def start(self) -> None:
        """ Start the monitoring loop in the background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("EAPController started")

    async def stop(self) -> None:
        """For the monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("EAPController stopped - stats: %s", self.stats())

    # ---------------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """
        Monitors the phase of the session and intercepts the RESETTING.
        Runs as a task in the background while the recording is active.
        """
        from roboclaw.embodied.service.session.record import RecordPhase

        while self._running:
            try:
                phase = self._phase.phase
                if phase == RecordPhase.RESETTING:
                    await self._handle_reset()
                await asyncio.sleep(self._poll)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("EAPController monitor loop error")
                await asyncio.sleep(self._poll * 2)

    # ---------------------------------------------------------------------------
    # Reset handling
    # ---------------------------------------------------------------------------

    async def _handle_reset(self) -> None:
        """
        Executes the automatic reset cycle.

        Try until max_reset_retries. If fails, escalate to human intervention.
        """

        from roboclaw.embodied.service.session.record import RecordPhase

        self.total_resets_attempted += 1
        logger.info(
            "EAP: reset detected (attempt #%d total)", self.total_resets_attempted
        )

        last_result: ResetResult | None = None

        for attempt in range(1, self._max_retries + 1):

            if self._phase.phase != RecordPhase.RESETTING:
                logger.info("EAP: phase changed before reset attempt %d, aborting", attempt)
                return

            logger.info("EAP: executing reset policy π← (attempt %d/%d)", attempt, self._max_retries)
            try:
                result = await self._executor.execute_reset()
            except Exception as exc:
                logger.warning("EAP: reset executor raised: %s", exc)
                result = ResetResult(success=False, message=str(exc), attempts=attempt)

            last_result = result

            if result.success:
                self.total_resets_succeeded += 1
                logger.info("EAP: reset succeeded — sending skip_reset")
                self._store_episode("eap_reset", "success", result)

                if self._phase.phase == RecordPhase.RESETTING:
                    try:
                        await self._phase.request_skip_reset()
                    except RuntimeError as e:
                        logger.warning("EAP: failed to send skip_reset: %s", e)
                return

            logger.warning(
                "EAP: reset attempt %d/%d failed: %s",
                attempt, self._max_retries, result.message,
            )
            self._store_episode("eap_reset", "failed", result)

            if attempt < self._max_retries:
                await asyncio.sleep(self._poll * 2)

        self.total_resets_escalated += 1
        logger.warning(
            "EAP: all %d reset attempts failed — waiting for human intervention. "
            "Last: %s",
            self._max_retries,
            last_result,
        )
        await asyncio.sleep(self._poll * 20)

    def _store_episode(
        self, subtask: str, outcome: str, result: ResetResult
    ) -> None:
        """
        Records the reset attempt in the episodic memory, if available.
        """
        if self._mem is None:
            return

        env_state = {
            "eap_attempts": result.attempts,
            "reset_message": result.message,
            **result.env_state_after,
        }

        try:
            self._mem.store(subtask=subtask, outcome=outcome, env_state=env_state, kind="eap_reset")
        except Exception as exc:
            logger.warning("EAP: failed to store episode in memory: %s", exc)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "total_resets_attempted": self.total_resets_attempted,
            "total_resets_succeeded": self.total_resets_succeeded,
            "total_resets_escalated": self.total_resets_escalated,
            "success_rate": (
                round(self.total_resets_succeeded / self.total_resets_attempted, 2)
                if self.total_resets_attempted else 0.0
            ),
        }

# ---------------------------------------------------------------------------
# SpotResetExecutor
# ---------------------------------------------------------------------------


class SpotResetExecutor:
    """
    ResetExecutor implementation that calls a reset policy via SpotService.

    The inverse policy is executed as a sequence of commands 
    to the Spot robot arm/base that revert the effect of the direct policy.
 
    Args:
        spot_service:     instance of SpotService
        reset_sequence:   list of dicts with {action, kwargs} to execute in sequence
                          e.g., [{"action": "arm_go_to_pose", "kwargs": {"x":0.5,"y":0,"z":0.3}},
                                  {"action": "move_backward",  "kwargs": {"distance_m": 0.5}}]
        timeout_s:        total timeout for the reset (seconds)
    """

    def __init__(
            self,
            spot_service: Any,
            reset_sequence: list[dict[str, Any]],
            timeout_s: float = 30.0
    ) -> None:
        self._svc = spot_service
        self._sequence = reset_sequence
        self._timeout = timeout_s

    async def execute_reset(self, env_state: dict | None = None) -> ResetResult:
        """Executes the reset sequence with timeout"""
        try:
            result = await asyncio.wait_for(
                self._run_sequence(), timeout=self._timeout
            )
            return result
        except asyncio.TimeoutError:
            return ResetResult(
                success=False,
                message=f"Reset timed out after {self._timeout}s",
                )

    async def _run_sequence(self) -> ResetResult:
        """Executes each step of the reset sequence"""
        import json

        for i, step in enumerate(self._sequence):
            action = step.get("action")
            kwargs = step.get("kwargs", {})

            method = getattr(self._svc, action, None)
            if method is None:
                return ResetResult(
                    success=False,
                    message=f"SpotService has no method '{action}' (step {i+1})",
                )

            try:
                result_json = await method(**kwargs)
                data = json.loads(result_json) if isinstance(result_json, str) else {}

                if data.get("status") != "ok":
                    return ResetResult(
                        success=False,
                        message=f"Step {i+1} ({action}) failed: {data.get('error', 'unknown')}",
                    )
            except Exception as exc:
                return ResetResult(
                    success=False,
                    message=f"Step {i+1} ({action}) raised: {exc}",
                )

        return ResetResult(success=True, message="Reset sequence completed")

# ---------------------------------------------------------------------------
# LeRobotResetExecutor
# ---------------------------------------------------------------------------

class LeRobotResetExecutor:
    """
    ResetExecutor that executes a LeRobot-trained policy as π←.

    Uses the EmbodiedService.run_inference() with the inverse policy checkpoint, 
    executing the reset in the same way as the direct policy — without new
    infrastructure.
 
    Args:
        embodied_service:       EmbodiedService of the parent session
        reset_checkpoint_path:  path to the inverse policy checkpoint
        reset_task:             description of the reset task (for the VLA)
        num_episodes:           always 1 for reset
        episode_time_s:         maximum time for the reset
    """

    def __init__(
            self,
            embodied_service: Any,
            reset_checkpoint_path: str,
            reset_task: str = "reset_environment",
            episode_time_s: int = 30,
    ) -> None:
        self._svc = embodied_service
        self._checkpoint = reset_checkpoint_path
        self._task = reset_task
        self._episode_time_s = episode_time_s

    async def execute_reset(self, env_state: dict | None = None) -> ResetResult:
        """Executes the reset policy via run_inference() and EmbodiedService"""
        try:
            result = await self._svc.run_inference(
                checkpoint_path=self._checkpoint,
                task=self._task,
                num_episodes=1,
                episode_time_s=self._episode_time_s,
                use_cameras=True,
            )
            success = "error" not in result.lower() and "failed" not in result.lower()
            return ResetResult(
                success=success,
                message=result,
                env_state_after={"reset_task": self._task},  # Optionally update with new state if available
            )
        except Exception as exc:
            return ResetResult(
                success=False,
                message=str(exc),
            )

