"""
test_tools.py
Isolated tests for SpotToolGroups — without ROS2, without hardware.

Runs with:
    cd RoboClaw && python roboclaw/embodied/spot/test_tools.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
import types

# Guarantee that the root directory of the RoboClaw project is in sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Isolate the heavy agent/__init__ before any import of roboclaw
for _name in ("roboclaw.agent", "roboclaw.agent.tools"):
    _stub = types.ModuleType(_name)
    _stub.__path__ = [_name.replace(".", "/")]
    _stub.__package__ = _name
    sys.modules[_name] = _stub


# ---------------------------------------------------------------------------
# Mock of SpotService (with ROS2)
# ---------------------------------------------------------------------------

from roboclaw.agent.tools.base import Tool  # noqa: E402
from roboclaw.embodied.spot.tools import (  # noqa: E402
    SpotBaseTool, SpotArmTool, SpotPerceptionTool, create_spot_tools
)


# ---------------------------------------------------------------------------
# Mock of SpotService (without ROS2)
# ---------------------------------------------------------------------------

class MockSpotService:
    """Simula o SpotService retornando JSON realista sem chamar ROS2."""

    async def move_forward(self, distance_m=1.0):
        return json.dumps({"status": "ok", "action": "move_forward",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def move_backward(self, distance_m=1.0):
        return json.dumps({"status": "ok", "action": "move_backward",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def move_left(self, distance_m=1.0):
        return json.dumps({"status": "ok", "action": "move_left",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def move_right(self, distance_m=1.0):
        return json.dumps({"status": "ok", "action": "move_right",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def rotate(self, angle_deg):
        direction = "left (CCW)" if angle_deg > 0 else "right (CW)"
        return json.dumps({"status": "ok", "action": "rotate",
                           "angle_deg": angle_deg, "direction": direction,
                           "duration_s": round(abs(angle_deg)/28.6, 2)})

    async def navigate_to_pose(self, x, y, yaw_deg=0.0, frame_id="map", timeout_s=60.0):
        return json.dumps({"status": "ok", "action": "navigate_to_pose",
                           "target": {"x": x, "y": y, "yaw_deg": yaw_deg, "frame": frame_id}})

    async def arm_move_cartesian(self, dx=0.0, dy=0.0, dz=0.0):
        return json.dumps({"status": "ok", "action": "arm_move_cartesian",
                           "delta": {"dx": dx, "dy": dy, "dz": dz},
                           "current_pose": {"x": 0.5+dx, "y": 0.0+dy, "z": 0.3+dz,
                                            "roll": 0, "pitch": 0, "yaw": 0}})

    async def arm_go_to_pose(self, x, y, z, roll_deg=0, pitch_deg=0, yaw_deg=0, frame_id="body"):
        return json.dumps({"status": "ok", "action": "arm_go_to_pose",
                           "pose": {"x": x, "y": y, "z": z,
                                    "roll": roll_deg, "pitch": pitch_deg, "yaw": yaw_deg},
                           "frame_id": frame_id})

    async def segment_pcl(self, max_objects=5, min_confidence=0.5):
        return json.dumps({"status": "ok", "action": "segment_pcl",
                           "objects_found": 2,
                           "objects": [
                               {"id": 0, "label": "cup", "confidence": 0.92,
                                "centroid": {"x": 0.6, "y": 0.1, "z": 0.2},
                                "bbox_size": {"x": 0.08, "y": 0.08, "z": 0.12}},
                               {"id": 1, "label": "bottle", "confidence": 0.78,
                                "centroid": {"x": 0.7, "y": -0.1, "z": 0.25},
                                "bbox_size": {"x": 0.06, "y": 0.06, "z": 0.22}},
                           ]})

    async def detect_objects(self, camera="hand"):
        return json.dumps({"status": "ok", "action": "detect_objects",
                           "camera": camera, "detections_found": 1,
                           "detections": [
                               {"label": "cup", "confidence": 0.88,
                                "bbox_px": {"x": 120, "y": 90, "w": 60, "h": 80}}
                           ]})

    async def get_gripper_state(self):
        return json.dumps({"status": "ok", "action": "get_gripper_state",
                           "gripper": {"position": 0.0, "force_n": 12.4,
                                       "is_holding": True, "object_detected": True}})


# ---------------------------------------------------------------------------
# Mock of the episodic memory
# ---------------------------------------------------------------------------

class MockMemory:
    def __init__(self):
        self.stored = []
        self.retrieved = []

    def store(self, subtask, outcome, env_state):
        self.stored.append({"subtask": subtask, "outcome": outcome, "env_state": env_state})

    def retrieve(self, query, top_k=3, as_context_string=False):
        self.retrieved.append(query)
        return "No relevant past robotic experience found." if as_context_string else []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def run_tests():
    svc = MockSpotService()
    mem = MockMemory()

    # ── spot_base ──────────────────────────────────────────────────────

    base = SpotBaseTool(spot_service=svc, episode_memory=mem)
    assert base.name == "spot_base"

    r = await base.execute(action="move_forward", distance_m=2.0)
    assert "FORWARD" in r.upper() or "forward" in r
    assert "2.00" in r
    print("✓ spot_base: move_forward")

    r = await base.execute(action="move_backward", distance_m=0.5)
    assert "backward" in r.lower() or "BACKWARD" in r.upper()
    print("✓ spot_base: move_backward")

    r = await base.execute(action="move_left")
    assert "left" in r.lower()
    print("✓ spot_base: move_left (default distance)")

    r = await base.execute(action="rotate_left", angle_deg=45.0)
    assert "45" in r or "left" in r.lower()
    print("✓ spot_base: rotate_left")

    r = await base.execute(action="rotate_right", angle_deg=90.0)
    assert "right" in r.lower() or "90" in r
    print("✓ spot_base: rotate_right")

    r = await base.execute(action="navigate_to_pose", x=3.0, y=1.5, yaw_deg=90.0)
    assert "3.00" in r and "1.50" in r
    print("✓ spot_base: navigate_to_pose")

    r = await base.execute(action="navigate_to_pose")
    assert "Error" in r
    print("✓ spot_base: navigate_to_pose sem x/y retorna erro")

    r = await base.execute(action="teleport")
    assert "Error" in r
    print("✓ spot_base: ação inválida retorna erro")

    # ── spot_arm ───────────────────────────────────────────────────────

    arm = SpotArmTool(spot_service=svc, episode_memory=mem)
    assert arm.name == "spot_arm"

    for direction in ("left", "right", "up", "down", "forward", "backward"):
        r = await arm.execute(action=f"move_arm_{direction}", step_m=0.2)
        assert direction in r.lower()
        print(f"✓ spot_arm: move_arm_{direction}")

    r = await arm.execute(action="arm_go_to_pose", x=0.8, y=0.1, z=0.5,
                           pitch_deg=30.0, frame_id="odom")
    assert "0.800" in r and "0.500" in r
    print("✓ spot_arm: arm_go_to_pose")

    r = await arm.execute(action="arm_go_to_pose")
    assert "Error" in r
    print("✓ spot_arm: arm_go_to_pose sem x/y/z retorna erro")

    # ── spot_perception ────────────────────────────────────────────────

    perc = SpotPerceptionTool(spot_service=svc, episode_memory=mem)
    assert perc.name == "spot_perception"

    r = await perc.execute(action="segment_pcl", max_objects=5)
    assert "cup" in r and "bottle" in r
    assert "2 object" in r
    print("✓ spot_perception: segment_pcl")

    r = await perc.execute(action="detect_objects", camera="hand")
    assert "cup" in r and "hand" in r
    print("✓ spot_perception: detect_objects")

    r = await perc.execute(action="get_gripper_state")
    assert "holding" in r.lower()
    print("✓ spot_perception: get_gripper_state")

    r = await perc.execute(action="unknown_action")
    assert "Error" in r
    print("✓ spot_perception: ação inválida retorna erro")

    # ── integration with the episodic memory ───────────────────────────────

    assert len(mem.stored) > 0, "Nenhum episódio foi gravado"
    outcomes = [e["outcome"] for e in mem.stored]
    assert "success" in outcomes
    assert len(mem.retrieved) > 0, "retrieve() nunca foi chamado"
    print("✓ memória: store() e retrieve() chamados durante execução")

    actions_stored = [e["subtask"] for e in mem.stored]
    assert "move_forward" in actions_stored
    assert "segment_pcl" in actions_stored
    assert "arm_go_to_pose" in actions_stored
    print("✓ memória: episódios de base, braço e percepção gravados")

    # ── factory ────────────────────────────────────────────────────────

    tools = create_spot_tools(svc, mem)
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"spot_base", "spot_arm", "spot_perception"}
    print("✓ create_spot_tools: 3 tools instanciados corretamente")

    # ── JSON Schema válido para o ToolRegistry ─────────────────────────

    for tool in tools:
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        params = schema["function"]["parameters"]
        assert "action" in params["properties"]
        assert params["required"] == ["action"]
    print("✓ to_schema(): JSON Schema válido para todos os tools")

    print()
    print("Todos os testes passaram.")


if __name__ == "__main__":
    asyncio.run(run_tests())