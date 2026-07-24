"""
test_tools.py — testes isolados sem ROS2, sem hardware.
cd RoboClaw && python roboclaw/embodied/spot/test_tools.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

for _n in ("roboclaw.agent", "roboclaw.agent.tools", "roboclaw.agent.tools.base"):
    _s = types.ModuleType(_n); _s.__path__ = []; sys.modules[_n] = _s

# Stub da classe Tool
class _Tool:
    @property
    def name(self): return ""
    @property
    def description(self): return ""
    @property
    def parameters(self): return {}
    async def execute(self, **kwargs): return ""
    def to_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}

sys.modules["roboclaw.agent.tools.base"].Tool = _Tool


# ---------------------------------------------------------------------------
# Mock do SpotService — simula grasp_pipeline sem ROS2
# ---------------------------------------------------------------------------

class MockSpotService:
    def __init__(self, fail_at: str | None = None):
        """fail_at: nome da fase que deve falhar (para testes de erro)."""
        self._fail_at = fail_at
        self._last_segmented_points = object()  # truthy
        self._last_grasp_poses = object()
        self._last_target_pose = object()
        self._cam_lock = __import__("threading").Lock()
        self._latest_rgb   = object()  # câmeras disponíveis por padrão
        self._latest_depth = object()
        self._latest_k_matrix = object()
        self.calls: list[str] = []

    def _ensure_ready(self): pass

    async def segment_object(self, object_name: str, timeout_s: float = 10.0) -> str:
        self.calls.append("segment_object")
        if self._fail_at == "segment":
            return json.dumps({"status": "error", "error": "objeto não encontrado"})
        return json.dumps({"status": "ok", "action": "segment_object",
                           "object_name": object_name, "points_width": 1024,
                           "points_height": 1, "frame_id": "body"})

    async def generate_grasps(self, timeout_s: float = 15.0) -> str:
        self.calls.append("generate_grasps")
        if self._fail_at == "grasps":
            return json.dumps({"status": "error", "error": "nenhuma pose gerada"})
        return json.dumps({"status": "ok", "action": "generate_grasps",
                           "n_poses": 5, "top_poses": [
                               {"index": 0, "position": {"x": 0.6, "y": 0.0, "z": 0.2},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
                           ]})

    async def plan_trajectory(self, timeout_s: float = 20.0) -> str:
        self.calls.append("plan_trajectory")
        if self._fail_at == "plan":
            return json.dumps({"status": "error", "error": "cuRobo falhou"})
        return json.dumps({"status": "ok", "action": "plan_trajectory",
                           "target_pose": {"frame_id": "body",
                                           "position": {"x": 0.6, "y": 0.0, "z": 0.2},
                                           "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}})

    async def execute_grasp(self, close_gripper: bool = True, timeout_s: float = 30.0) -> str:
        self.calls.append("execute_grasp")
        if self._fail_at == "execute":
            return json.dumps({"status": "error", "error": "arm motion failed"})
        return json.dumps({"status": "ok", "action": "execute_grasp",
                           "gripper": "closed" if close_gripper else "open"})

    async def move_to_home(self, timeout_s: float = 20.0) -> str:
        self.calls.append("move_to_home")
        return json.dumps({"status": "ok", "action": "move_to_home"})

    async def open_gripper(self, timeout_s: float = 5.0) -> str:
        self.calls.append("open_gripper")
        return json.dumps({"status": "ok", "action": "open_gripper", "message": "gripper opened"})

    async def close_gripper(self, timeout_s: float = 5.0) -> str:
        self.calls.append("close_gripper")
        return json.dumps({"status": "ok", "action": "close_gripper", "message": "gripper closed"})

    async def move_forward(self, distance_m: float = 1.0) -> str:
        self.calls.append("move_forward")
        return json.dumps({"status": "ok", "action": "move_forward",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def move_backward(self, distance_m: float = 1.0) -> str:
        self.calls.append("move_backward")
        return json.dumps({"status": "ok", "action": "move_backward",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def move_left(self, distance_m: float = 1.0) -> str:
        self.calls.append("move_left")
        return json.dumps({"status": "ok", "action": "move_left",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def move_right(self, distance_m: float = 1.0) -> str:
        self.calls.append("move_right")
        return json.dumps({"status": "ok", "action": "move_right",
                           "distance_m": distance_m, "duration_s": round(distance_m/0.5, 2)})

    async def rotate(self, angle_deg: float) -> str:
        self.calls.append("rotate")
        direction = "left (CCW)" if angle_deg > 0 else "right (CW)"
        return json.dumps({"status": "ok", "action": "rotate",
                           "angle_deg": angle_deg, "direction": direction,
                           "duration_s": round(abs(angle_deg)/28.6, 2)})

    async def navigate_to_pose(self, x, y, yaw_deg=0.0, frame_id="map", timeout_s=60.0) -> str:
        self.calls.append("navigate_to_pose")
        return json.dumps({"status": "ok", "action": "navigate_to_pose",
                           "target": {"x": x, "y": y, "yaw_deg": yaw_deg}})


class MockMemory:
    def __init__(self):
        self.stored: list[dict] = []
        self.retrieved: list[str] = []
    def store(self, subtask, outcome, env_state):
        self.stored.append({"subtask": subtask, "outcome": outcome})
    def retrieve(self, query, top_k=3, as_context_string=False):
        self.retrieved.append(query)
        return [] if not as_context_string else ""


# ---------------------------------------------------------------------------
# Imports do módulo a testar
# ---------------------------------------------------------------------------

from roboclaw.embodied.spot.tools import (
    SpotArmTool,
    SpotBaseTool,
    SpotPerceptionTool,
    create_spot_tools,
)

# ---------------------------------------------------------------------------
# Testes: SpotBaseTool
# ---------------------------------------------------------------------------

async def test_base_move_actions():
    svc = MockSpotService(); base = SpotBaseTool(svc)
    for action, expected in [
        ("move_forward",  "forward"),
        ("move_backward", "backward"),
        ("move_left",     "left"),
        ("move_right",    "right"),
    ]:
        r = await base.execute(action=action, distance_m=1.5)
        assert expected in r.lower(), f"'{expected}' não em: {r!r}"
    print("✓ spot_base: move_forward/backward/left/right")

async def test_base_rotate():
    svc = MockSpotService(); base = SpotBaseTool(svc)
    r = await base.execute(action="rotate_left",  angle_deg=90.0)
    assert "CCW" in r or "left" in r.lower()
    r = await base.execute(action="rotate_right", angle_deg=45.0)
    assert "CW" in r or "right" in r.lower()
    print("✓ spot_base: rotate_left / rotate_right")

async def test_base_navigate():
    svc = MockSpotService(); base = SpotBaseTool(svc)
    r = await base.execute(action="navigate_to_pose", x=3.0, y=1.5)
    assert "3.00" in r and "1.50" in r
    print("✓ spot_base: navigate_to_pose")

async def test_base_navigate_missing_xy():
    base = SpotBaseTool(MockSpotService())
    r = await base.execute(action="navigate_to_pose")
    assert "Erro" in r or "Error" in r
    print("✓ spot_base: navigate sem x/y → erro")

async def test_base_unknown_action():
    r = await SpotBaseTool(MockSpotService()).execute(action="teleport")
    assert "Erro" in r or "Error" in r
    print("✓ spot_base: ação inválida → erro")

async def test_base_schema():
    base = SpotBaseTool(MockSpotService())
    schema = base.to_schema()
    assert schema["type"] == "function"
    assert "action" in schema["function"]["parameters"]["properties"]
    print("✓ spot_base: JSON Schema válido")

# ---------------------------------------------------------------------------
# Testes: SpotArmTool — fases individuais
# ---------------------------------------------------------------------------

async def test_arm_segment_object():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="segment_object", object_name="cup")
    assert "1024" in r or "ponto" in r.lower()
    assert "segment_object" in svc.calls
    print("✓ spot_arm: segment_object")

async def test_arm_segment_no_name():
    r = await SpotArmTool(MockSpotService()).execute(action="segment_object")
    assert "Erro" in r or "Error" in r
    print("✓ spot_arm: segment_object sem object_name → erro")

async def test_arm_generate_grasps():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="generate_grasps")
    assert "5" in r or "pose" in r.lower()
    assert "generate_grasps" in svc.calls
    print("✓ spot_arm: generate_grasps")

async def test_arm_plan_trajectory():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="plan_trajectory")
    assert "cuRobo" in r or "trajetória" in r.lower() or "Trajetória" in r
    assert "plan_trajectory" in svc.calls
    print("✓ spot_arm: plan_trajectory")

async def test_arm_execute_grasp_close():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="execute_grasp", close_gripper=True)
    assert "closed" in r.lower() or "fechado" in r.lower()
    print("✓ spot_arm: execute_grasp (close)")

async def test_arm_execute_grasp_open():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="execute_grasp", close_gripper=False)
    assert "open" in r.lower() or "aberto" in r.lower()
    print("✓ spot_arm: execute_grasp (open)")

async def test_arm_move_to_home():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="move_to_home")
    assert "home" in r.lower()
    assert "move_to_home" in svc.calls
    print("✓ spot_arm: move_to_home")

async def test_arm_open_gripper():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="open_gripper")
    assert "open" in r.lower() or "aberto" in r.lower()
    print("✓ spot_arm: open_gripper")

async def test_arm_close_gripper():
    svc = MockSpotService(); arm = SpotArmTool(svc)
    r = await arm.execute(action="close_gripper")
    assert "close" in r.lower() or "fechado" in r.lower()
    print("✓ spot_arm: close_gripper")

async def test_arm_unknown_action():
    r = await SpotArmTool(MockSpotService()).execute(action="fly")
    assert "Erro" in r or "Error" in r
    print("✓ spot_arm: ação inválida → erro")

# ---------------------------------------------------------------------------
# Testes: full_grasp_pipeline
# ---------------------------------------------------------------------------

async def test_full_pipeline_success():
    svc = MockSpotService(); mem = MockMemory()
    arm = SpotArmTool(svc, episode_memory=mem)
    r = await arm.execute(action="full_grasp_pipeline", object_name="bottle")
    assert "bottle" in r
    assert "1." in r and "2." in r and "3." in r and "4." in r
    assert svc.calls == ["segment_object", "generate_grasps", "plan_trajectory", "execute_grasp"]
    assert len(mem.stored) == 4
    assert all(e["outcome"] == "success" for e in mem.stored)
    print("✓ full_grasp_pipeline: sucesso completo (4 fases)")

async def test_full_pipeline_fails_at_segment():
    svc = MockSpotService(fail_at="segment")
    arm = SpotArmTool(svc)
    r = await arm.execute(action="full_grasp_pipeline", object_name="cup")
    assert "Fase 1" in r or "segmentação" in r.lower()
    assert "generate_grasps" not in svc.calls
    print("✓ full_grasp_pipeline: falha na fase 1 para pipeline corretamente")

async def test_full_pipeline_fails_at_grasps():
    svc = MockSpotService(fail_at="grasps")
    arm = SpotArmTool(svc)
    r = await arm.execute(action="full_grasp_pipeline", object_name="cup")
    assert "Fase 2" in r or "GraspNet" in r or "grasp" in r.lower()
    assert "plan_trajectory" not in svc.calls
    print("✓ full_grasp_pipeline: falha na fase 2 para pipeline corretamente")

async def test_full_pipeline_fails_at_plan():
    svc = MockSpotService(fail_at="plan")
    arm = SpotArmTool(svc)
    r = await arm.execute(action="full_grasp_pipeline", object_name="cup")
    assert "Fase 3" in r or "cuRobo" in r
    assert "execute_grasp" not in svc.calls
    print("✓ full_grasp_pipeline: falha na fase 3 para pipeline corretamente")

async def test_full_pipeline_fails_at_execute():
    svc = MockSpotService(fail_at="execute")
    arm = SpotArmTool(svc)
    r = await arm.execute(action="full_grasp_pipeline", object_name="cup")
    assert "Fase 4" in r or "execução" in r.lower()
    print("✓ full_grasp_pipeline: falha na fase 4 para pipeline corretamente")

async def test_full_pipeline_no_name():
    r = await SpotArmTool(MockSpotService()).execute(action="full_grasp_pipeline")
    assert "Erro" in r or "Error" in r
    print("✓ full_grasp_pipeline: sem object_name → erro")

# ---------------------------------------------------------------------------
# Testes: SpotPerceptionTool
# ---------------------------------------------------------------------------

async def test_perception_camera_status_ok():
    svc = MockSpotService()
    perc = SpotPerceptionTool(svc)
    r = await perc.execute(action="camera_status")
    assert "prontas" in r.lower() or "ready" in r.lower()
    assert "RGB" in r and "Depth" in r and "CameraInfo" in r
    print("✓ spot_perception: camera_status — câmeras disponíveis")

async def test_perception_camera_status_missing():
    svc = MockSpotService()
    svc._latest_rgb = None   # simula câmera RGB ausente
    perc = SpotPerceptionTool(svc)
    r = await perc.execute(action="camera_status")
    assert "NÃO" in r or "not" in r.lower() or "falta" in r.lower()
    assert "rgb" in r.lower()
    print("✓ spot_perception: camera_status — câmera ausente detectada")

async def test_perception_schema():
    schema = SpotPerceptionTool(MockSpotService()).to_schema()
    assert schema["type"] == "function"
    assert "camera_status" in schema["function"]["parameters"]["properties"]["action"]["enum"]
    print("✓ spot_perception: JSON Schema válido")

async def test_perception_unknown_action():
    r = await SpotPerceptionTool(MockSpotService()).execute(action="lidar_scan")
    assert "Erro" in r or "Error" in r
    print("✓ spot_perception: ação inválida → erro")

# ---------------------------------------------------------------------------
# Testes: integração memória + factory
# ---------------------------------------------------------------------------

async def test_memory_hooks_all_groups():
    svc = MockSpotService(); mem = MockMemory()
    import pathlib as _pl
    import tempfile as _tf

    from roboclaw.embodied.spot.location_memory import LocationMemory as _LM
    with _tf.TemporaryDirectory() as _tmp:
        _lm = _LM(_pl.Path(_tmp) / "locs.json")
        tools = create_spot_tools(svc, mem, location_memory=_lm)
        base = next(t for t in tools if t.name == "spot_base")
        arm  = next(t for t in tools if t.name == "spot_arm")
        perc = next(t for t in tools if t.name == "spot_perception")

        await base.execute(action="move_forward", distance_m=1.0)
        await arm.execute(action="segment_object", object_name="cup")
        await perc.execute(action="camera_status")

        assert len(mem.stored) >= 3
    subtasks = [e["subtask"] for e in mem.stored]
    assert "move_forward" in subtasks
    assert "segment_object" in subtasks
    assert "camera_status" in subtasks
    assert len(mem.retrieved) >= 3
    print("✓ integração: store() e retrieve() disparados em todos os grupos")

async def test_factory_names():
    tools = create_spot_tools(MockSpotService())
    names = {t.name for t in tools}
    assert {"spot_base", "spot_arm", "spot_perception"}.issubset(names)
    assert len(tools) >= 3  # pode incluir spot_location
    print("✓ factory: 3 tools com nomes corretos")

async def test_all_schemas_valid():
    for tool in create_spot_tools(MockSpotService()):
        schema = tool.to_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        assert fn["parameters"]["required"] == ["action"]
    print("✓ todos os tools: JSON Schema válido para o ToolRegistry")

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    print("=" * 62)
    print("Spot Tools (grasp_pipeline) — testes isolados")
    print("=" * 62); print()

    tests = [
        # Base
        test_base_move_actions,
        test_base_rotate,
        test_base_navigate,
        test_base_navigate_missing_xy,
        test_base_unknown_action,
        test_base_schema,
        # Arm — fases individuais
        test_arm_segment_object,
        test_arm_segment_no_name,
        test_arm_generate_grasps,
        test_arm_plan_trajectory,
        test_arm_execute_grasp_close,
        test_arm_execute_grasp_open,
        test_arm_move_to_home,
        test_arm_open_gripper,
        test_arm_close_gripper,
        test_arm_unknown_action,
        # Full pipeline
        test_full_pipeline_success,
        test_full_pipeline_fails_at_segment,
        test_full_pipeline_fails_at_grasps,
        test_full_pipeline_fails_at_plan,
        test_full_pipeline_fails_at_execute,
        test_full_pipeline_no_name,
        # Percepção
        test_perception_camera_status_ok,
        test_perception_camera_status_missing,
        test_perception_schema,
        test_perception_unknown_action,
        # Integração
        test_memory_hooks_all_groups,
        test_factory_names,
        test_all_schemas_valid,
    ]

    failed = 0
    for t in tests:
        try:
            await t()
        except Exception:
            import traceback
            print(f"✗ {t.__name__}: FAILED")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 62)
    total = len(tests)
    if failed:
        print(f"{failed}/{total} FALHARAM.")
    else:
        print(f"Todos os {total} testes passaram.")
    print("=" * 62)
    return failed

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
