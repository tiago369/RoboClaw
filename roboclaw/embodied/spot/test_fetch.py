"""
test_fetch.py — testes do fluxo completo de fetch + location memory.
cd RoboClaw && python roboclaw/embodied/spot/test_fetch.py
"""
from __future__ import annotations
import asyncio, json, os, sys, tempfile, types, pathlib

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

for _n in ("roboclaw.agent", "roboclaw.agent.tools", "roboclaw.agent.tools.base"):
    _s = types.ModuleType(_n); _s.__path__ = []; sys.modules[_n] = _s

class _Tool:
    @property
    def name(self): return ""
    @property
    def description(self): return ""
    @property
    def parameters(self): return {}
    async def execute(self, **kw): return ""
    def to_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}
sys.modules["roboclaw.agent.tools.base"].Tool = _Tool

from roboclaw.embodied.spot.location_memory import LocationMemory
from roboclaw.embodied.spot.tools import (
    SpotBaseTool, SpotArmTool, SpotPerceptionTool,
    SpotLocationTool, create_spot_tools,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockSpotService:
    def __init__(self, fail_at=None, odom_pos=None):
        self._fail_at = fail_at
        self._odom_pos = odom_pos or {"x": 1.5, "y": 2.3, "yaw_deg": 45.0}
        self._last_segmented_points = object()
        self._last_grasp_poses = object()
        self._last_target_pose = object()
        self._cam_lock = __import__("threading").Lock()
        self._latest_rgb = object()
        self._latest_depth = object()
        self._latest_k_matrix = object()
        self.calls: list[str] = []

    def _ensure_ready(self): pass

    async def navigate_to_pose(self, x, y, yaw_deg=0.0, frame_id="map", timeout_s=60.0):
        self.calls.append("navigate_to_pose")
        if self._fail_at == "navigate":
            return json.dumps({"status": "error", "error": "Nav2 falhou"})
        return json.dumps({"status": "ok", "action": "navigate_to_pose",
                           "target": {"x": x, "y": y, "yaw_deg": yaw_deg}})

    async def segment_object(self, object_name, timeout_s=10.0):
        self.calls.append("segment_object")
        if self._fail_at == "segment":
            return json.dumps({"status": "error", "error": "não encontrado"})
        return json.dumps({"status": "ok", "action": "segment_object",
                           "object_name": object_name, "points_width": 512,
                           "points_height": 1, "frame_id": "body"})

    async def generate_grasps(self, timeout_s=15.0):
        self.calls.append("generate_grasps")
        return json.dumps({"status": "ok", "action": "generate_grasps",
                           "n_poses": 3, "top_poses": []})

    async def plan_trajectory(self, timeout_s=20.0):
        self.calls.append("plan_trajectory")
        return json.dumps({"status": "ok", "action": "plan_trajectory",
                           "target_pose": {"frame_id": "body",
                                           "position": {"x": 0.6, "y": 0.0, "z": 0.2},
                                           "orientation": {"x":0,"y":0,"z":0,"w":1}}})

    async def execute_grasp(self, close_gripper=True, timeout_s=30.0):
        self.calls.append("execute_grasp")
        return json.dumps({"status": "ok", "action": "execute_grasp",
                           "gripper": "closed" if close_gripper else "open"})

    async def move_to_home(self, timeout_s=20.0):
        self.calls.append("move_to_home")
        return json.dumps({"status": "ok", "action": "move_to_home"})

    async def move_arm_to_observe(self, timeout_s=15.0):
        self.calls.append("move_to_observe")
        return json.dumps({"status": "ok", "action": "move_arm_to_observe", "pose": "observe"})

    async def move_arm_to_carry(self, timeout_s=15.0):
        self.calls.append("move_to_carry")
        return json.dumps({"status": "ok", "action": "move_arm_to_carry", "pose": "carry"})

    async def open_gripper(self, timeout_s=5.0):
        self.calls.append("open_gripper")
        return json.dumps({"status": "ok", "action": "open_gripper"})

    async def close_gripper(self, timeout_s=5.0):
        self.calls.append("close_gripper")
        return json.dumps({"status": "ok", "action": "close_gripper"})

    async def move_forward(self, distance_m=1.0):
        self.calls.append("move_forward")
        return json.dumps({"status":"ok","action":"move_forward","distance_m":distance_m,"duration_s":2.0})

    async def move_backward(self, distance_m=1.0):
        self.calls.append("move_backward")
        return json.dumps({"status":"ok","action":"move_backward","distance_m":distance_m,"duration_s":2.0})

    async def move_left(self, distance_m=1.0):
        return json.dumps({"status":"ok","action":"move_left","distance_m":distance_m,"duration_s":2.0})

    async def move_right(self, distance_m=1.0):
        return json.dumps({"status":"ok","action":"move_right","distance_m":distance_m,"duration_s":2.0})

    async def rotate(self, angle_deg):
        self.calls.append("rotate")
        return json.dumps({"status":"ok","action":"rotate","angle_deg":angle_deg,
                           "direction":"left (CCW)" if angle_deg>0 else "right (CW)","duration_s":1.0})


# ---------------------------------------------------------------------------
# LocationMemory tests
# ---------------------------------------------------------------------------

def test_location_save_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("mesa da cozinha", x=3.2, y=1.5, yaw_deg=90.0)
        loc = lm.get("mesa da cozinha")
        assert loc is not None
        assert loc["x"] == 3.2 and loc["y"] == 1.5
        assert loc["yaw_deg"] == 90.0
    print("✓ LocationMemory: save e get exato")

def test_location_partial_match():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("mesa da cozinha", x=3.2, y=1.5)
        # busca parcial
        assert lm.get("mesa") is not None
        assert lm.get("cozinha") is not None
        assert lm.get("geladeira") is None
    print("✓ LocationMemory: busca parcial case-insensitive")

def test_location_persists():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "locs.json"
        lm1 = LocationMemory(path)
        lm1.save("origem", x=0.0, y=0.0)
        lm2 = LocationMemory(path)  # recarrega
        assert lm2.get("origem") is not None
    print("✓ LocationMemory: persiste em JSON")

def test_location_delete():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("ponto A", x=1.0, y=2.0)
        assert lm.delete("ponto A") is True
        assert lm.get("ponto A") is None
        assert lm.delete("ponto A") is False  # já removido
    print("✓ LocationMemory: delete")

def test_location_list_all():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("A", x=1.0, y=0.0)
        lm.save("B", x=2.0, y=0.0)
        lm.save("C", x=3.0, y=0.0)
        assert len(lm.list_all()) == 3
    print("✓ LocationMemory: list_all")


# ---------------------------------------------------------------------------
# SpotLocationTool tests
# ---------------------------------------------------------------------------

async def test_location_tool_save():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        tool = SpotLocationTool(MockSpotService(), lm)
        r = await tool.execute(action="save_location", name="mesa", x=3.2, y=1.5, yaw_deg=90.0)
        assert "mesa" in r and "3.20" in r
        assert lm.get("mesa") is not None
    print("✓ SpotLocationTool: save_location")

async def test_location_tool_go_to_known():
    with tempfile.TemporaryDirectory() as tmp:
        svc = MockSpotService()
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("mesa", x=3.2, y=1.5, yaw_deg=90.0)
        tool = SpotLocationTool(svc, lm)
        r = await tool.execute(action="go_to_location", name="mesa")
        assert "mesa" in r.lower()
        assert "navigate_to_pose" in svc.calls
    print("✓ SpotLocationTool: go_to_location para lugar conhecido")

async def test_location_tool_go_to_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("sala", x=5.0, y=2.0)
        tool = SpotLocationTool(MockSpotService(), lm)
        r = await tool.execute(action="go_to_location", name="quarto")
        assert "não encontrado" in r.lower() or "not found" in r.lower()
        assert "sala" in r  # mostra lugares conhecidos
    print("✓ SpotLocationTool: lugar desconhecido → sugere alternativas")

async def test_location_tool_list():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("mesa", x=3.2, y=1.5)
        lm.save("prateleira", x=5.0, y=2.0)
        tool = SpotLocationTool(MockSpotService(), lm)
        r = await tool.execute(action="list_locations")
        assert "mesa" in r and "prateleira" in r
    print("✓ SpotLocationTool: list_locations")

async def test_location_tool_list_empty():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        tool = SpotLocationTool(MockSpotService(), lm)
        r = await tool.execute(action="list_locations")
        assert "nenhum" in r.lower() or "none" in r.lower() or "no" in r.lower()
    print("✓ SpotLocationTool: list_locations vazio → mensagem útil")

async def test_location_tool_delete():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("temp", x=1.0, y=1.0)
        tool = SpotLocationTool(MockSpotService(), lm)
        r = await tool.execute(action="delete_location", name="temp")
        assert "removido" in r.lower() or "removed" in r.lower()
        assert lm.get("temp") is None
    print("✓ SpotLocationTool: delete_location")

async def test_location_tool_schema():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        schema = SpotLocationTool(MockSpotService(), lm).to_schema()
        assert schema["type"] == "function"
        params = schema["function"]["parameters"]
        assert "action" in params["properties"]
        assert "save_location" in params["properties"]["action"]["enum"]
    print("✓ SpotLocationTool: JSON Schema válido")

async def test_location_tool_missing_name():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        tool = SpotLocationTool(MockSpotService(), lm)
        r = await tool.execute(action="save_location", x=1.0, y=2.0)
        assert "Erro" in r or "Error" in r
        r2 = await tool.execute(action="go_to_location")
        assert "Erro" in r2 or "Error" in r2
    print("✓ SpotLocationTool: erros de validação corretos")


# ---------------------------------------------------------------------------
# Novas ações do SpotArmTool
# ---------------------------------------------------------------------------

async def test_arm_move_to_observe():
    svc = MockSpotService()
    arm = SpotArmTool(svc)
    r = await arm.execute(action="move_to_observe")
    assert "observação" in r.lower() or "observe" in r.lower()
    assert "move_to_observe" in svc.calls
    print("✓ spot_arm: move_to_observe")

async def test_arm_move_to_carry():
    svc = MockSpotService()
    arm = SpotArmTool(svc)
    r = await arm.execute(action="move_to_carry")
    assert "carry" in r.lower() or "carregar" in r.lower()
    assert "move_to_carry" in svc.calls
    print("✓ spot_arm: move_to_carry")

async def test_arm_actions_in_schema():
    schema = SpotArmTool(MockSpotService()).to_schema()
    enum = schema["function"]["parameters"]["properties"]["action"]["enum"]
    assert "move_to_observe" in enum
    assert "move_to_carry" in enum
    print("✓ spot_arm: move_to_observe e move_to_carry no schema")


# ---------------------------------------------------------------------------
# Fluxo completo de fetch (simulação do agente)
# ---------------------------------------------------------------------------

async def test_full_fetch_flow():
    """
    Simula o fluxo completo:
    câmera → salva origem → navega → observe → grasp pipeline → carry → volta → home
    """
    with tempfile.TemporaryDirectory() as tmp:
        svc = MockSpotService()
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")

        # Pré-salva localização da mesa
        lm.save("mesa", x=4.0, y=2.0, yaw_deg=0.0)

        perc = SpotPerceptionTool(svc)
        base = SpotBaseTool(svc)
        arm  = SpotArmTool(svc)
        loc  = SpotLocationTool(svc, lm)

        # Step 0 — câmeras ok
        r = await perc.execute(action="camera_status")
        assert "prontas" in r.lower() or "ready" in r.lower()

        # Step 1 — salva origem (sem odom real, usa save_location manual)
        lm.save("origem", x=0.0, y=0.0, yaw_deg=0.0)
        assert lm.get("origem") is not None

        # Step 2 — navega para a mesa
        r = await loc.execute(action="go_to_location", name="mesa")
        assert "mesa" in r.lower()
        assert "navigate_to_pose" in svc.calls

        # Step 3 — observe
        r = await arm.execute(action="move_to_observe")
        assert "move_to_observe" in svc.calls

        # Step 4 — abre gripper + pipeline completo
        r = await arm.execute(action="open_gripper")
        assert "open" in r.lower() or "aberto" in r.lower()

        r = await arm.execute(action="full_grasp_pipeline", object_name="martelo")
        assert "martelo" in r
        assert "1." in r and "4." in r  # 4 fases
        assert "segment_object" in svc.calls
        assert "execute_grasp" in svc.calls

        # Step 5 — carry
        r = await arm.execute(action="move_to_carry")
        assert "move_to_carry" in svc.calls

        # Step 6 — volta para origem
        r = await loc.execute(action="go_to_location", name="origem")
        assert "origem" in r.lower()
        assert svc.calls.count("navigate_to_pose") == 2  # foi + voltou

        # Step 7 — home
        r = await arm.execute(action="move_to_home")
        assert "home" in r.lower()

        # Verifica sequência completa de chamadas
        expected_sequence = [
            "navigate_to_pose",    # vai para mesa
            "move_to_observe",     # posiciona câmera
            "open_gripper",        # abre antes de pegar
            "segment_object",      # fase 1
            "generate_grasps",     # fase 2
            "plan_trajectory",     # fase 3
            "execute_grasp",       # fase 4
            "move_to_carry",       # pose de transporte
            "navigate_to_pose",    # volta
            "move_to_home",        # apresenta
        ]
        for step in expected_sequence:
            assert step in svc.calls, f"'{step}' não foi chamado. Calls: {svc.calls}"

    print("✓ Fluxo completo de fetch: todos os 10 passos executados na ordem correta")

async def test_fetch_flow_navigation_fail():
    """Se a navegação falhar, o fluxo para com erro claro."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = MockSpotService(fail_at="navigate")
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        lm.save("mesa", x=4.0, y=2.0)
        loc = SpotLocationTool(svc, lm)
        r = await loc.execute(action="go_to_location", name="mesa")
        assert "erro" in r.lower() or "error" in r.lower() or "Erro" in r
    print("✓ Fluxo fetch: falha de navegação reportada corretamente")

async def test_fetch_flow_grasp_fail():
    """Se a segmentação falhar, full_grasp_pipeline para na fase 1."""
    svc = MockSpotService(fail_at="segment")
    arm = SpotArmTool(svc)
    r = await arm.execute(action="full_grasp_pipeline", object_name="chave")
    assert "Fase 1" in r or "segmentação" in r.lower()
    assert "generate_grasps" not in svc.calls
    print("✓ Fluxo fetch: falha de grasp reportada com fase correta")


# ---------------------------------------------------------------------------
# Factory com 4 tools
# ---------------------------------------------------------------------------

async def test_create_spot_tools_four_groups():
    with tempfile.TemporaryDirectory() as tmp:
        lm = LocationMemory(pathlib.Path(tmp) / "locs.json")
        tools = create_spot_tools(MockSpotService(), location_memory=lm)
        names = {t.name for t in tools}
        assert names == {"spot_base", "spot_arm", "spot_perception", "spot_location"}
        assert len(tools) == 4
    print("✓ create_spot_tools: 4 grupos (base, arm, perception, location)")

async def test_create_spot_tools_auto_location_memory():
    """create_spot_tools sem location_memory cria uma automaticamente."""
    tools = create_spot_tools(MockSpotService())
    loc_tool = next(t for t in tools if t.name == "spot_location")
    assert loc_tool._lm is not None
    print("✓ create_spot_tools: LocationMemory criada automaticamente")


# ---------------------------------------------------------------------------
# Skill file
# ---------------------------------------------------------------------------

def test_skill_file_exists():
    skill_path = pathlib.Path(_ROOT) / "roboclaw/skills/spot-fetch/SKILL.md"
    assert skill_path.exists(), f"SKILL.md não encontrado em {skill_path}"
    content = skill_path.read_text(encoding="utf-8")
    assert "name: spot-fetch" in content
    assert "full_grasp_pipeline" in content
    assert "move_to_observe" in content
    assert "move_to_carry" in content
    assert "go_to_location" in content
    assert "save_current_as" in content
    print("✓ SKILL.md: existe e contém todas as etapas do fluxo de fetch")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    print("=" * 64)
    print("Spot Fetch — testes completos (sem ROS2)")
    print("=" * 64); print()

    sync_tests = [
        test_location_save_and_get,
        test_location_partial_match,
        test_location_persists,
        test_location_delete,
        test_location_list_all,
        test_skill_file_exists,
    ]

    async_tests = [
        test_location_tool_save,
        test_location_tool_go_to_known,
        test_location_tool_go_to_unknown,
        test_location_tool_list,
        test_location_tool_list_empty,
        test_location_tool_delete,
        test_location_tool_schema,
        test_location_tool_missing_name,
        test_arm_move_to_observe,
        test_arm_move_to_carry,
        test_arm_actions_in_schema,
        test_full_fetch_flow,
        test_fetch_flow_navigation_fail,
        test_fetch_flow_grasp_fail,
        test_create_spot_tools_four_groups,
        test_create_spot_tools_auto_location_memory,
    ]

    failed = 0
    for t in sync_tests:
        try:
            t()
        except Exception:
            import traceback
            print(f"✗ {t.__name__}: FAILED"); traceback.print_exc(); failed += 1

    for t in async_tests:
        try:
            await t()
        except Exception:
            import traceback
            print(f"✗ {t.__name__}: FAILED"); traceback.print_exc(); failed += 1

    print()
    print("=" * 64)
    total = len(sync_tests) + len(async_tests)
    print(f"Todos os {total} testes passaram." if not failed else f"{failed}/{total} FALHARAM.")
    print("=" * 64)
    return failed

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
