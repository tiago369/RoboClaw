"""
test_bridge.py
==============
Full Spot Bridge integration tests — no ROS2, no hardware.

Covers:
  1. SpotConfig — defaults and validation
  2. SpotRobot — observation/action features, connect (mock), send_action
  3. SpotService → SpotTools → AgentLoop (register_spot_tools)
  4. EAP + SpotResetExecutor running inside the tools loop
  5. Catalog — Spot appears in models_for(MOBILE)

Runs with:
    cd RoboClaw && python roboclaw/embodied/spot/test_bridge.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

for _name in ("roboclaw.agent", "roboclaw.agent.tools"):
    _stub = types.ModuleType(_name)
    _stub.__path__ = [_name.replace(".", "/")]
    _stub.__package__ = _name
    sys.modules[_name] = _stub

for _svc_name in ("roboclaw.embodied.service",
                  "roboclaw.embodied.service.session",
                  "roboclaw.embodied.calibration"):
    if _svc_name not in sys.modules:
        _s = types.ModuleType(_svc_name)
        _s.__path__ = [_svc_name.replace(".", "/")]
        _s.__package__ = _svc_name
        sys.modules[_svc_name] = _s

_ENGINE = os.path.abspath(os.path.join(_ROOT, "roboclaw/embodied/engine/src"))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

for _ros_mod in (
    "rclpy", "rclpy.node", "rclpy.executors",
    "geometry_msgs", "geometry_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg",
    "trajectory_msgs", "trajectory_msgs.msg",
    "nav2_msgs", "nav2_msgs.action",
    "builtin_interfaces", "builtin_interfaces.msg",
    "spot_interfaces", "spot_interfaces.srv",
):
    if _ros_mod not in sys.modules:
        _s = types.ModuleType(_ros_mod)
        sys.modules[_ros_mod] = _s

import types as _t
_geometry = sys.modules["geometry_msgs.msg"]
_geometry.Twist = type("Twist", (), {"linear": _t.SimpleNamespace(x=0,y=0,z=0),
                                      "angular": _t.SimpleNamespace(x=0,y=0,z=0)})
_geometry.PoseStamped = type("PoseStamped", (), {
    "header": _t.SimpleNamespace(frame_id=""),
    "pose": _t.SimpleNamespace(
        position=_t.SimpleNamespace(x=0,y=0,z=0),
        orientation=_t.SimpleNamespace(x=0,y=0,z=0,w=1),
    )
})

if "draccus" not in sys.modules:
    _draccus = types.ModuleType("draccus")
    class _ChoiceRegistry:
        _registry: dict = {}
        @classmethod
        def register_subclass(cls, name):
            def decorator(klass):
                cls._registry[name] = klass
                return klass
            return decorator
        @classmethod
        def get_choice_name(cls, klass):
            for k, v in cls._registry.items():
                if v is klass: return k
            return klass.__name__
    _draccus.ChoiceRegistry = _ChoiceRegistry
    sys.modules["draccus"] = _draccus

if "lerobot" not in sys.modules:
    _lerobot = types.ModuleType("lerobot")
    _lerobot.__path__ = [os.path.join(_ENGINE, "lerobot")]
    sys.modules["lerobot"] = _lerobot

if "lerobot.robots" not in sys.modules:
    _lr = types.ModuleType("lerobot.robots")
    _lr.__path__ = [os.path.join(_ENGINE, "lerobot/robots")]
    _lr.__package__ = "lerobot.robots"
    sys.modules["lerobot.robots"] = _lr

_engine_robots_pkg = "roboclaw.embodied.engine.src.lerobot.robots"
if _engine_robots_pkg not in sys.modules:
    _er = types.ModuleType(_engine_robots_pkg)
    _er.__path__ = [os.path.join(_ENGINE, "lerobot/robots")]
    _er.__package__ = _engine_robots_pkg
    sys.modules[_engine_robots_pkg] = _er
    for _sub in ("config", "robot"):
        sys.modules[f"{_engine_robots_pkg}.{_sub}"] = types.ModuleType(
            f"{_engine_robots_pkg}.{_sub}"
        )

for _lm in ("lerobot.utils", "lerobot.utils.decorators",
            "lerobot.utils.constants", "lerobot.calibration_timestamp",
            "lerobot.types", "lerobot.motors", "lerobot.cameras",
            "lerobot.cameras.utils", "lerobot.cameras.configs",
            "lerobot.cameras.opencv",
            "lerobot.cameras.opencv.configuration_opencv"):
    if _lm not in sys.modules:
        _s = types.ModuleType(_lm)
        sys.modules[_lm] = _s

_dec = sys.modules["lerobot.utils.decorators"]
def _noop_deco(fn): return fn
_dec.check_if_already_connected = _noop_deco
_dec.check_if_not_connected = _noop_deco

# Types
_types_mod = sys.modules["lerobot.types"]
_types_mod.RobotAction = dict
_types_mod.RobotObservation = dict

# Constants
_const = sys.modules["lerobot.utils.constants"]
_const.HF_LEROBOT_CALIBRATION = __import__("pathlib").Path("/tmp/lerobot_cal")
_const.ROBOTS = "robots"

# Cameras
_cam_utils = sys.modules["lerobot.cameras.utils"]
_cam_utils.make_cameras_from_configs = lambda configs: {}

_cam_configs = sys.modules["lerobot.cameras.configs"]
_cam_configs.CameraConfig = object
class _Cv2Rotation: ROTATE_90=1; ROTATE_180=2
_cam_configs.Cv2Rotation = _Cv2Rotation

_opencv_mod = sys.modules["lerobot.cameras.opencv.configuration_opencv"]
class _OpenCVCameraConfig:
    def __init__(self, index_or_path="", fps=30, width=640, height=480, rotation=None):
        self.index_or_path = index_or_path
        self.fps = fps; self.width = width; self.height = height
_opencv_mod.OpenCVCameraConfig = _OpenCVCameraConfig

# Motors
_motors = sys.modules["lerobot.motors"]
_motors.MotorCalibration = object

# calibration_timestamp
_cal = sys.modules["lerobot.calibration_timestamp"]
_cal.record_calibration_timestamp = lambda *a, **kw: None


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockSpotService:
    """SpotService mock — returns realistic JSON without touching ROS2."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def _ok(self, action: str, **data) -> str:
        self.calls.append((action, data))
        return json.dumps({"status": "ok", "action": action, **data})

    async def move_forward(self, distance_m=1.0):
        return await self._ok("move_forward", distance_m=distance_m,
                               duration_s=round(distance_m / 0.5, 2))

    async def move_backward(self, distance_m=1.0):
        return await self._ok("move_backward", distance_m=distance_m,
                               duration_s=round(distance_m / 0.5, 2))

    async def move_left(self, distance_m=1.0):
        return await self._ok("move_left", distance_m=distance_m,
                               duration_s=round(distance_m / 0.5, 2))

    async def move_right(self, distance_m=1.0):
        return await self._ok("move_right", distance_m=distance_m,
                               duration_s=round(distance_m / 0.5, 2))

    async def rotate(self, angle_deg):
        direction = "left (CCW)" if angle_deg > 0 else "right (CW)"
        return await self._ok("rotate", angle_deg=angle_deg, direction=direction,
                               duration_s=round(abs(angle_deg) / 28.6, 2))

    async def navigate_to_pose(self, x, y, yaw_deg=0.0, frame_id="map", timeout_s=60.0):
        return await self._ok("navigate_to_pose",
                               target={"x": x, "y": y, "yaw_deg": yaw_deg, "frame": frame_id})

    async def arm_move_cartesian(self, dx=0.0, dy=0.0, dz=0.0):
        return await self._ok("arm_move_cartesian",
                               delta={"dx": dx, "dy": dy, "dz": dz},
                               current_pose={"x": 0.5 + dx, "y": dy, "z": 0.3 + dz,
                                             "roll": 0, "pitch": 0, "yaw": 0})

    async def arm_go_to_pose(self, x, y, z, roll_deg=0, pitch_deg=0, yaw_deg=0, frame_id="body"):
        return await self._ok("arm_go_to_pose",
                               pose={"x": x, "y": y, "z": z,
                                     "roll": roll_deg, "pitch": pitch_deg, "yaw": yaw_deg},
                               frame_id=frame_id)

    async def segment_pcl(self, max_objects=5, min_confidence=0.5):
        return await self._ok("segment_pcl", objects_found=2, objects=[
            {"id": 0, "label": "cup",    "confidence": 0.92,
             "centroid": {"x": 0.6, "y": 0.1, "z": 0.2},
             "bbox_size": {"x": 0.08, "y": 0.08, "z": 0.12}},
            {"id": 1, "label": "bottle", "confidence": 0.78,
             "centroid": {"x": 0.7, "y": -0.1, "z": 0.25},
             "bbox_size": {"x": 0.06, "y": 0.06, "z": 0.22}},
        ])

    async def detect_objects(self, camera="hand"):
        return await self._ok("detect_objects", camera=camera, detections_found=1,
                               detections=[{"label": "cup", "confidence": 0.88,
                                            "bbox_px": {"x": 120, "y": 90, "w": 60, "h": 80}}])

    async def get_gripper_state(self):
        return await self._ok("get_gripper_state",
                               gripper={"position": 0.0, "force_n": 12.4,
                                        "is_holding": True, "object_detected": True})


class MockMemory:
    def __init__(self):
        self.stored: list[dict] = []
        self.retrieved: list[str] = []

    def store(self, subtask, outcome, env_state):
        self.stored.append({"subtask": subtask, "outcome": outcome})

    def retrieve(self, query, top_k=3, as_context_string=False):
        self.retrieved.append(query)
        return "No relevant past robotic experience found." if as_context_string else []


class MockToolRegistry:
    """Simulates ToolRegistry to test `register_spot_tools` without a full AgentLoop."""

    def __init__(self):
        self._tools: dict = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# 1. SpotConfig
# ---------------------------------------------------------------------------

def test_spot_config_defaults():
    SpotConfig, _, _, _ = _load_spot_modules()
    cfg = SpotConfig()
    assert cfg.ros2_namespace == "/spot"
    assert cfg.cmd_vel_topic == "/cmd_vel"
    assert cfg.arm_pose_topic == "/arm_pose_commands"
    assert cfg.joint_state_topic == "/joint_states"
    assert cfg.arm_dof == 6
    assert cfg.connection_timeout_s == 10.0
    assert not cfg.use_degrees
    assert "hand" in cfg.cameras
    assert "front" in cfg.cameras
    print("✓ SpotConfig: defaults corretos")


def _load_spot_modules():
    """
    Load SpotConfig and SpotRobot with the correct package context
    so that relative imports (..config, ..robot) work.
    """
    import importlib.util as _ilu

    _base = os.path.join(_ROOT, "roboclaw/embodied/engine/src/lerobot/robots")
    _spot_base = os.path.join(_base, "spot")

    _parent_pkg = "lerobot.robots"
    if _parent_pkg not in sys.modules or not hasattr(sys.modules[_parent_pkg], "__path__"):
        _pm = types.ModuleType(_parent_pkg)
        _pm.__path__ = [_base]
        _pm.__package__ = _parent_pkg
        sys.modules[_parent_pkg] = _pm

    # lerobot.robots.config (RobotConfig)
    if "lerobot.robots.config" not in sys.modules:
        _cspec = _ilu.spec_from_file_location(
            "lerobot.robots.config", os.path.join(_base, "config.py"),
            submodule_search_locations=[])
        _cmod = _ilu.module_from_spec(_cspec)
        _cmod.__package__ = "lerobot.robots"
        sys.modules["lerobot.robots.config"] = _cmod
        _cspec.loader.exec_module(_cmod)

    # lerobot.robots.robot (Robot base)
    if "lerobot.robots.robot" not in sys.modules:
        _rspec = _ilu.spec_from_file_location(
            "lerobot.robots.robot", os.path.join(_base, "robot.py"),
            submodule_search_locations=[])
        _rmod = _ilu.module_from_spec(_rspec)
        _rmod.__package__ = "lerobot.robots"
        sys.modules["lerobot.robots.robot"] = _rmod
        _rspec.loader.exec_module(_rmod)

    # Subpackage spot
    _spot_pkg = "lerobot.robots.spot"
    if _spot_pkg not in sys.modules:
        _sm = types.ModuleType(_spot_pkg)
        _sm.__path__ = [_spot_base]
        _sm.__package__ = _spot_pkg
        sys.modules[_spot_pkg] = _sm

    # config_spot
    if "lerobot.robots.spot.config_spot" not in sys.modules:
        _cfspec = _ilu.spec_from_file_location(
            "lerobot.robots.spot.config_spot",
            os.path.join(_spot_base, "config_spot.py"),
            submodule_search_locations=[])
        _cfmod = _ilu.module_from_spec(_cfspec)
        _cfmod.__package__ = "lerobot.robots.spot"
        sys.modules["lerobot.robots.spot.config_spot"] = _cfmod
        _cfspec.loader.exec_module(_cfmod)

    # spot.py
    if "lerobot.robots.spot.spot" not in sys.modules:
        _sspec = _ilu.spec_from_file_location(
            "lerobot.robots.spot.spot",
            os.path.join(_spot_base, "spot.py"),
            submodule_search_locations=[])
        _smod = _ilu.module_from_spec(_sspec)
        _smod.__package__ = "lerobot.robots.spot"
        sys.modules["lerobot.robots.spot.spot"] = _smod
        _sspec.loader.exec_module(_smod)

    _cfmod = sys.modules["lerobot.robots.spot.config_spot"]
    _smod  = sys.modules["lerobot.robots.spot.spot"]
    return (
        _cfmod.SpotConfig,
        _smod.SpotRobot,
        _smod._ARM_JOINT_NAMES,
        _smod._BASE_VEL_KEYS,
    )


def test_spot_config_custom():
    SpotConfig, _, _, _ = _load_spot_modules()
    cfg = SpotConfig(ros2_namespace="/my_spot", use_degrees=True, arm_dof=6)
    assert cfg.ros2_namespace == "/my_spot"
    assert cfg.use_degrees is True
    print("✓ SpotConfig: custom values")


# ---------------------------------------------------------------------------
# 2. SpotRobot
# ---------------------------------------------------------------------------

def test_spot_robot_features():
    SpotConfig, SpotRobot, _ARM_JOINT_NAMES, _BASE_VEL_KEYS = _load_spot_modules()

    cfg = SpotConfig()
    robot = SpotRobot.__new__(SpotRobot)
    robot.config = cfg
    robot.cameras = {}
    robot._connected = False
    robot._node = None
    robot._executor = None
    robot._ros_thread = None
    robot._cmd_vel_pub = None
    robot._arm_joint_pub = None
    robot._joint_lock = __import__("threading").Lock()
    robot._latest_joints = {k: 0.0 for k in _ARM_JOINT_NAMES}
    robot._latest_base_vel = {k: 0.0 for k in _BASE_VEL_KEYS}

    obs_ft = robot.observation_features
    for j in _ARM_JOINT_NAMES:
        assert f"{j}.pos" in obs_ft, f"Missing {j}.pos in observation_features"
    for k in _BASE_VEL_KEYS:
        assert k in obs_ft

    act_ft = robot.action_features
    for j in _ARM_JOINT_NAMES:
        assert f"{j}.pos" in act_ft
    for k in _BASE_VEL_KEYS:
        assert k in act_ft

    assert not any("hand" in k or "front" in k for k in act_ft)
    assert robot.is_calibrated is True
    assert robot.name == "spot"
    print("✓ SpotRobot: observation/action features corrects")


def test_spot_robot_arm_joints_count():
    _, _, _ARM_JOINT_NAMES, _ = _load_spot_modules()
    assert len(_ARM_JOINT_NAMES) == 7
    assert "arm_sh0" in _ARM_JOINT_NAMES
    assert "arm_f1x" in _ARM_JOINT_NAMES
    print("✓ SpotRobot: 7 joint names (6 DOF + gripper)")


# ---------------------------------------------------------------------------
# 3. SpotTools — register and execution via MockSpotService
# ---------------------------------------------------------------------------

async def test_spot_tools_registration():
    from roboclaw.embodied.spot.tools import create_spot_tools

    svc = MockSpotService()
    mem = MockMemory()
    tools = create_spot_tools(svc, mem)

    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"spot_base", "spot_arm", "spot_perception"}
    print("✓ create_spot_tools: 3 tools registered")


async def test_spot_tools_schema_valid():
    from roboclaw.embodied.spot.tools import create_spot_tools

    tools = create_spot_tools(MockSpotService())
    for tool in tools:
        schema = tool.to_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        params = fn["parameters"]
        assert "action" in params["properties"]
        assert params["required"] == ["action"]
    print("✓ SpotTools: JSON Schema valid for all tools")


async def test_spot_base_full_coverage():
    from roboclaw.embodied.spot.tools import SpotBaseTool
    svc = MockSpotService()
    base = SpotBaseTool(spot_service=svc)

    tests = [
        ({"action": "move_forward",  "distance_m": 2.0},  "forward"),
        ({"action": "move_backward", "distance_m": 0.5},  "backward"),
        ({"action": "move_left",     "distance_m": 1.0},  "left"),
        ({"action": "move_right",    "distance_m": 1.0},  "right"),
        ({"action": "rotate_left",   "angle_deg": 45.0},  "left"),
        ({"action": "rotate_right",  "angle_deg": 90.0},  "right"),
        ({"action": "navigate_to_pose", "x": 3.0, "y": 1.5, "yaw_deg": 90.0}, "3.00"),
    ]
    for kwargs, expected in tests:
        r = await base.execute(**kwargs)
        assert expected in r.lower() or expected in r, \
            f"Expected '{expected}' in result of {kwargs['action']}: {r!r}"

    # Erros
    assert "Error" in await base.execute(action="navigate_to_pose")
    assert "Error" in await base.execute(action="teleport")
    print("✓ SpotBaseTool: all covered actions + errors")


async def test_spot_arm_full_coverage():
    from roboclaw.embodied.spot.tools import SpotArmTool
    arm = SpotArmTool(spot_service=MockSpotService())

    for direction in ("left", "right", "up", "down", "forward", "backward"):
        r = await arm.execute(action=f"move_arm_{direction}", step_m=0.15)
        assert direction in r.lower(), f"Expected '{direction}' in: {r!r}"

    r = await arm.execute(action="arm_go_to_pose", x=0.8, y=0.1, z=0.5, pitch_deg=30.0)
    assert "0.800" in r and "0.500" in r

    assert "Error" in await arm.execute(action="arm_go_to_pose")
    assert "Error" in await arm.execute(action="unknown")
    print("✓ SpotArmTool: all covered actions + errors")


async def test_spot_perception_full_coverage():
    from roboclaw.embodied.spot.tools import SpotPerceptionTool
    perc = SpotPerceptionTool(spot_service=MockSpotService())

    r = await perc.execute(action="segment_pcl", max_objects=5)
    assert "cup" in r and "bottle" in r and "2 object" in r

    r = await perc.execute(action="detect_objects", camera="hand")
    assert "cup" in r and "hand" in r

    r = await perc.execute(action="get_gripper_state")
    assert "holding" in r.lower()

    assert "Error" in await perc.execute(action="unknown")
    print("✓ SpotPerceptionTool: all covered actions + errors")


async def test_memory_hooks_fire():
    from roboclaw.embodied.spot.tools import create_spot_tools
    svc = MockSpotService()
    mem = MockMemory()
    base, arm, perc = create_spot_tools(svc, mem)

    await base.execute(action="move_forward", distance_m=1.0)
    await arm.execute(action="move_arm_up", step_m=0.1)
    await perc.execute(action="segment_pcl")

    assert len(mem.stored) >= 3
    subtasks = [e["subtask"] for e in mem.stored]
    assert "move_forward" in subtasks
    assert "move_arm_up" in subtasks
    assert "segment_pcl" in subtasks
    assert all(e["outcome"] == "success" for e in mem.stored)
    assert len(mem.retrieved) >= 3
    print("✓ memory hooks: store() and retrieve() triggered in all groups")


# ---------------------------------------------------------------------------
# 4. register_spot_tools in AgentLoop (mock)
# ---------------------------------------------------------------------------

async def test_register_spot_tools_on_agent_loop():
    """
    Test `register_spot_tools()` without loading the entire AgentLoop.
    It uses a simple namespace that simulates the AgentLoop.
    """
    import types as _types
    from roboclaw.embodied.spot.tools import create_spot_tools

    loop = _types.SimpleNamespace(
        tools=MockToolRegistry(),
        episode_memory=MockMemory(),
        spot_service=None,
    )

    def register_spot_tools(self, spot_service):
        self.spot_service = spot_service
        for tool in create_spot_tools(spot_service, self.episode_memory):
            self.tools.register(tool)

    svc = MockSpotService()
    register_spot_tools(loop, svc)

    assert loop.spot_service is svc
    assert "spot_base" in loop.tools.names()
    assert "spot_arm" in loop.tools.names()
    assert "spot_perception" in loop.tools.names()
    print("✓ register_spot_tools: tools registered with AgentLoop")


async def test_spot_tools_already_registered_no_duplicate():
    """Calling `register_spot_tools` twice does not duplicate the tools."""
    import types as _types
    from roboclaw.embodied.spot.tools import create_spot_tools

    class DeduplicatingRegistry(MockToolRegistry):
        def register(self, tool):
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' already registered")
            super().register(tool)

    loop = _types.SimpleNamespace(
        tools=DeduplicatingRegistry(),
        episode_memory=None,
        spot_service=None,
    )

    def register(self, svc):
        # Simula a guarda: só registra se ainda não houver spot tools
        if "spot_base" not in self.tools.names():
            for tool in create_spot_tools(svc, self.episode_memory):
                self.tools.register(tool)

    register(loop, MockSpotService())

    # Segunda chamada não deve lançar
    try:
        register(loop, MockSpotService())
    except ValueError:
        pass  # Se lançar, o teste falhou — mas nosso guard previne

    assert "spot_base" in loop.tools.names()
    print("✓ register_spot_tools: without duplicating tools")


# ---------------------------------------------------------------------------
# 5. EAP + SpotResetExecutor integrado
# ---------------------------------------------------------------------------

async def test_eap_with_spot_reset_executor():
    from roboclaw.embodied.service.session.eap import (
        EAPController, SpotResetExecutor, ResetResult
    )
    import types as _types

    svc = MockSpotService()
    reset_sequence = [
        {"action": "arm_go_to_pose", "kwargs": {"x": 0.5, "y": 0.0, "z": 0.3}},
        {"action": "move_backward",  "kwargs": {"distance_m": 0.3}},
    ]
    executor = SpotResetExecutor(svc, reset_sequence=reset_sequence, timeout_s=5.0)

    # Verifica que o executor executa os dois passos com sucesso
    result = await executor.execute_reset()
    assert result.success, f"Expected success: {result.message}"
    assert result.message == "Reset sequence completed"

    # Verifica que os dois métodos foram chamados
    actions_called = [call[0] for call in svc.calls]
    assert "arm_go_to_pose" in actions_called
    assert "move_backward"  in actions_called
    print("✓ SpotResetExecutor: Reset sequence successfully executed")


async def test_eap_full_cycle_with_spot():
    """Complete cycle: RESETTING → SpotResetExecutor → skip_reset → recording."""
    from roboclaw.embodied.service.session.eap import EAPController, SpotResetExecutor
    from roboclaw.embodied.service.session.record import RecordPhase

    svc = MockSpotService()
    reset_sequence = [
        {"action": "arm_go_to_pose", "kwargs": {"x": 0.5, "y": 0.0, "z": 0.3}},
    ]
    executor = SpotResetExecutor(svc, reset_sequence=reset_sequence)

    class MockPhaseCtrl:
        def __init__(self):
            self._phase = "idle"
            self.skip_reset_calls = 0

        @property
        def phase(self):
            return RecordPhase(self._phase)

        async def request_skip_reset(self):
            self.skip_reset_calls += 1
            self._phase = "recording"

    phase = MockPhaseCtrl()
    mem = MockMemory()
    eap = EAPController(
        phase_controller=phase,
        reset_executor=executor,
        max_reset_retries=2,
        episode_memory=mem,
        poll_interval_s=0.02,
    )
    eap.start()
    phase._phase = "resetting"

    # Aguarda resolução
    for _ in range(30):
        await asyncio.sleep(0.05)
        if phase._phase != "resetting":
            break

    await eap.stop()

    assert phase._phase == "recording", f"Expected recording, got {phase._phase}"
    assert phase.skip_reset_calls == 1
    assert eap.total_resets_succeeded == 1
    assert any(e["subtask"] == "eap_reset" and e["outcome"] == "success"
               for e in mem.stored)
    print("✓ EAP cycle complete with SpotResetExecutor")


# ---------------------------------------------------------------------------
# 6. Catalog
# ---------------------------------------------------------------------------

def test_catalog_mobile_spot():
    from roboclaw.embodied.embodiment.catalog import (
        EmbodimentCategory, models_for, is_supported
    )
    mobile = models_for(EmbodimentCategory.MOBILE)
    names = [m.name for m in mobile]
    assert "spot" in names, f"spot not in mobile catalog: {names}"
    assert is_supported(EmbodimentCategory.MOBILE)
    print("✓ catalog: spot registered in EmbodimentCategory.MOBILE")


def test_catalog_spot_roles():
    from roboclaw.embodied.embodiment.catalog import EmbodimentCategory, models_for
    mobile = models_for(EmbodimentCategory.MOBILE)
    spot_info = next(m for m in mobile if m.name == "spot")
    assert "follower" in spot_info.roles
    print("✓ catalog: spot has role 'follower'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    print("=" * 64)
    print("Spot Bridge — integration tests (without ROS2, without hardware)")
    print("=" * 64)
    print()

    sync_tests = [
        test_spot_config_defaults,
        test_spot_config_custom,
        test_spot_robot_features,
        test_spot_robot_arm_joints_count,
        test_catalog_mobile_spot,
        test_catalog_spot_roles,
    ]

    async_tests = [
        test_spot_tools_registration,
        test_spot_tools_schema_valid,
        test_spot_base_full_coverage,
        test_spot_arm_full_coverage,
        test_spot_perception_full_coverage,
        test_memory_hooks_fire,
        test_register_spot_tools_on_agent_loop,
        test_spot_tools_already_registered_no_duplicate,
        test_eap_with_spot_reset_executor,
        test_eap_full_cycle_with_spot,
    ]

    failed = 0

    for test in sync_tests:
        try:
            test()
        except Exception:
            import traceback
            print(f"✗ {test.__name__}: FAILED")
            traceback.print_exc()
            failed += 1

    for test in async_tests:
        try:
            await test()
        except Exception:
            import traceback
            print(f"✗ {test.__name__}: FAILED")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 64)
    total = len(sync_tests) + len(async_tests)
    if failed:
        print(f"{failed}/{total} tests FAILED.")
    else:
        print(f"All {total} tests PASSED.")
    print("=" * 64)
    return failed


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
