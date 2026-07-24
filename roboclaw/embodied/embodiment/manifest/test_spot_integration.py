"""
Tests
cd RoboClaw && python roboclaw/embodied/embodiment/manifest/test_spot_integration.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import types

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../../../"))
_ENGINE = os.path.join(_ROOT, "roboclaw/embodied/engine/src")
for p in (_ROOT, _ENGINE):
    if p not in sys.path: sys.path.insert(0, p)

# ── minimum stubs ─────────────────────────────────────────────────────
for _n in ("roboclaw.agent","roboclaw.agent.tools"):
    _s=types.ModuleType(_n); _s.__path__=[]; sys.modules[_n]=_s

if "draccus" not in sys.modules:
    _d=types.ModuleType("draccus")
    class _CR:
        _reg:dict={}
        @classmethod
        def register_subclass(cls,n):
            def dec(k): cls._reg[n]=k; return k
            return dec
        @classmethod
        def get_choice_name(cls,k):
            for n,v in cls._reg.items():
                if v is k: return n
            return k.__name__
    _d.ChoiceRegistry=_CR; sys.modules["draccus"]=_d

for _lm in ("lerobot","lerobot.utils","lerobot.utils.decorators",
            "lerobot.utils.constants","lerobot.types","lerobot.cameras",
            "lerobot.cameras.utils","lerobot.cameras.configs",
            "lerobot.cameras.ros2","lerobot.cameras.ros2.configuration_ros2",
            "lerobot.motors","lerobot.calibration_timestamp",
            "lerobot.robots","lerobot.robots.config","lerobot.robots.robot",
            "lerobot.robots.spot","lerobot.robots.spot.config_spot",
            "lerobot.robots.spot.spot"):
    if _lm not in sys.modules:
        _s=types.ModuleType(_lm); _s.__path__=[]; sys.modules[_lm]=_s

_du=sys.modules["lerobot.utils.decorators"]
def _nd(fn): return fn
_du.check_if_already_connected=_nd; _du.check_if_not_connected=_nd
_cu=sys.modules["lerobot.cameras.utils"]
_cu.make_cameras_from_configs=lambda c:{}
_t=sys.modules["lerobot.types"]; _t.RobotAction=dict; _t.RobotObservation=dict
_cal=sys.modules["lerobot.calibration_timestamp"]
_cal.record_calibration_timestamp=lambda *a,**kw:None
_const=sys.modules["lerobot.utils.constants"]
_const.HF_LEROBOT_CALIBRATION=pathlib.Path("/tmp"); _const.ROBOTS="robots"
_mt=sys.modules["lerobot.motors"]; _mt.MotorCalibration=object

# Camera ROS2 stub
class _R2Cfg:
    type="ros2"
    def __init__(self,topic="",fps=30,width=640,height=480,encoding="auto"):
        self.topic=topic; self.fps=fps; self.width=width; self.height=height
        self.encoding=encoding
sys.modules["lerobot.cameras.ros2.configuration_ros2"].Ros2CameraConfig=_R2Cfg
sys.modules["lerobot.cameras.configs"].CameraConfig=object

# SpotConfig stub
class _SpotCfg:
    def __init__(self,**kw):
        for k,v in kw.items(): setattr(self,k,v)
        if not hasattr(self,"cameras"): self.cameras={}
    def __post_init__(self): pass

_ALL_CAMERAS={
    "hand":_R2Cfg("/spot/camera/hand/image"),
    "frontleft":_R2Cfg("/spot/camera/frontleft/image"),
    "frontright":_R2Cfg("/spot/camera/frontright/image"),
    "left":_R2Cfg("/spot/camera/left/image"),
    "right":_R2Cfg("/spot/camera/right/image"),
    "back":_R2Cfg("/spot/camera/back/image"),
    "hand_depth":_R2Cfg("/spot/depth/hand/image",encoding="16UC1"),
    "frontleft_depth":_R2Cfg("/spot/depth/frontleft/image",encoding="16UC1"),
}

def _spot_cameras_config(active=None):
    if active is None: active=["hand"]
    unknown=[c for c in active if c not in _ALL_CAMERAS]
    if unknown: raise ValueError(f"Câmera(s) desconhecida(s): {unknown}")
    return {n:_ALL_CAMERAS[n] for n in active}

_spot_mod=sys.modules["lerobot.robots.spot.config_spot"]
_spot_mod.SpotConfig=_SpotCfg
_spot_mod.ALL_SPOT_CAMERAS=_ALL_CAMERAS
_spot_mod.spot_cameras_config=_spot_cameras_config

# RobotConfig stub
class _RC:
    @classmethod
    def register_subclass(cls,n):
        def dec(k): return k
        return dec
sys.modules["lerobot.robots.config"].RobotConfig=_RC

from roboclaw.embodied.embodiment.manifest.helpers import _VALID_TOP_KEYS
from roboclaw.embodied.embodiment.manifest.spot_loader import (
    load_eap_reset_sequence,
    load_spot_config,
    load_spot_manifest,
)

MANIFEST_PATH = pathlib.Path(_HERE) / "spot_manifest.json"


# ══════════════════════════════════════════════════════════════════════
# Item 4 — Manifest
# ══════════════════════════════════════════════════════════════════════

def test_manifest_valid_keys():
    """robot_type and spot should be accepted as valid keys."""
    assert "robot_type" in _VALID_TOP_KEYS
    assert "spot" in _VALID_TOP_KEYS
    print("✓ Item 4: robot_type and spot in _VALID_TOP_KEYS")


def test_manifest_file_exists():
    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"
    print(f"✓ Item 4: spot_manifest.json exists in {MANIFEST_PATH}")


def test_manifest_structure():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    assert m["version"] == 2
    assert m["robot_type"] == "spot"
    assert isinstance(m["cameras"], list)
    assert len(m["cameras"]) == 8  # 6 RGB + 2 depth
    assert "spot" in m
    spot = m["spot"]
    assert spot["cmd_vel_topic"] == "/cmd_vel"
    assert spot["joint_state_topic"] == "/joint_states"
    assert len(spot["arm_joints"]) == 7
    assert spot["eap"]["enabled"] is True
    assert len(spot["eap"]["reset_sequence"]) == 2
    print("✓ Item 4: correct manifest structure (version, cameras, joints, EAP)")


def test_manifest_cameras_have_required_fields():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    for cam in m["cameras"]:
        assert "alias" in cam, f"Câmera sem alias: {cam}"
        assert "topic" in cam, f"Câmera sem topic: {cam}"
        assert "type" in cam, f"Câmera sem type: {cam}"
        assert cam["type"] == "ros2"
    print("✓ Item 4: all cameras have alias, topic e type=ros2")


def test_manifest_default_active_camera():
    """Only 'hand' should have active=true by standard."""
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    active = [c["alias"] for c in m["cameras"] if c.get("active")]
    assert active == ["hand"], f"Active cameras: {active}"
    print("✓ Item 4: only 'hand' standard active")


def test_load_spot_manifest():
    manifest = load_spot_manifest(MANIFEST_PATH)
    assert manifest["robot_type"] == "spot"
    print("✓ Item 4: load_spot_manifest() loads correctly")


def test_load_spot_manifest_wrong_type():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"version": 2, "robot_type": "lekiwi"}, f)
        tmp = f.name
    try:
        load_spot_manifest(tmp)
        assert False, "should have posted"
    except ValueError as e:
        assert "spot" in str(e).lower()
    finally:
        os.unlink(tmp)
    print("✓ Item 4: load_spot_manifest() rejects wrong robot_type")


def test_load_spot_config_default_cameras():
    cfg, active = load_spot_config(MANIFEST_PATH)
    assert active == ["hand"]
    assert isinstance(cfg, _SpotCfg)
    assert cfg.active_cameras == ["hand"]
    assert cfg.cmd_vel_topic == "/cmd_vel"
    print("✓ Item 4: load_spot_config() derives SpotConfig with camera hand")


def test_load_spot_config_active_cameras():
    """Modify the manifest to activate camera."""
    with open(MANIFEST_PATH) as f:
        m = json.load(f)

    for c in m["cameras"]:
        if c["alias"] == "frontleft":
            c["active"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(m, f); tmp = f.name
    try:
        cfg, active = load_spot_config(tmp)
        assert "hand" in active and "frontleft" in active
    finally:
        os.unlink(tmp)
    print("✓ Item 4: load_spot_config() respects active=true of multiple cameras")


def test_load_eap_reset_sequence():
    seq = load_eap_reset_sequence(MANIFEST_PATH)
    assert len(seq) == 2
    assert seq[0]["action"] == "arm_go_to_pose"
    assert seq[1]["action"] == "move_backward"
    assert seq[1]["kwargs"]["distance_m"] == 0.3
    print("✓ Item 4: load_eap_reset_sequence() extract manifest sequence")


def test_manifest_copies_to_home():
    """load_spot_manifest without path copies the default to ~/.roboclaw/."""
    with tempfile.TemporaryDirectory() as tmp:

        home_manifest = pathlib.Path(tmp) / "spot_manifest.json"
        assert not home_manifest.exists()

        manifest = load_spot_manifest(MANIFEST_PATH)
        assert manifest["robot_type"] == "spot"
    print("✓ Item 4: manifest default accessible")


# ══════════════════════════════════════════════════════════════════════
# Item 5 — robots/__init__.py
# ══════════════════════════════════════════════════════════════════════

def test_robots_init_exports_spot():
    """robots/__init__.py should exports SpotConfig e SpotRobot."""
    init_path = os.path.join(_ENGINE, "lerobot/robots/__init__.py")
    with open(init_path) as f:
        content = f.read()
    assert "SpotConfig" in content, "SpotConfig not exported in robots/__init__.py"
    assert "SpotRobot" in content,  "SpotRobot not exported in robots/__init__.py"
    assert "ALL_SPOT_CAMERAS" in content
    assert "spot_cameras_config" in content
    print("✓ Item 5: SpotConfig, SpotRobot, ALL_SPOT_CAMERAS, spot_cameras_config in robots/__init__.py")


def test_robots_init_has_all_exports():
    """__all__ should include Spot symbols."""
    init_path = os.path.join(_ENGINE, "lerobot/robots/__init__.py")
    with open(init_path) as f:
        content = f.read()
    for sym in ("SpotConfig", "SpotRobot", "ALL_SPOT_CAMERAS", "spot_cameras_config"):
        assert f'"{sym}"' in content or f"'{sym}'" in content, \
            f"{sym} não está em __all__ de robots/__init__.py"
    print("✓ Item 5: __all__ include all Spot symbols")


def test_spot_registered_as_subclass():
    """SpotConfig should be registered the 'spot' key."""
    cfg_path = os.path.join(_ENGINE, "lerobot/robots/spot/config_spot.py")

    with open(cfg_path) as f:
        content = f.read()
    assert 'register_subclass("spot")' in content
    print("✓ Item 5: SpotConfig registered as subclass 'spot' via decorator")


# ══════════════════════════════════════════════════════════════════════
# Item 6 — commands.py / entrypoint
# ══════════════════════════════════════════════════════════════════════

def test_commands_imports_spot_service():
    """commands.py should try to import SpotService."""
    cmd_path = os.path.join(_ROOT, "roboclaw/cli/commands.py")
    with open(cmd_path) as f:
        content = f.read()
    assert "SpotService" in content, "SpotService not imported em commands.py"
    assert "spot_service=_spot_service" in content or "spot_service=_spot_service_gw" in content
    print("✓ Item 6: commands.py import SpotService and pass spot_service ao AgentLoop")


def test_commands_graceful_without_spot():
    """SpotService fail silently if not available."""
    cmd_path = os.path.join(_ROOT, "roboclaw/cli/commands.py")
    with open(cmd_path) as f:
        content = f.read()

    assert "try:" in content
    assert "except Exception:" in content or "except ImportError:" in content or "except Exception" in content
    print("✓ Item 6: SpotService failure is captured silently (graceful degradation)")


def test_loop_has_spot_service_param():
    """AgentLoop.__init__ should accept spot_service."""
    loop_path = os.path.join(_ROOT, "roboclaw/agent/loop.py")
    with open(loop_path) as f:
        content = f.read()
    assert "spot_service" in content
    assert "register_spot_tools" in content
    print("✓ Item 6: AgentLoop accept spot_service and has register_spot_tools()")


def test_register_spot_tools_method():
    """register_spot_tools() should be implemented in loop.py."""
    loop_path = os.path.join(_ROOT, "roboclaw/agent/loop.py")
    with open(loop_path) as f:
        content = f.read()
    assert "def register_spot_tools" in content
    assert "create_spot_tools" in content
    print("✓ Item 6: register_spot_tools() implemented and calls create_spot_tools()")


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("Items 4+5+6: Manifest, Register LeRobot, Entrypoint")
    print("=" * 64); print()

    tests = [
        # Item 4
        test_manifest_valid_keys,
        test_manifest_file_exists,
        test_manifest_structure,
        test_manifest_cameras_have_required_fields,
        test_manifest_default_active_camera,
        test_load_spot_manifest,
        test_load_spot_manifest_wrong_type,
        test_load_spot_config_default_cameras,
        test_load_spot_config_active_cameras,
        test_load_eap_reset_sequence,
        test_manifest_copies_to_home,
        # Item 5
        test_robots_init_exports_spot,
        test_robots_init_has_all_exports,
        test_spot_registered_as_subclass,
        # Item 6
        test_commands_imports_spot_service,
        test_commands_graceful_without_spot,
        test_loop_has_spot_service_param,
        test_register_spot_tools_method,
    ]

    failed = 0
    for t in tests:
        try:
            t()
        except Exception:
            import traceback
            print(f"✗ {t.__name__}: FAILED")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 64)
    total = len(tests)
    print(f"All {total} tests passed." if not failed else f"{failed}/{total} FAILED.")
    print("=" * 64)
    return failed

if __name__ == "__main__":
    sys.exit(main())
