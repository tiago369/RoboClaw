"""
Loads the Spot manifest and derives SpotConfig + SpotService configured.

Usage:
    from roboclaw.embodied.embodiment.manifest.spot_loader import load_spot_config
 
    cfg, active_cams = load_spot_config()   # use ~/.roboclaw/spot_manifest.json
    robot = SpotRobot(cfg)
    robot.connect()
 
    # Or loading a customized path:
    cfg, active_cams = load_spot_config("/path/to/spot_manifest.json")
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from .helpers import get_roboclaw_home

_DEFAULT_MANIFEST = pathlib.Path(__file__).parent / "spot_manifest.json"


def load_spot_manifest(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """
    Loads and validates the Spot manifest.

    Args:
        path: JSON path. None = ~/.roboclaw/spot_manifest.json (copy the default if doesn't exist)

    Returns:
        dict with full spot manifest.
    """
    if path is None:
        home_manifest = get_roboclaw_home() / "spot_manifest.json"
        if not home_manifest.exists():
            home_manifest.parent.mkdir(parents=True, exist_ok=True)
            home_manifest.write_text(_DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        path = home_manifest

    path = pathlib.Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Manifest of Spot not found: {path}")

    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("robot_type") != "spot":
        raise ValueError(
            f"Manifest in {path} is not from Spot (robot_type={manifest.get('robot_type')!r})"
        )

    return manifest


def load_spot_config(
    manifest_path: str | pathlib.Path | None = None,
):
    """
    Loads with the manifest and returns (SpotConfig, list[str]) where the list
    contains the aliases from the cameras with active=true.

    Args:
        manifest_path: JSON path. None = ~/.roboclaw/spot_manifest.json
    
    Returns:
        (SpotConfig, active_camera_aliases)
    
    Example:
        cfg, cams = load_spot_config()
        # cfg.active_cameras == ["hand"]  (se só hand estiver active=true)
        robot = SpotRobot(cfg)
    """
    from lerobot.robots.spot.config_spot import SpotConfig

    manifest = load_spot_manifest(manifest_path)
    spot_cfg = manifest.get("spot", {})

    all_cams = manifest.get("cameras", [])
    active_aliases = [
        c["alias"] for c in all_cams
        if c.get("active", False) and c.get("type") == "ros2"
    ]

    if not active_aliases:
        active_aliases = ["hand"]

    cfg = SpotConfig(
        ros2_namespace=spot_cfg.get("ros2_namespace", "/spot"),
        cmd_vel_topic=spot_cfg.get("cmd_vel_topic", "/cmd_vel"),
        arm_pose_topic=spot_cfg.get("arm_pose_topic", "/arm_pose_commands"),
        joint_state_topic=spot_cfg.get("joint_state_topic", "/joint_states"),
        connection_timeout_s=spot_cfg.get("connection_timeout_s", 10.0),
        use_degrees=spot_cfg.get("use_degrees", False),
        active_cameras=active_aliases,
    )

    return cfg, active_aliases


def load_eap_reset_sequence(manifest_path: str | pathlib.Path | None = None) -> list[dict]:
    """
    Extracts the sequence of reset and EAP from the manifest.

    Returns:
        Steps list for the SpotResetExecutor, or default sequence if not defined.
    """
    try:
        manifest = load_spot_manifest(manifest_path)
        eap_cfg = manifest.get("spot", {}).get("eap", {})
        if eap_cfg.get("enabled", True):
            return eap_cfg.get("reset_sequence", _default_reset_sequence())
    except Exception:
        pass
    return _default_reset_sequence()


def _default_reset_sequence() -> list[dict]:
    return [
         {"action": "arm_go_to_pose", "kwargs": {"x": 0.5, "y": 0.0, "z": 0.3, "pitch_deg": 0.0}},
        {"action": "move_backward",  "kwargs": {"distance_m": 0.3}},
    ]
