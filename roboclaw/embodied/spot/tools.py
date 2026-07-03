"""
tools.py
========
SpotToolGroup - Spot tools

Three groups, each a Tool registered in AgentLoop:
 
    spot_base        — mobile base: linear, rotation, autonomous navigation
    spot_arm         — arm: incremental and absolute pose
    spot_perception  — perception: PCL segmentation, object detection, gripper

Integration with AgentLoop (_register_default_tools):

    from roboclaw.embodied.spot.tools import create_spot_tools
    from roboclaw.embodied.spot.service import SpotService

    spot_svc = SpotService()
    for tool in create_spot_tools(spot_svc, self.episode_memory):
        self.tools.register(tool)

Episodic memory hooks:
  - Before each `execute`: call `retrieve()` to check the failure history
  - After each `execute`: call `store()` with the outcome and `env_state` of the result
"""

from __future__ import annotations

import json
from typing import Any

from roboclaw.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# Constants of the actions per group
# ---------------------------------------------------------------------------

_BASE_ACTIONS = [
    "move_forward", "move_backward", "move_left", "move_right",
    "rotate_left", "rotate_right", "navigate_to_pose",  
]

_ARM_ACTIONS = [
    "move_arm_left", "move_arm_right",
    "move_arm_up",   "move_arm_down",
    "move_arm_forward", "move_arm_backward",
    "arm_go_to_pose",
]

_PERCEPTION_ACTIONS = [
    "segment_pcl",
    "detect_objects",
    "get_gripper_state",
]

# ---------------------------------------------------------------------------
# Base Class for Spot Tools
# ---------------------------------------------------------------------------

class _SpotTool(Tool):
    """
    Base class for all SpotToolGroups.
    Holds the SpotService and RoboClawMemory injected by AgentLoop.
    """

    def __init__(
            self,
            spot_service: Any,
            episode_memory: Any = None,
    ) -> None:
        self._svc = spot_service
        self._mem = episode_memory

    # ---------------------------------------------------------------------------
    # Episodic memory hooks
    # ---------------------------------------------------------------------------
    def _retrieve_context(self, action: str, kwargs: dict) -> str:
        """
        Queries episodic memory before execution.
        Returns a context string or null if memory is unavailable.
        """
        if self._mem is None:
            return ""
        query = f"{action} {' '.join(str(v) for v in kwargs.values())}"
        return self._mem.retrieve(query, top_k=3, as_context_string=True)
    
    def _store_result(
            self,
            action: str,
            result_json: str,
            extra_state: dict | None = None,
    ) -> None:
        """
        Saves the episode to memory after execution.
        Determines the outcome based on the 'status' field in the result JSON.
        """
        if self._mem is None:
            return
        try:
            data = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            data = {}
        
        outcome = "success" if data.get("status") == "ok" else "failed"
        env_state = {k: v for k, v in data.items() if k != "status"}
        if extra_state:
            env_state.update(extra_state)
        
        self._mem.store(subtask=action, 
                        outcome=outcome,
                        env_state=env_state)
        
    # ---------------------------------------------------------------------------
    # Group 1: Spot Base
    # ---------------------------------------------------------------------------

class SpotBaseTool(_SpotTool):
    """
    Control spot's mobile base.

    Available actions:
      move_forward / move_backward / move_left / move_right
        distance_m: float = 1.0  (0.01 – 10.0)
 
      rotate_left / rotate_right
        angle_deg: float = 90.0  (0 – 360)
 
      navigate_to_pose
        x: float (required)
        y: float (required)
        yaw_deg: float = 0.0
        frame_id: str = "map"
    """

    @property
    def name(self) -> str:
        return "spot_base"
    
    @property
    def description(self) -> str:
        return (
            "Control the Spot mobile base. "
            "Supports linear movements (forward/backward/left/right), "
            "in-place rotation (left/right), and autonomous navigation to a map pose. "
            "Use navigate_to_pose for longer distances or room-to-room movements."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _BASE_ACTIONS,
                    "description": "The base action to perform.",
                },
                "distance_m": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 10.0,
                    "description": "Distance to travel in metres. Default 1.0.",
                },
                "angle_deg": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 360.0,
                    "description": "Angle to rotate in degrees. Default 90.",
                },
                "x": {
                    "type": "number",
                    "description": "Target X in the map frame (metres). Required for navigate_to_pose.",
                },
                "y": {
                    "type": "number",
                    "description": "Target Y in the map frame (metres). Required for navigate_to_pose.",
                },
                "yaw_deg": {
                    "type": "number",
                    "description": "Desired heading at goal in degrees. Default 0.",
                },
                "frame_id": {
                    "type": "string",
                    "description": "Coordinate frame for navigate_to_pose. Default 'map'.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }
    
    async def execute(self, **kwargs: Any) -> str:
        """
        Executes the specified base action with the given parameters.
        Returns a JSON string with the result.
        """
        action = kwargs.get("action", "")

        if action not in _BASE_ACTIONS:
            return f"Error: unknown action '{action}'. Valid actions are: {', '.join(_BASE_ACTIONS)}"
        
        # Retrieve context from episodic memory
        context = self._retrieve_context(action, kwargs)

        if action == "move_forward":
            result = await self._svc.move_forward(kwargs.get("distance_m", 1.0))
        elif action == "move_backward":
            result = await self._svc.move_backward(kwargs.get("distance_m", 1.0))
        elif action == "move_left":
            result = await self._svc.move_left(kwargs.get("distance_m", 1.0))
        elif action == "move_right":
            result = await self._svc.move_right(kwargs.get("distance_m", 1.0))
        elif action == "rotate_left":
            result = await self._svc.rotate(abs(kwargs.get("angle_deg", 90.0)))
        elif action == "rotate_right":
            result = await self._svc.rotate(-abs(kwargs.get("angle_deg", 90.0)))
        elif action == "navigate_to_pose":
            x = kwargs.get("x")
            y = kwargs.get("y")
            yaw_deg = kwargs.get("yaw_deg", 0.0)
            frame_id = kwargs.get("frame_id", "map")
            if x is None or y is None:
                return "Error: 'x' and 'y' parameters are required for navigate_to_pose."
            result = await self._svc.navigate_to_pose(x, y, yaw_deg, frame_id)
        else:
            return f"Error: action '{action}' is not implemented."

        # Store the result in episodic memory
        self._store_result(action, result)

        try:
            data = json.loads(result)
            return self._format_base_result(action, data)
        except (json.JSONDecodeError, TypeError):
            return result
    
    @staticmethod
    def _format_base_result(action: str, data: dict) -> str:
        """
        Formats the result of a base action into a human-readable string.
        """
        if data.get("status") != "ok":
            return f"Error executing {action}: {data.get('error', 'unknown')}"
        if action in ("move_forward", "move_backward", "move_left", "move_right"):
            return (
                f"Base moved {action.replace('move_', '')} "
                f"{data['distance_m']:.2f} m in {data['duration_s']:.1f}s."
            )
        if action in ("rotate_left", "rotate_right"):
            return (
                f"Base rotated {data['direction']} "
                f"{abs(data['angle_deg']):.1f}° in {data['duration_s']:.1f}s."
            )
        if action == "navigate_to_pose":
            t = data.get("target", {})
            return (
                f"Navigation complete: reached "
                f"(x={t.get('x', 0):.2f}, y={t.get('y', 0):.2f}, "
                f"yaw={t.get('yaw_deg', 0):.1f}°)."
            )
        return json.dumps(data)
    

# ---------------------------------------------------------------------------
# Group 2: Spot Arm
# ---------------------------------------------------------------------------

class SpotArmTool(_SpotTool):
    """
    Spot Arm control

    Available actions:

    Controle do braço do Spot.
 
      move_arm_left / move_arm_right / move_arm_up /
      move_arm_down / move_arm_forward / move_arm_backward
        step_m: float = 0.3  (0.001 - 1.0)

      arm_go_to_pose
        x, y, z: float (obrigatórios, em metros no frame body)
        roll_deg, pitch_deg, yaw_deg: float = 0.0
        frame_id: str = "body"
    """

    @property
    def name(self) -> str:
        return "spot_arm"

    @property
    def description(self) -> str:
        return (
            "Control the Spot arm. "
            "Supports incremental Cartesian moves (left/right/up/down/forward/backward) "
            "and absolute end-effector pose targeting via arm_go_to_pose."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _ARM_ACTIONS,
                    "description": "The arm action to perform.",
                },
                "step_m": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 1.0,
                    "description": "Step size for incremental moves in metres. Default 0.3.",
                },
                "x": {
                    "type": "number",
                    "description": "Target X position of end-effector (metres). Required for arm_go_to_pose.",
                },
                "y": {
                    "type": "number",
                    "description": "Target Y position of end-effector (metres). Required for arm_go_to_pose.",
                },
                "z": {
                    "type": "number",
                    "description": "Target Z position of end-effector (metres). Required for arm_go_to_pose.",
                },
                "roll_deg": {
                    "type": "number",
                    "description": "End-effector roll in degrees. Default 0.",
                },
                "pitch_deg": {
                    "type": "number",
                    "description": "End-effector pitch in degrees. Default 0.",
                },
                "yaw_deg": {
                    "type": "number",
                    "description": "End-effector yaw in degrees. Default 0.",
                },
                "frame_id": {
                    "type": "string",
                    "description": "Coordinate frame for arm_go_to_pose. Default 'body'.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }


    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in _ARM_ACTIONS:
            return f"Error: unknown spot_arm action '{action}'."
 
        self._retrieve_context(action, kwargs)
 
        step = kwargs.get("step_m", 0.3)
 
        _incremental = {
            "move_arm_left":     {"dy": -step},
            "move_arm_right":    {"dy": +step},
            "move_arm_up":       {"dz": +step},
            "move_arm_down":     {"dz": -step},
            "move_arm_forward":  {"dx": +step},
            "move_arm_backward": {"dx": -step},
        }
 
        if action in _incremental:
            result = await self._svc.arm_move_cartesian(**_incremental[action])
        elif action == "arm_go_to_pose":
            if not all(k in kwargs for k in ("x", "y", "z")):
                return "Error: arm_go_to_pose requires x, y, and z."
            result = await self._svc.arm_go_to_pose(
                x=kwargs["x"],
                y=kwargs["y"],
                z=kwargs["z"],
                roll_deg=kwargs.get("roll_deg", 0.0),
                pitch_deg=kwargs.get("pitch_deg", 0.0),
                yaw_deg=kwargs.get("yaw_deg", 0.0),
                frame_id=kwargs.get("frame_id", "body"),
            )
        else:
            return f"Error: unhandled arm action '{action}'."
 
        self._store_result(action, result)
 
        try:
            data = json.loads(result)
            return self._format_arm_result(action, data, step)
        except (json.JSONDecodeError, TypeError):
            return result

    @staticmethod
    def _format_arm_result(action: str, data: dict, step: float) -> str:
        if data.get("status") != "ok":
            return f"Error executing {action}: {data.get('error', 'unknown')}"
        if action != "arm_go_to_pose":
            direction = action.replace("move_arm_", "")
            return f"Arm moved {direction} by {step:.3f} m."
        pose = data.get("pose", {})
        return (
            f"Arm pose set: position=({pose.get('x', 0):.3f}, "
            f"{pose.get('y', 0):.3f}, {pose.get('z', 0):.3f}) m, "
            f"frame='{data.get('frame_id', 'body')}'."
        )

# ---------------------------------------------------------------------------
# Group 3: Spot Perception
# ---------------------------------------------------------------------------

# TODO: Fix for the right spot perception
class SpotPerceptionTool(_SpotTool):
    """
    Perception services for Spot (ROS2 calls).

    Available actions: 
      segment_pcl
        max_objects: int = 5
        min_confidence: float = 0.5
        → Segments the point cloud of the arm and returns objects with centroid and bbox.
 
      detect_objects
        camera: str = "hand"  ("hand" | "frontleft" | "frontright" | "back")
        → Executes 2D detection on the specified camera.
 
      get_gripper_state
        → Returns the position, force, and grasp status of the gripper.
 
    Precondition declarations (used by the VLM for causal planning):
      segment_pcl      → the arm must be in an observation position
      detect_objects   → the target camera must be operational
      get_gripper_state → the arm must be active
 
    Effect declarations (used by the VLM for causal planning):
      segment_pcl      → produces a list of objects with poses in the arm's frame
      detect_objects   → produces 2D detections with bounding boxes in pixels
      get_gripper_state → reveals whether the gripper is holding an object
    """

    @property
    def name(self) -> str:
        return "spot_perception"
 
    @property
    def description(self) -> str:
        return (
            "Spot perception services (ROS2 service calls). "
            "segment_pcl: segments the arm point cloud and returns detected objects "
            "with 3D centroids and bounding boxes — requires arm in observation pose. "
            "detect_objects: runs 2D object detection on a specified camera. "
            "get_gripper_state: returns current gripper position, force, and grasp status. "
            "Call segment_pcl or detect_objects BEFORE planning a grasp to know object poses."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _PERCEPTION_ACTIONS,
                    "description": "The perception action to perform.",
                },
                "max_objects": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum objects to return from segment_pcl. Default 5.",
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence threshold for segment_pcl. Default 0.5.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["hand", "frontleft", "frontright", "back"],
                    "description": "Camera for detect_objects. Default 'hand'.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in _PERCEPTION_ACTIONS:
            return f"Error: unknown spot_perception action '{action}'."
 
        self._retrieve_context(action, kwargs)
 
        if action == "segment_pcl":
            result = await self._svc.segment_pcl(
                max_objects=kwargs.get("max_objects", 5),
                min_confidence=kwargs.get("min_confidence", 0.5),
            )
        elif action == "detect_objects":
            result = await self._svc.detect_objects(
                camera=kwargs.get("camera", "hand"),
            )
        elif action == "get_gripper_state":
            result = await self._svc.get_gripper_state()
        else:
            return f"Error: unhandled perception action '{action}'."
 
        self._store_result(action, result)
 
        try:
            data = json.loads(result)
            return self._format_perception_result(action, data)
        except (json.JSONDecodeError, TypeError):
            return result
 
    @staticmethod
    def _format_perception_result(action: str, data: dict) -> str:
        if data.get("status") != "ok":
            return f"Error executing {action}: {data.get('error', 'unknown')}"
 
        if action == "segment_pcl":
            n = data.get("objects_found", 0)
            if n == 0:
                return "PCL segmentation complete: no objects detected."
            objs = data.get("objects", [])
            lines = [f"PCL segmentation: {n} object(s) found."]
            for o in objs:
                c = o["centroid"]
                lines.append(
                    f"  [{o['label']}] confidence={o['confidence']:.2f} "
                    f"centroid=({c['x']:.3f}, {c['y']:.3f}, {c['z']:.3f}) m"
                )
            return "\n".join(lines)

        if action == "detect_objects":
            n = data.get("detections_found", 0)
            if n == 0:
                return f"Object detection on '{data.get('camera')}': no objects found."
            dets = data.get("detections", [])
            lines = [f"Detection on '{data.get('camera')}': {n} object(s)."]
            for d in dets:
                lines.append(
                    f"  [{d['label']}] confidence={d['confidence']:.2f}"
                )
            return "\n".join(lines)

        if action == "get_gripper_state":
            g = data.get("gripper", {})
            holding = "holding object" if g.get("is_holding") else "empty"
            return (
                f"Gripper: position={g.get('position', 0):.2f} "
                f"(0=closed, 1=open), force={g.get('force_n', 0):.1f}N, "
                f"status={holding}."
            )

        return json.dumps(data)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
 
def create_spot_tools(
    spot_service: Any,
    episode_memory: Any = None,
) -> list[_SpotTool]:
    """
    Instantiates the three Spot tool groups.
 
    Args:
        spot_service:   instantiates the SpotService (or mock for testing)
        episode_memory: instantiates the RoboClawMemory (optional)
 
    Returns:
        List ready to register in the ToolRegistry of the AgentLoop.
 
    Usage in AgentLoop._register_default_tools():
        from roboclaw.embodied.spot.tools import create_spot_tools
        from roboclaw.embodied.spot.service import SpotService
 
        spot_svc = SpotService()
        for tool in create_spot_tools(spot_svc, self.episode_memory):
            self.tools.register(tool)
    """
    kwargs = {"spot_service": spot_service, "episode_memory": episode_memory}
    return [
        SpotBaseTool(**kwargs),
        SpotArmTool(**kwargs),
        SpotPerceptionTool(**kwargs),
    ]
