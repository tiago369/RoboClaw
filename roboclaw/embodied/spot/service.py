"""
service.py
==========
SpotService - ROS2 wrapper for the Boston Dynamics Spot with the arm

Similar to EmbodiedService for generic hardware, but specialized
for Spot: it encapsulates ROS2 service clients, Twist publishers, and
PoseStamped, and exposes high-level methods that SpotToolGroups call.
 
Lazy connection: The ROS2 node is only initialized when the first method is
called—with no import overhead in environments without ROS2.
 
Use:
    svc = SpotService()
    await svc.move_forward(1.0)
    result = await svc.segment_pcl(max_objects=5)
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

class SpotServiceError(Exception):
    """Error executing a Spot ROS2 service."""

class SpotService:
    """
    A single point of control for all Spot ROS2 operations.

    Management:
      - Mobile base: published to /cmd_vel via geometry_msgs/Twist
      - Arm: publish to /arm_pose_commands via geometry_msgs/PoseStamped
      - Navigation: Nav2 action client (NavigateToPose)
      - Perception: ROS2 service clients (PCL segmentation, detection, etc.)
 
    All methods are asynchronous for compatibility with AgentLoop.
    Actual ROS2 communication takes place via rclpy on a dedicated thread.
    """

    def __init__(self, node_name: str = "roboclaw_spot_node") -> None:
        self._node_name = node_name
        self._node: Any = None
        self._ros_thread: threading.Thread | None = None
        self._executor: Any = None
        self._lock = threading.Lock()
        self._ready = False

        self._cmd_vel_pub: Any = None
        self._arm_pose_pub: Any = None

        self._pcl_segment_client: Any = None
        self._object_detect_client: Any = None
        self._gripper_state_client: Any = None

        self._arm_pose = {"x": 0.5, "y": 0.0, "z": 0.3,
                          "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        
    def _init_ros(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from geometry_msgs.msg import Twist, PoseStamped

        if not rclpy.ok():
            rclpy.init()
        
        self._node = rclpy.create_node(self._node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        from geometry_msgs.msg import Twist, PoseStamped
        self._cmd_vel_pub = self._node.create_publisher(
            Twist, "/cmd_vel", 10
        )

        self._arm_pose_pub = self._node.create_publisher(
            PoseStamped, "/arm_pose_commands", 10
        )

        # TODO: fix this
        try:
            from spot_interfaces.srv import SegmentPCL, DetectObjects, GetGripperState
            self._pcl_segment_client = self._node.create_client(
                SegmentPCL, "/spot/perception/segment_pcl"
            )
            self._object_detect_client = self._node.create_client(
                DetectObjects, "/spot/perception/detect_objects"
            )
            self._gripper_state_client = self._node.create_client(
                GetGripperState, "/spot/arm/gripper_state"
            )
        except ImportError:
            # spot_interfaces não instalado — percepção indisponível
            self._node.get_logger().warning(
                "spot_interfaces not found — perception services unavailable"
            )
        
        self._ros_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._ros_thread.start()
        self._ready = True

    def _ensure_ready(self) -> None:
        with self._lock:
            if not self._ready:
                self._init_ros()
    
    # --------------------------------------------------------------
    # Mobile Base
    # --------------------------------------------------------------

    async def move_forward(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vx=+abs(distance_m))

    async def move_backward(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vx=-abs(distance_m))

    async def move_left(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vy=+abs(distance_m))

    async def move_right(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vy=-abs(distance_m))
    
    async def _move_linear(
            self,
            vx: float = 0.0,
            vy: float = 0.0,
            duration_s: float | None = None
    ) -> str:
        """
        Publish Twist to /cmd_vel.

        If duration_s is not specified, it is estimated based on the magnitude of the velocity,
        assuming a nominal velocity of 0.5 m/s.
        """
        self._ensure_ready()

        from geometry_msgs.msg import Twist

        speed = 0.5
        dist = abs(vx) or abs(vy)
        dur = duration_s or (dist / speed)

        twist = Twist()
        twist.linear.x = float(vx / dist * speed) if dist else 0.0
        twist.linear.y = float(vy / dist * speed) if dist else 0.0
 
        await self._publish_for(self._cmd_vel_pub, twist, dur)

        direction = (
            "forward" if vx > 0 else
            "backward" if vx < 0 else
            "left" if vy > 0 else "rigth"
        )

        return json.dumps({
            "status": "ok",
            "action": f"move_{direction}",
            "distance_m": dist,
            "duration_s": round(dur, 2)
        })

    async def rotate(self, angle_deg: float) -> str:
        """
        Rotate the base in place
        angle_deg > 0 → CCW (left), < 0 → CW (right).
        """

        self._ensure_ready()

        import math
        from geometry_msgs.msg import Twist

        omega = 0.5
        angle_rad = math.radians(angle_deg)
        dur = abs(angle_rad) / omega

        twist = Twist()
        twist.angular.z = math.copysign(omega, angle_rad)

        await self._publish_for(self._cmd_vel_pub, twist, dur)

        direction = "left (CCW)" if angle_deg > 0 else "right (CW)"
        return json.dumps({
            "status": "ok",
            "action": "rotate",
            "angle_deg": angle_deg,
            "direction": direction,
            "duration_s": round(dur, 2),
        })

    # --------------------------------------------------------------
    # Mobile base - NAV2
    # --------------------------------------------------------------

    async def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw_deg: float = 0.0,
        frame_id: str = "map",
        timeout_s: float = 60.0,
    ) -> str:
        """
        Sends a goal to the Nav2 NavigateToPose action server.
        Blocks until arrival or timeout.
        """

        self._ensure_ready()

        try:
            from nav2_msgs.action import NavigateToPose as Nav2Goal
            from rclpy.action import ActionClient
            import math
        except ImportError:
            return json.dumps({
                "status": "error",
                "error": "nav2_msgs not available — install navigation2",
            })
 
        client = ActionClient(
            self._node, Nav2Goal, "/navigate_to_pose"
        )
        if not client.wait_for_server(timeout_sec=5.0):
            return json.dumps({
                "status": "error",
                "error": "Nav2 action server not available",
            })
 
        goal = Nav2Goal.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
 
        half = math.radians(yaw_deg) / 2.0
        goal.pose.pose.orientation.z = math.sin(half)
        goal.pose.pose.orientation.w = math.cos(half)
 
        future = client.send_goal_async(goal)
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            return json.dumps({
                "status": "error",
                "error": f"Navigation timed out after {timeout_s}s",
            })
 
        return json.dumps({
            "status": "ok",
            "action": "navigate_to_pose",
            "target": {"x": x, "y": y, "yaw_deg": yaw_deg, "frame": frame_id},
        })
    
    # --------------------------------------------------------------
    # Arm
    # --------------------------------------------------------------

    async def arm_move_cartesian(
            self,
            dx: float = 0.0,
            dy: float = 0.0,
            dz: float = 0.0,
    ) -> str:
        """Apply the delta to the end-effector's current pose and publish"""
        self._ensure_ready()

        self._arm_pose["x"] += dx
        self._arm_pose["y"] += dy
        self._arm_pose["z"] += dz

        await self._publish_arm_pose()

        delta_str = ", ".join(
            f"d{k}={v:+.3f}" for k, v in [("x", dx), ("y", dy), ("z", dz)] if v != 0
        )

        return json.dumps({
            "status": "ok",
            "action": "arm_move_cartesian",
            "delta": {"dx": dx, "dy": dy, "dz": dz},
            "current_pose": dict(self._arm_pose),
        })

    async def arm_go_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll_deg:  float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg:   float = 0.0,
        frame_id:  str   = "body",
    ) -> str:
        """Move the end-effector to an absolute pose in the specified frame."""
        self._ensure_ready()
 
        self._arm_pose.update({
            "x": x, "y": y, "z": z,
            "roll": roll_deg, "pitch": pitch_deg, "yaw": yaw_deg,
        })
        await self._publish_arm_pose(frame_id=frame_id)
 
        return json.dumps({
            "status": "ok",
            "action": "arm_go_to_pose",
            "pose": dict(self._arm_pose),
            "frame_id": frame_id,
        })
 
    async def _publish_arm_pose(self, frame_id: str = "body") -> None:
        """Publish the current PoseStamped to /arm_pose_commands."""
        import math
        from geometry_msgs.msg import PoseStamped
 
        p = self._arm_pose
        msg = PoseStamped()
        msg.header.frame_id = frame_id
 
        msg.pose.position.x = float(p["x"])
        msg.pose.position.y = float(p["y"])
        msg.pose.position.z = float(p["z"])
 
        r = math.radians(p["roll"])
        pitch = math.radians(p["pitch"])
        y_ = math.radians(p["yaw"])
 
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(y_ / 2), math.sin(y_ / 2)
 
        msg.pose.orientation.w = cr * cp * cy + sr * sp * sy
        msg.pose.orientation.x = sr * cp * cy - cr * sp * sy
        msg.pose.orientation.y = cr * sp * cy + sr * cp * sy
        msg.pose.orientation.z = cr * cp * sy - sr * sp * cy
 
        self._arm_pose_pub.publish(msg)

    # --------------------------------------------------------------
    # TODO: change this to my packages
    # --------------------------------------------------------------

    async def segment_pcl(
        self,
        max_objects: int = 5,
        min_confidence: float = 0.5,
        timeout_s: float = 5.0,
    ) -> str:
        """
        Call /spot/perception/segment_pcl.
        Returns JSON containing a list of segmented objects and their bounding boxes.
        """
        self._ensure_ready()
 
        if self._pcl_segment_client is None:
            return json.dumps({
                "status": "error",
                "error": "PCL segmentation service not available (spot_interfaces missing)",
            })
 
        if not self._pcl_segment_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({
                "status": "error",
                "error": f"/spot/perception/segment_pcl not responding after {timeout_s}s",
            })
 
        from spot_interfaces.srv import SegmentPCL
        request = SegmentPCL.Request()
        request.max_objects = max_objects
        request.min_confidence = min_confidence
 
        future = self._pcl_segment_client.call_async(request)
        try:
            response = await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            return json.dumps({
                "status": "error",
                "error": "segment_pcl timed out",
            })
 
        objects = [
            {
                "id": o.id,
                "label": o.label,
                "confidence": round(o.confidence, 3),
                "centroid": {
                    "x": round(o.centroid.x, 3),
                    "y": round(o.centroid.y, 3),
                    "z": round(o.centroid.z, 3),
                },
                "bbox_size": {
                    "x": round(o.bbox.x, 3),
                    "y": round(o.bbox.y, 3),
                    "z": round(o.bbox.z, 3),
                },
            }
            for o in response.objects
        ]
 
        return json.dumps({
            "status": "ok",
            "action": "segment_pcl",
            "objects_found": len(objects),
            "objects": objects,
        })
 
    async def detect_objects(
        self,
        camera: str = "hand",
        timeout_s: float = 5.0,
    ) -> str:
        """
        Call /spot/perception/detect_objects.
        Returns JSON with 2D detections from the specified camera.
        camera: "hand" | "frontleft" | "frontright" | "back"
        """
        self._ensure_ready()
 
        if self._object_detect_client is None:
            return json.dumps({
                "status": "error",
                "error": "Object detection service not available",
            })
 
        if not self._object_detect_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({
                "status": "error",
                "error": "/spot/perception/detect_objects not responding",
            })
 
        from spot_interfaces.srv import DetectObjects
        request = DetectObjects.Request()
        request.camera = camera
 
        future = self._object_detect_client.call_async(request)
        try:
            response = await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            return json.dumps({
                "status": "error",
                "error": "detect_objects timed out",
            })
 
        detections = [
            {
                "label": d.label,
                "confidence": round(d.confidence, 3),
                "bbox_px": {
                    "x": d.bbox.x, "y": d.bbox.y,
                    "w": d.bbox.width, "h": d.bbox.height,
                },
            }
            for d in response.detections
        ]
 
        return json.dumps({
            "status": "ok",
            "action": "detect_objects",
            "camera": camera,
            "detections_found": len(detections),
            "detections": detections,
        })
 
    async def get_gripper_state(self, timeout_s: float = 3.0) -> str:
        """
        Call /spot/arm/gripper_state.
        Returns the current state of the gripper: position, force, and whether it is gripping.
        """
        self._ensure_ready()
 
        if self._gripper_state_client is None:
            return json.dumps({
                "status": "error",
                "error": "Gripper state service not available",
            })
 
        if not self._gripper_state_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({
                "status": "error",
                "error": "/spot/arm/gripper_state not responding",
            })
 
        from spot_interfaces.srv import GetGripperState
        future = self._gripper_state_client.call_async(
            GetGripperState.Request()
        )
        try:
            response = await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            return json.dumps({
                "status": "error",
                "error": "get_gripper_state timed out",
            })
 
        return json.dumps({
            "status": "ok",
            "action": "get_gripper_state",
            "gripper": {
                "position": round(response.position, 3),
                "force_n": round(response.force_n, 2),
                "is_holding": response.is_holding,
                "object_detected": response.object_detected,
            },
        })
 
    # ------------------------------------------------------------------
    # Internal Utilities
    # ------------------------------------------------------------------
 
    async def _publish_for(
        self, publisher: Any, msg: Any, duration_s: float
    ) -> None:
        """Publishes a message at 10 Hz for duration_s, then stops."""
        hz = 10
        interval = 1.0 / hz
        steps = max(1, int(duration_s * hz))
 
        for _ in range(steps):
            publisher.publish(msg)
            await asyncio.sleep(interval)
 
        from geometry_msgs.msg import Twist
        if isinstance(msg, Twist):
            publisher.publish(Twist())
 
    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()
 
