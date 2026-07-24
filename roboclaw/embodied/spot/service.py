"""
service.py
==========
SpotService — wrapper ROS2 para o Spot com braço, integrado ao grasp_pipeline.

Expõe os 4 serviços reais do grasp_pipeline_interfaces:
  tree_d_segment_service   → ThreeDSegmentator (Gemini + depth → PointCloud2)
  contact_graspnet_service → Graspnet (PointCloud2 → PoseArray de grasps)
  cu_robo_service          → TargetPoses (PoseArray → PoseStamped planejada)
  grasping_service         → Grasping (PoseStamped + GripperCommand → executa)

Também expõe controle direto de base e câmera para o AgentLoop.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any


class SpotServiceError(Exception):
    pass


class SpotService:
    """
    Ponto único de controle para todas as operações ROS2 do Spot.

    Todos os métodos são async — compatíveis com o AgentLoop assíncrono.
    Inicialização ROS2 é lazy: o nó só é criado na primeira chamada.
    """

    def __init__(self, node_name: str = "roboclaw_spot_node") -> None:
        self._node_name = node_name
        self._node: Any = None
        self._executor: Any = None
        self._ros_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ready = False

        # Camera cache (preenchido por subscribers)
        self._latest_rgb: Any = None
        self._latest_depth: Any = None
        self._latest_k_matrix: Any = None
        self._cam_lock = threading.Lock()

        # Publishers
        self._cmd_vel_pub: Any = None
        self._arm_pose_pub: Any = None

        # Arm status tracking (feedback de /arm_command_status)
        self._arm_done_event = threading.Event()
        self._arm_success = False

        # Service clients — grasp_pipeline_interfaces
        self._seg_client: Any = None       # ThreeDSegmentator
        self._grasp_client: Any = None     # Graspnet
        self._curobo_client: Any = None    # TargetPoses
        self._grasping_client: Any = None  # Grasping

        # Gripper clients — std_srvs/Trigger
        self._open_gripper_client: Any = None
        self._close_gripper_client: Any = None

    # ------------------------------------------------------------------
    # Inicialização ROS2 (lazy)
    # ------------------------------------------------------------------

    def _init_ros(self) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(self._node_name)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        # Publishers
        from geometry_msgs.msg import PoseStamped, Twist
        self._cmd_vel_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)
        self._arm_pose_pub = self._node.create_publisher(PoseStamped, "/arm_pose_commands", 10)

        # Camera subscribers
        from sensor_msgs.msg import CameraInfo, Image
        self._node.create_subscription(Image,      "/camera/hand/image_raw",    self._rgb_cb,   10)
        self._node.create_subscription(Image,      "/depth/hand/image_raw",     self._depth_cb, 10)
        self._node.create_subscription(CameraInfo, "/camera/hand/camera_info",  self._info_cb,  10)

        # Arm status subscriber
        from std_msgs.msg import String
        self._node.create_subscription(String, "/arm_command_status", self._arm_status_cb, 10)

        # Grasp pipeline service clients
        from grasp_pipeline_interfaces.srv import Grasping, Graspnet, TargetPoses, ThreeDSegmentator
        self._seg_client     = self._node.create_client(ThreeDSegmentator, "tree_d_segment_service")
        self._grasp_client   = self._node.create_client(Graspnet,          "contact_graspnet_service")
        self._curobo_client  = self._node.create_client(TargetPoses,       "cu_robo_service")
        self._grasping_client = self._node.create_client(Grasping,         "grasping_service")

        # Gripper Trigger services
        from std_srvs.srv import Trigger
        self._open_gripper_client  = self._node.create_client(Trigger, "/open_gripper")
        self._close_gripper_client = self._node.create_client(Trigger, "/close_gripper")

        self._ros_thread = threading.Thread(
            target=self._executor.spin, daemon=True
        )
        self._ros_thread.start()
        self._ready = True

    def _ensure_ready(self) -> None:
        with self._lock:
            if not self._ready:
                self._init_ros()

    # ------------------------------------------------------------------
    # Camera callbacks
    # ------------------------------------------------------------------

    def _rgb_cb(self, msg: Any) -> None:
        with self._cam_lock:
            self._latest_rgb = msg

    def _depth_cb(self, msg: Any) -> None:
        with self._cam_lock:
            self._latest_depth = msg

    def _info_cb(self, msg: Any) -> None:
        with self._cam_lock:
            self._latest_k_matrix = msg

    def _arm_status_cb(self, msg: Any) -> None:
        status = msg.data.upper()
        self._arm_success = "SUCCESS" in status
        self._arm_done_event.set()

    # ------------------------------------------------------------------
    # Pipeline de grasp — 4 fases
    # ------------------------------------------------------------------

    async def segment_object(
        self,
        object_name: str,
        timeout_s: float = 10.0,
    ) -> str:
        """
        Fase 1 — Percepção: segmenta o objeto pelo nome.

        Chama tree_d_segment_service (ThreeDSegmentator).
        Usa a câmera RGB+Depth+CameraInfo mais recente do cache.

        Returns:
            JSON com objects_found e metadados da PointCloud2 segmentada,
            ou status=error se câmera indisponível ou objeto não encontrado.
        """
        self._ensure_ready()

        with self._cam_lock:
            rgb   = self._latest_rgb
            depth = self._latest_depth
            k     = self._latest_k_matrix

        if not all([rgb, depth, k]):
            return json.dumps({
                "status": "error",
                "error": "Dados de câmera não disponíveis. Verifique os tópicos "
                         "/camera/hand/image_raw, /depth/hand/image_raw e "
                         "/camera/hand/camera_info.",
            })

        if not self._seg_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({
                "status": "error",
                "error": f"tree_d_segment_service não respondeu em {timeout_s}s",
            })

        from grasp_pipeline_interfaces.srv import ThreeDSegmentator
        from std_msgs.msg import String
        req = ThreeDSegmentator.Request()
        req.rgb_img   = rgb
        req.depth_img = depth
        req.k_matrix  = k
        name_msg = String()
        name_msg.data = object_name
        req.object_name = name_msg

        try:
            future = self._seg_client.call_async(req)
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": "segment_object timed out"})

        if not res.segmented_points or res.segmented_points.width == 0:
            return json.dumps({
                "status": "error",
                "error": f"Objeto '{object_name}' não encontrado ou nenhum ponto segmentado.",
            })

        # Guarda internamente para uso na fase seguinte
        self._last_segmented_points = res.segmented_points

        return json.dumps({
            "status": "ok",
            "action": "segment_object",
            "object_name": object_name,
            "points_width": res.segmented_points.width,
            "points_height": res.segmented_points.height,
            "frame_id": res.segmented_points.header.frame_id,
        })

    async def generate_grasps(self, timeout_s: float = 15.0) -> str:
        """
        Fase 2 — Geração de grasps: Contact GraspNet sobre a PointCloud2.

        Chama contact_graspnet_service (Graspnet).
        Usa a PointCloud2 da última chamada a segment_object().

        Returns:
            JSON com n_poses e lista de poses candidatas.
        """
        self._ensure_ready()

        if not hasattr(self, "_last_segmented_points") or self._last_segmented_points is None:
            return json.dumps({
                "status": "error",
                "error": "Nenhuma PointCloud2 disponível. Execute segment_object() primeiro.",
            })

        if not self._grasp_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({
                "status": "error",
                "error": f"contact_graspnet_service não respondeu em {timeout_s}s",
            })

        from grasp_pipeline_interfaces.srv import Graspnet
        req = Graspnet.Request()
        req.points = self._last_segmented_points

        try:
            future = self._grasp_client.call_async(req)
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": "generate_grasps timed out"})

        if not res.grasp_poses.poses:
            return json.dumps({
                "status": "error",
                "error": "Nenhuma pose de grasp gerada pelo Contact GraspNet.",
            })

        self._last_grasp_poses = res.grasp_poses

        poses_summary = [
            {
                "index": i,
                "position": {
                    "x": round(p.position.x, 4),
                    "y": round(p.position.y, 4),
                    "z": round(p.position.z, 4),
                },
                "orientation": {
                    "x": round(p.orientation.x, 4),
                    "y": round(p.orientation.y, 4),
                    "z": round(p.orientation.z, 4),
                    "w": round(p.orientation.w, 4),
                },
            }
            for i, p in enumerate(res.grasp_poses.poses[:5])  # resume top-5
        ]

        return json.dumps({
            "status": "ok",
            "action": "generate_grasps",
            "n_poses": len(res.grasp_poses.poses),
            "top_poses": poses_summary,
        })

    async def plan_trajectory(self, timeout_s: float = 20.0) -> str:
        """
        Fase 3 — Planejamento: cuRobo escolhe a melhor pose e planeja trajetória.

        Chama cu_robo_service (TargetPoses).
        Usa as poses da última chamada a generate_grasps().

        Returns:
            JSON com a target_pose selecionada ou error se planejamento falhou.
        """
        self._ensure_ready()

        if not hasattr(self, "_last_grasp_poses") or self._last_grasp_poses is None:
            return json.dumps({
                "status": "error",
                "error": "Nenhuma PoseArray disponível. Execute generate_grasps() primeiro.",
            })

        if not self._curobo_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({
                "status": "error",
                "error": f"cu_robo_service não respondeu em {timeout_s}s",
            })

        from grasp_pipeline_interfaces.srv import TargetPoses
        req = TargetPoses.Request()
        req.grasp_poses = self._last_grasp_poses

        try:
            future = self._curobo_client.call_async(req)
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": "plan_trajectory timed out"})

        if not res.success:
            return json.dumps({
                "status": "error",
                "error": "cuRobo não encontrou trajetória válida para nenhuma das poses.",
            })

        self._last_target_pose = res.target_pose
        p = res.target_pose.pose

        return json.dumps({
            "status": "ok",
            "action": "plan_trajectory",
            "target_pose": {
                "frame_id": res.target_pose.header.frame_id,
                "position": {"x": round(p.position.x, 4), "y": round(p.position.y, 4), "z": round(p.position.z, 4)},
                "orientation": {"x": round(p.orientation.x, 4), "y": round(p.orientation.y, 4),
                                "z": round(p.orientation.z, 4), "w": round(p.orientation.w, 4)},
            },
        })

    async def execute_grasp(
        self,
        close_gripper: bool = True,
        timeout_s: float = 30.0,
    ) -> str:
        """
        Fase 4 — Execução: move o braço e fecha o gripper.

        Chama grasping_service (Grasping) com a target_pose do planejamento.
        O nó grasp_action_client publica em /arm_pose_commands e chama
        /open_gripper ou /close_gripper conforme o GripperCommand.

        Args:
            close_gripper: True = fechar gripper (pegar), False = abrir (soltar)
        """
        self._ensure_ready()

        if not hasattr(self, "_last_target_pose") or self._last_target_pose is None:
            return json.dumps({
                "status": "error",
                "error": "Nenhuma target_pose disponível. Execute plan_trajectory() primeiro.",
            })

        if not self._grasping_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({
                "status": "error",
                "error": f"grasping_service não respondeu em {timeout_s}s",
            })

        from control_msgs.msg import GripperCommand
        from grasp_pipeline_interfaces.srv import Grasping

        req = Grasping.Request()
        req.arm_command = self._last_target_pose

        g_cmd = GripperCommand()
        g_cmd.position   = 0.0 if close_gripper else 1.0
        g_cmd.max_effort = 10.0
        req.gripper_command = g_cmd

        try:
            future = self._grasping_client.call_async(req)
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": "execute_grasp timed out"})

        if not res.success:
            return json.dumps({
                "status": "error",
                "error": "Execução falhou — verifique o arm controller e o gripper.",
            })

        return json.dumps({
            "status": "ok",
            "action": "execute_grasp",
            "gripper": "closed" if close_gripper else "open",
        })

    async def move_to_home(self, timeout_s: float = 20.0) -> str:
        """
        Move braço para a pose home e abre o gripper.
        Usa grasping_service com a home pose hardcoded do state_machine_node.
        """
        self._ensure_ready()

        if not self._grasping_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({"status": "error", "error": "grasping_service indisponível"})

        from control_msgs.msg import GripperCommand
        from geometry_msgs.msg import PoseStamped
        from grasp_pipeline_interfaces.srv import Grasping

        home = PoseStamped()
        home.header.frame_id = "body"
        home.pose.position.x = 0.5
        home.pose.position.y = 0.0
        home.pose.position.z = 0.5
        home.pose.orientation.w = 1.0

        req = Grasping.Request()
        req.arm_command = home
        g_cmd = GripperCommand()
        g_cmd.position   = 1.0   # aberto
        g_cmd.max_effort = 10.0
        req.gripper_command = g_cmd

        try:
            future = self._grasping_client.call_async(req)
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": "move_to_home timed out"})

        return json.dumps({
            "status": "ok" if res.success else "error",
            "action": "move_to_home",
            **({"error": "Falha ao mover para home"} if not res.success else {}),
        })

    async def open_gripper(self, timeout_s: float = 5.0) -> str:
        """Abre o gripper via /open_gripper (std_srvs/Trigger)."""
        self._ensure_ready()
        return await self._call_trigger(self._open_gripper_client, "open_gripper", timeout_s)

    async def close_gripper(self, timeout_s: float = 5.0) -> str:
        """Fecha o gripper via /close_gripper (std_srvs/Trigger)."""
        self._ensure_ready()
        return await self._call_trigger(self._close_gripper_client, "close_gripper", timeout_s)

    # ------------------------------------------------------------------
    # Base móvel
    # ------------------------------------------------------------------

    async def move_forward(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vx=+abs(distance_m))

    async def move_backward(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vx=-abs(distance_m))

    async def move_left(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vy=+abs(distance_m))

    async def move_right(self, distance_m: float = 1.0) -> str:
        return await self._move_linear(vy=-abs(distance_m))

    async def rotate(self, angle_deg: float) -> str:
        import math
        self._ensure_ready()
        from geometry_msgs.msg import Twist
        omega = 0.5
        angle_rad = math.radians(angle_deg)
        dur = abs(angle_rad) / omega
        twist = Twist()
        twist.angular.z = math.copysign(omega, angle_rad)
        await self._publish_for(self._cmd_vel_pub, twist, dur)
        direction = "left (CCW)" if angle_deg > 0 else "right (CW)"
        return json.dumps({"status": "ok", "action": "rotate",
                           "angle_deg": angle_deg, "direction": direction,
                           "duration_s": round(dur, 2)})

    async def navigate_to_pose(self, x: float, y: float, yaw_deg: float = 0.0,
                                frame_id: str = "map", timeout_s: float = 60.0) -> str:
        self._ensure_ready()
        import math
        try:
            from nav2_msgs.action import NavigateToPose as Nav2Goal
            from rclpy.action import ActionClient
        except ImportError:
            return json.dumps({"status": "error", "error": "nav2_msgs não disponível"})

        client = ActionClient(self._node, Nav2Goal, "/navigate_to_pose")
        if not client.wait_for_server(timeout_sec=5.0):
            return json.dumps({"status": "error", "error": "Nav2 não disponível"})

        goal = Nav2Goal.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        half = math.radians(yaw_deg) / 2.0
        goal.pose.pose.orientation.z = math.sin(half)
        goal.pose.pose.orientation.w = math.cos(half)

        try:
            future = client.send_goal_async(goal)
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": f"Navegação timeout após {timeout_s}s"})

        return json.dumps({"status": "ok", "action": "navigate_to_pose",
                           "target": {"x": x, "y": y, "yaw_deg": yaw_deg}})

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    async def _move_linear(self, vx: float = 0.0, vy: float = 0.0) -> str:
        self._ensure_ready()
        from geometry_msgs.msg import Twist
        speed = 0.5
        dist  = abs(vx) or abs(vy)
        dur   = dist / speed if dist else 0.0
        twist = Twist()
        if dist:
            twist.linear.x = vx / dist * speed
            twist.linear.y = vy / dist * speed
        await self._publish_for(self._cmd_vel_pub, twist, dur)
        direction = ("forward" if vx > 0 else "backward" if vx < 0
                     else "left" if vy > 0 else "right")
        return json.dumps({"status": "ok", "action": f"move_{direction}",
                           "distance_m": dist, "duration_s": round(dur, 2)})

    async def _publish_for(self, publisher: Any, msg: Any, duration_s: float) -> None:
        hz = 10
        steps = max(1, int(duration_s * hz))
        for _ in range(steps):
            publisher.publish(msg)
            await asyncio.sleep(1.0 / hz)
        from geometry_msgs.msg import Twist
        if isinstance(msg, Twist):
            publisher.publish(Twist())

    async def _call_trigger(self, client: Any, label: str, timeout_s: float) -> str:
        if not client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({"status": "error", "error": f"{label} service não disponível"})
        from std_srvs.srv import Trigger
        try:
            future = client.call_async(Trigger.Request())
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
            return json.dumps({
                "status": "ok" if res.success else "error",
                "action": label,
                **({"error": res.message} if not res.success else {"message": res.message}),
            })
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": f"{label} timed out"})

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()

    # ------------------------------------------------------------------
    # Poses especiais do braço
    # ------------------------------------------------------------------

    async def move_arm_to_observe(self, timeout_s: float = 15.0) -> str:
        """
        Move o braço para a pose de observação — posição elevada que
        maximiza o campo de visão da câmera hand antes da segmentação.
        Pose calibrada para o Spot: braço esticado para frente e ligeiramente
        para cima, câmera apontando para a superfície de trabalho.
        """
        self._ensure_ready()
        if not self._grasping_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({"status": "error", "error": "grasping_service indisponível"})

        from control_msgs.msg import GripperCommand
        from geometry_msgs.msg import PoseStamped
        from grasp_pipeline_interfaces.srv import Grasping

        observe_pose = PoseStamped()
        observe_pose.header.frame_id = "body"
        observe_pose.pose.position.x = 0.7   # frente
        observe_pose.pose.position.y = 0.0   # centro
        observe_pose.pose.position.z = 0.2   # altura moderada
        # Câmera apontando ligeiramente para baixo (pitch ~30°)
        import math
        pitch = math.radians(-30)
        observe_pose.pose.orientation.y = math.sin(pitch / 2)
        observe_pose.pose.orientation.w = math.cos(pitch / 2)

        req = Grasping.Request()
        req.arm_command = observe_pose
        g_cmd = GripperCommand()
        g_cmd.position   = 1.0   # gripper aberto
        g_cmd.max_effort = 5.0
        req.gripper_command = g_cmd

        try:
            future = self._grasping_client.call_async(req)
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": "move_arm_to_observe timed out"})

        return json.dumps({
            "status": "ok" if res.success else "error",
            "action": "move_arm_to_observe",
            "pose": "observe",
            **({"error": "Falha ao mover para pose de observação"} if not res.success else {}),
        })

    async def move_arm_to_carry(self, timeout_s: float = 15.0) -> str:
        """
        Move o braço para a pose de carry — posição compacta e segura
        para transportar o objeto durante a navegação sem colidir com
        obstáculos ou desestabilizar o Spot.
        """
        self._ensure_ready()
        if not self._grasping_client.wait_for_service(timeout_sec=timeout_s):
            return json.dumps({"status": "error", "error": "grasping_service indisponível"})

        from control_msgs.msg import GripperCommand
        from geometry_msgs.msg import PoseStamped
        from grasp_pipeline_interfaces.srv import Grasping

        carry_pose = PoseStamped()
        carry_pose.header.frame_id = "body"
        carry_pose.pose.position.x = 0.4   # recolhido
        carry_pose.pose.position.y = 0.0
        carry_pose.pose.position.z = 0.3   # elevado para não arrastar
        carry_pose.pose.orientation.w = 1.0  # neutro

        req = Grasping.Request()
        req.arm_command = carry_pose
        g_cmd = GripperCommand()
        g_cmd.position   = 0.0   # gripper fechado (segurando o objeto)
        g_cmd.max_effort = 10.0
        req.gripper_command = g_cmd

        try:
            future = self._grasping_client.call_async(req)
            res = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except asyncio.TimeoutError:
            return json.dumps({"status": "error", "error": "move_arm_to_carry timed out"})

        return json.dumps({
            "status": "ok" if res.success else "error",
            "action": "move_arm_to_carry",
            "pose": "carry",
            **({"error": "Falha ao mover para pose de carry"} if not res.success else {}),
        })
