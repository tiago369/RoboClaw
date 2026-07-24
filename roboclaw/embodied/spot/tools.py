"""
tools.py
========
SpotToolGroup — tools do Spot no padrão Tool/ToolRegistry do roboclaw.

Três grupos baseados no grasp_pipeline real:

  spot_base        — base móvel: linear, rotação, navegação
  spot_arm         — pipeline completo de grasp em 4 fases:
                     segment → grasps → plan → execute
                     + home, open/close gripper
  spot_perception  — câmera: status e leitura de câmera hand

Integração no AgentLoop:
    from roboclaw.embodied.spot.tools import create_spot_tools
    from roboclaw.embodied.spot.service import SpotService
    for tool in create_spot_tools(SpotService(), episode_memory):
        loop.tools.register(tool)
"""
from __future__ import annotations

import json
from typing import Any

from roboclaw.agent.tools.base import Tool

_BASE_ACTIONS = [
    "move_forward", "move_backward", "move_left", "move_right",
    "rotate_left", "rotate_right", "navigate_to_pose",
]

_ARM_ACTIONS = [
    "segment_object",       # Fase 1 — percepção: Gemini + depth → PointCloud2
    "generate_grasps",      # Fase 2 — Contact GraspNet → PoseArray
    "plan_trajectory",      # Fase 3 — cuRobo → target_pose
    "execute_grasp",        # Fase 4 — executa movimento + fecha gripper
    "full_grasp_pipeline",  # Fases 1-4 encadeadas num único tool call
    "move_to_home",         # Move braço para home e abre gripper
    "move_to_observe",      # Move braço para pose de observação (pré-segmentação)
    "move_to_carry",        # Move braço para pose de carry (pós-grasp, durante nav)
    "open_gripper",         # Abre gripper via /open_gripper
    "close_gripper",        # Fecha gripper via /close_gripper
]

_PERCEPTION_ACTIONS = [
    "camera_status",       # Verifica se câmera hand está publicando
]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _SpotTool(Tool):
    def __init__(self, spot_service: Any, episode_memory: Any = None) -> None:
        self._svc = spot_service
        self._mem = episode_memory

    def _retrieve(self, action: str, kwargs: dict) -> None:
        if self._mem is None:
            return
        query = f"{action} {' '.join(str(v) for v in kwargs.values())}"
        self._mem.retrieve(query, top_k=3, as_context_string=True)

    def _store(self, action: str, result_json: str, extra: dict | None = None) -> None:
        if self._mem is None:
            return
        try:
            data = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            data = {}
        outcome = "success" if data.get("status") == "ok" else "failed"
        env = {k: v for k, v in data.items() if k != "status"}
        if extra:
            env.update(extra)
        self._mem.store(subtask=action, outcome=outcome, env_state=env)


# ---------------------------------------------------------------------------
# Grupo 1: Base móvel
# ---------------------------------------------------------------------------

class SpotBaseTool(_SpotTool):
    """
    Controle da base móvel do Spot.

    Ações:
      move_forward / move_backward / move_left / move_right
        distance_m: float = 1.0

      rotate_left / rotate_right
        angle_deg: float = 90.0

      navigate_to_pose
        x: float, y: float (obrigatórios)
        yaw_deg: float = 0.0
        frame_id: str = "map"
    """

    @property
    def name(self) -> str:
        return "spot_base"

    @property
    def description(self) -> str:
        return (
            "Controla a base móvel do Spot. "
            "Movimentos lineares (forward/backward/left/right), rotação in-place "
            "(rotate_left/rotate_right) e navegação autônoma (navigate_to_pose). "
            "Use navigate_to_pose para deslocamentos longos entre ambientes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": _BASE_ACTIONS,
                           "description": "Ação de base a executar."},
                "distance_m": {"type": "number", "minimum": 0.01, "maximum": 10.0,
                               "description": "Distância em metros. Default 1.0."},
                "angle_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0,
                              "description": "Ângulo em graus. Default 90."},
                "x": {"type": "number", "description": "X no frame map (metros). Obrigatório para navigate_to_pose."},
                "y": {"type": "number", "description": "Y no frame map (metros). Obrigatório para navigate_to_pose."},
                "yaw_deg": {"type": "number", "description": "Heading no destino em graus. Default 0."},
                "frame_id": {"type": "string", "description": "Frame da navegação. Default 'map'."},
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in _BASE_ACTIONS:
            return f"Erro: ação desconhecida '{action}' para spot_base."

        self._retrieve(action, kwargs)

        if action == "move_forward":
            result = await self._svc.move_forward(kwargs.get("distance_m", 1.0))
        elif action == "move_backward":
            result = await self._svc.move_backward(kwargs.get("distance_m", 1.0))
        elif action == "move_left":
            result = await self._svc.move_left(kwargs.get("distance_m", 1.0))
        elif action == "move_right":
            result = await self._svc.move_right(kwargs.get("distance_m", 1.0))
        elif action == "rotate_left":
            result = await self._svc.rotate(+abs(kwargs.get("angle_deg", 90.0)))
        elif action == "rotate_right":
            result = await self._svc.rotate(-abs(kwargs.get("angle_deg", 90.0)))
        elif action == "navigate_to_pose":
            if "x" not in kwargs or "y" not in kwargs:
                return "Erro: navigate_to_pose requer x e y."
            result = await self._svc.navigate_to_pose(
                x=kwargs["x"], y=kwargs["y"],
                yaw_deg=kwargs.get("yaw_deg", 0.0),
                frame_id=kwargs.get("frame_id", "map"),
            )
        else:
            return f"Erro: ação '{action}' não tratada."

        self._store(action, result)

        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result

        if data.get("status") != "ok":
            return f"Erro ao executar {action}: {data.get('error', 'desconhecido')}"

        if action in ("move_forward", "move_backward", "move_left", "move_right"):
            d = action.replace("move_", "")
            return f"Base movida {d} {data['distance_m']:.2f}m em {data['duration_s']:.1f}s."
        if "rotate" in action:
            return f"Base rotacionada {data['direction']} {abs(data['angle_deg']):.1f}° em {data['duration_s']:.1f}s."
        if action == "navigate_to_pose":
            t = data.get("target", {})
            return f"Navegação completa: destino ({t.get('x', 0):.2f}, {t.get('y', 0):.2f})."
        return result


# ---------------------------------------------------------------------------
# Grupo 2: Braço + Pipeline de Grasp
# ---------------------------------------------------------------------------

class SpotArmTool(_SpotTool):
    """
    Controla o braço do Spot e executa o pipeline completo de grasp.

    O pipeline usa os serviços reais do grasp_pipeline_interfaces:

      segment_object   → ThreeDSegmentator (Gemini + depth → PointCloud2)
      generate_grasps  → Contact GraspNet (PointCloud2 → PoseArray)
      plan_trajectory  → cuRobo (PoseArray → PoseStamped válida)
      execute_grasp    → Grasping (PoseStamped → executa no hardware)

      full_grasp_pipeline → executa as 4 fases encadeadas automaticamente

    Controle de gripper:
      open_gripper   → /open_gripper (std_srvs/Trigger)
      close_gripper  → /close_gripper (std_srvs/Trigger)
      move_to_home   → pose home + abre gripper
    """

    @property
    def name(self) -> str:
        return "spot_arm"

    @property
    def description(self) -> str:
        return (
            "Controla o braço do Spot e executa o pipeline de grasp autônomo. "
            "Para pegar um objeto: use full_grasp_pipeline com o nome do objeto. "
            "Fases individuais: segment_object → generate_grasps → plan_trajectory → execute_grasp. "
            "Controle do gripper: open_gripper / close_gripper. "
            "Retornar à posição segura: move_to_home."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _ARM_ACTIONS,
                    "description": "Ação do braço a executar.",
                },
                "object_name": {
                    "type": "string",
                    "description": (
                        "Nome do objeto a pegar. Obrigatório para segment_object "
                        "e full_grasp_pipeline. Ex: 'cup', 'bottle', 'red box'."
                    ),
                },
                "close_gripper": {
                    "type": "boolean",
                    "description": (
                        "Para execute_grasp: true = fechar gripper (pegar objeto), "
                        "false = abrir (soltar). Default true."
                    ),
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in _ARM_ACTIONS:
            return f"Erro: ação desconhecida '{action}' para spot_arm."

        self._retrieve(action, kwargs)

        # -- Pipeline completo --
        if action == "full_grasp_pipeline":
            return await self._run_full_pipeline(kwargs)

        # -- Fases individuais --
        if action == "segment_object":
            name = kwargs.get("object_name", "")
            if not name:
                return "Erro: segment_object requer object_name."
            result = await self._svc.segment_object(name)

        elif action == "generate_grasps":
            result = await self._svc.generate_grasps()

        elif action == "plan_trajectory":
            result = await self._svc.plan_trajectory()

        elif action == "execute_grasp":
            result = await self._svc.execute_grasp(
                close_gripper=kwargs.get("close_gripper", True)
            )

        elif action == "move_to_home":
            result = await self._svc.move_to_home()

        elif action == "move_to_observe":
            result = await self._svc.move_arm_to_observe()

        elif action == "move_to_carry":
            result = await self._svc.move_arm_to_carry()

        elif action == "open_gripper":
            result = await self._svc.open_gripper()

        elif action == "close_gripper":
            result = await self._svc.close_gripper()

        else:
            return f"Erro: ação '{action}' não tratada."

        self._store(action, result)
        return self._format(action, result)

    async def _run_full_pipeline(self, kwargs: dict) -> str:
        """Executa as 4 fases em sequência com logging de cada etapa."""
        name = kwargs.get("object_name", "")
        if not name:
            return "Erro: full_grasp_pipeline requer object_name."

        steps = []

        # Fase 1 — Segmentação
        r1 = await self._svc.segment_object(name)
        self._store("segment_object", r1)
        d1 = _parse(r1)
        if d1.get("status") != "ok":
            return f"Pipeline falhou na Fase 1 (segmentação): {d1.get('error')}"
        steps.append(f"Segmentação: {d1.get('points_width', '?')} pontos encontrados para '{name}'")

        # Fase 2 — Geração de grasps
        r2 = await self._svc.generate_grasps()
        self._store("generate_grasps", r2)
        d2 = _parse(r2)
        if d2.get("status") != "ok":
            return f"Pipeline falhou na Fase 2 (Contact GraspNet): {d2.get('error')}"
        steps.append(f"Grasps: {d2.get('n_poses', '?')} poses candidatas geradas")

        # Fase 3 — Planejamento
        r3 = await self._svc.plan_trajectory()
        self._store("plan_trajectory", r3)
        d3 = _parse(r3)
        if d3.get("status") != "ok":
            return f"Pipeline falhou na Fase 3 (cuRobo): {d3.get('error')}"
        tp = d3.get("target_pose", {}).get("position", {})
        steps.append(f"Trajetória: pose alvo em ({tp.get('x',0):.3f}, {tp.get('y',0):.3f}, {tp.get('z',0):.3f})")

        # Fase 4 — Execução
        r4 = await self._svc.execute_grasp(close_gripper=True)
        self._store("execute_grasp", r4)
        d4 = _parse(r4)
        if d4.get("status") != "ok":
            return f"Pipeline falhou na Fase 4 (execução): {d4.get('error')}"
        steps.append("Execução: braço movido e gripper fechado")

        return (
            f"Pipeline completo para '{name}':\n" +
            "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        )

    @staticmethod
    def _format(action: str, result_json: str) -> str:
        data = _parse(result_json)
        if data.get("status") != "ok":
            return f"Erro em {action}: {data.get('error', 'desconhecido')}"

        msgs = {
            "segment_object":  lambda d: f"Segmentação completa: {d.get('points_width','?')} pontos de '{d.get('object_name','?')}' no frame {d.get('frame_id','?')}.",
            "generate_grasps": lambda d: f"{d.get('n_poses','?')} poses de grasp geradas pelo Contact GraspNet.",
            "plan_trajectory": lambda d: f"Trajetória planejada (cuRobo): pose alvo em frame '{d.get('target_pose',{}).get('frame_id','?')}'.",
            "execute_grasp":   lambda d: f"Execução concluída. Gripper: {d.get('gripper','?')}.",
            "move_to_home":    lambda _: "Braço retornou à posição home. Gripper aberto.",
            "move_to_observe": lambda _: "Braço em pose de observação. Câmera posicionada para segmentação.",
            "move_to_carry":   lambda _: "Braço em pose de carry. Pronto para navegar com o objeto.",
            "open_gripper":    lambda _: "Gripper aberto.",
            "close_gripper":   lambda _: "Gripper fechado.",
        }
        fn = msgs.get(action)
        return fn(data) if fn else result_json


# ---------------------------------------------------------------------------
# Grupo 3: Percepção — status de câmera
# ---------------------------------------------------------------------------

class SpotPerceptionTool(_SpotTool):
    """
    Verifica o status das câmeras do Spot usadas no pipeline de grasp.

    Ação:
      camera_status — verifica se RGB, depth e camera_info estão disponíveis
                      (necessário antes de segment_object)
    """

    @property
    def name(self) -> str:
        return "spot_perception"

    @property
    def description(self) -> str:
        return (
            "Verifica o status das câmeras do Spot usadas no pipeline de grasp. "
            "Use camera_status para confirmar que RGB, depth e camera_info estão "
            "disponíveis antes de executar segment_object ou full_grasp_pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _PERCEPTION_ACTIONS,
                    "description": "Ação de percepção.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in _PERCEPTION_ACTIONS:
            return f"Erro: ação desconhecida '{action}' para spot_perception."

        self._retrieve(action, kwargs)

        if action == "camera_status":
            result = await self._camera_status()
        else:
            return f"Erro: ação '{action}' não tratada."

        self._store(action, result)
        return self._format_perception(action, result)

    async def _camera_status(self) -> str:
        """Verifica se os três streams de câmera estão disponíveis."""
        self._svc._ensure_ready()
        with self._svc._cam_lock:
            has_rgb   = self._svc._latest_rgb   is not None
            has_depth = self._svc._latest_depth is not None
            has_info  = self._svc._latest_k_matrix is not None

        status = "ok" if all([has_rgb, has_depth, has_info]) else "warning"
        missing = [
            t for t, ok in [("rgb", has_rgb), ("depth", has_depth), ("camera_info", has_info)]
            if not ok
        ]

        return json.dumps({
            "status": status,
            "action": "camera_status",
            "rgb_available":   has_rgb,
            "depth_available": has_depth,
            "camera_info_available": has_info,
            "ready_for_grasp": all([has_rgb, has_depth, has_info]),
            "missing_streams": missing,
            "topics": {
                "rgb":   "/camera/hand/image_raw",
                "depth": "/depth/hand/image_raw",
                "info":  "/camera/hand/camera_info",
            },
        })

    @staticmethod
    def _format_perception(action: str, result_json: str) -> str:
        data = _parse(result_json)
        if action == "camera_status":
            if data.get("ready_for_grasp"):
                return "Câmeras prontas: RGB ✓, Depth ✓, CameraInfo ✓. Pipeline de grasp pode ser executado."
            missing = data.get("missing_streams", [])
            return (
                f"Câmeras NÃO prontas. Streams em falta: {', '.join(missing)}. "
                "Verifique se o driver do Spot e os tópicos de câmera estão ativos."
            )
        return result_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result_json: str) -> dict:
    try:
        return json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Grupo 4: Localização — lugares nomeados
# ---------------------------------------------------------------------------

_LOCATION_ACTIONS = [
    "save_location",        # salva posição atual ou coordenadas fornecidas com um nome
    "go_to_location",       # navega para um lugar pelo nome
    "list_locations",       # lista todos os lugares salvos
    "save_current_as",      # salva a posição atual do robô com um nome
    "delete_location",      # remove um lugar salvo
]


class SpotLocationTool(_SpotTool):
    """
    Gerencia lugares nomeados e navega para eles.

    Permite ao agente lembrar e reutilizar posições no mapa sem precisar
    de coordenadas explícitas. Essencial para tarefas como:
      "vá para a mesa da cozinha e traga o martelo"

    Ações:
      save_location     — salva (x, y, yaw_deg) com um nome descritivo
      go_to_location    — navega até um lugar salvo pelo nome
      list_locations    — lista todos os lugares conhecidos
      save_current_as   — salva posição atual do robô com um nome
      delete_location   — remove um lugar da memória
    """

    def __init__(
        self,
        spot_service: Any,
        location_memory: Any,
        episode_memory: Any = None,
    ) -> None:
        super().__init__(spot_service, episode_memory)
        self._lm = location_memory

    @property
    def name(self) -> str:
        return "spot_location"

    @property
    def description(self) -> str:
        return (
            "Gerencia lugares nomeados e navega para eles. "
            "Use save_location para registrar uma posição com nome descritivo. "
            "Use go_to_location para navegar até um lugar pelo nome (ex: 'mesa da cozinha'). "
            "Use list_locations para ver todos os lugares conhecidos. "
            "Use save_current_as para salvar onde o robô está agora."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _LOCATION_ACTIONS,
                    "description": "Ação de localização a executar.",
                },
                "name": {
                    "type": "string",
                    "description": "Nome do lugar. Ex: 'mesa da cozinha', 'origem', 'prateleira B'.",
                },
                "x": {"type": "number", "description": "Coordenada X no frame map (metros)."},
                "y": {"type": "number", "description": "Coordenada Y no frame map (metros)."},
                "yaw_deg": {"type": "number", "description": "Orientação em graus. Default 0."},
                "description": {
                    "type": "string",
                    "description": "Descrição opcional do lugar.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in _LOCATION_ACTIONS:
            return f"Erro: ação desconhecida '{action}' para spot_location."

        self._retrieve(action, kwargs)

        if action == "save_location":
            return self._save_location(kwargs)

        elif action == "go_to_location":
            return await self._go_to_location(kwargs)

        elif action == "list_locations":
            return self._list_locations()

        elif action == "save_current_as":
            return await self._save_current_as(kwargs)

        elif action == "delete_location":
            return self._delete_location(kwargs)

        return f"Erro: ação '{action}' não tratada."

    def _save_location(self, kwargs: dict) -> str:
        name = kwargs.get("name", "")
        if not name:
            return "Erro: save_location requer um name."
        if "x" not in kwargs or "y" not in kwargs:
            return "Erro: save_location requer x e y."
        self._lm.save(
            name=name,
            x=kwargs["x"],
            y=kwargs["y"],
            yaw_deg=kwargs.get("yaw_deg", 0.0),
            description=kwargs.get("description", ""),
        )
        return f"Lugar '{name}' salvo em ({kwargs['x']:.2f}, {kwargs['y']:.2f}, yaw={kwargs.get('yaw_deg', 0):.1f}°)."

    async def _go_to_location(self, kwargs: dict) -> str:
        name = kwargs.get("name", "")
        if not name:
            return "Erro: go_to_location requer um name."
        loc = self._lm.get(name)
        if loc is None:
            known = [v["name"] for v in self._lm.list_all()]
            known_str = ", ".join(f"'{n}'" for n in known) if known else "nenhum"
            return (
                f"Lugar '{name}' não encontrado na memória. "
                f"Lugares conhecidos: {known_str}. "
                f"Use save_location para registrar este lugar primeiro."
            )
        result = await self._svc.navigate_to_pose(
            x=loc["x"],
            y=loc["y"],
            yaw_deg=loc.get("yaw_deg", 0.0),
            frame_id=loc.get("frame_id", "map"),
        )
        data = _parse(result)
        if data.get("status") == "ok":
            self._store("go_to_location", result, {"location_name": name})
            return f"Cheguei em '{loc['name']}' ({loc['x']:.2f}, {loc['y']:.2f})."
        return f"Erro ao navegar para '{name}': {data.get('error', 'desconhecido')}"

    def _list_locations(self) -> str:
        locs = self._lm.list_all()
        if not locs:
            return "Nenhum lugar salvo ainda. Use save_location ou save_current_as para registrar posições."
        lines = ["Lugares conhecidos:"]
        for loc in locs:
            desc = f" — {loc['description']}" if loc.get("description") else ""
            lines.append(
                f"  • '{loc['name']}': ({loc['x']:.2f}, {loc['y']:.2f}, {loc.get('yaw_deg', 0):.1f}°){desc}"
            )
        return "\n".join(lines)

    async def _save_current_as(self, kwargs: dict) -> str:
        name = kwargs.get("name", "")
        if not name:
            return "Erro: save_current_as requer um name."

        # Tenta obter posição atual via odom
        pos = await self._get_current_position()
        if pos is None:
            return (
                "Não foi possível obter a posição atual do robô. "
                "Use save_location com x e y explícitos."
            )

        self._lm.save(
            name=name,
            x=pos["x"],
            y=pos["y"],
            yaw_deg=pos.get("yaw_deg", 0.0),
            description=kwargs.get("description", "salvo automaticamente"),
        )
        return (
            f"Posição atual salva como '{name}': "
            f"({pos['x']:.2f}, {pos['y']:.2f}, yaw={pos.get('yaw_deg', 0):.1f}°)."
        )

    def _delete_location(self, kwargs: dict) -> str:
        name = kwargs.get("name", "")
        if not name:
            return "Erro: delete_location requer um name."
        if self._lm.delete(name):
            return f"Lugar '{name}' removido da memória."
        return f"Lugar '{name}' não encontrado."

    async def _get_current_position(self) -> dict | None:
        """Tenta ler a posição atual do robô via /odom."""
        try:
            self._svc._ensure_ready()
            import asyncio

            from nav_msgs.msg import Odometry

            result: dict | None = None
            done = asyncio.Event()

            def _odom_cb(msg):
                nonlocal result
                p = msg.pose.pose.position
                import math
                q = msg.pose.pose.orientation
                yaw = math.degrees(
                    math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
                )
                result = {"x": round(p.x, 3), "y": round(p.y, 3), "yaw_deg": round(yaw, 1)}
                done.set()

            sub = self._svc._node.create_subscription(Odometry, "/odom", _odom_cb, 1)
            try:
                await asyncio.wait_for(done.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass
            finally:
                self._svc._node.destroy_subscription(sub)

            return result
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_spot_tools(
    spot_service: Any,
    episode_memory: Any = None,
    location_memory: Any = None,
) -> list[_SpotTool]:
    """
    Instancia todos os tools do Spot.

    Args:
        spot_service:     instância de SpotService
        episode_memory:   instância de RoboClawMemory (opcional)
        location_memory:  instância de LocationMemory (opcional)
                          Se None, cria uma com o path default.

    Returns:
        Lista com 4 tools prontos para o ToolRegistry.
    """
    from roboclaw.embodied.spot.location_memory import LocationMemory

    lm = location_memory or LocationMemory()
    kw = {"spot_service": spot_service, "episode_memory": episode_memory}
    return [
        SpotBaseTool(**kw),
        SpotArmTool(**kw),
        SpotPerceptionTool(**kw),
        SpotLocationTool(**kw, location_memory=lm),
    ]
