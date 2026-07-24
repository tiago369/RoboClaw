"""
location_memory.py
==================
Gerencia lugares nomeados com coordenadas (x, y, yaw) no frame map.

Persiste em JSON em ~/.roboclaw/locations.json.
Usado pelo spot_location tool para que o agente navegue por nome.

Exemplos:
    lm = LocationMemory()
    lm.save("mesa da cozinha", x=3.2, y=1.5, yaw_deg=90.0)
    loc = lm.get("mesa")        # busca parcial case-insensitive
    lm.save_current("origem")   # salva posição atual (via odom)
"""
from __future__ import annotations

import json
import pathlib
import time

_DEFAULT_PATH = pathlib.Path("~/.roboclaw/locations.json").expanduser()


class LocationMemory:
    """Armazena e recupera lugares nomeados com pose (x, y, yaw_deg)."""

    def __init__(self, path: str | pathlib.Path = _DEFAULT_PATH) -> None:
        self._path = pathlib.Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(
        self,
        name: str,
        x: float,
        y: float,
        yaw_deg: float = 0.0,
        frame_id: str = "map",
        description: str = "",
    ) -> None:
        """Salva ou atualiza um lugar nomeado."""
        key = name.lower().strip()
        self._data[key] = {
            "name": name,
            "x": float(x),
            "y": float(y),
            "yaw_deg": float(yaw_deg),
            "frame_id": frame_id,
            "description": description,
            "saved_at": time.time(),
        }
        self._persist()

    def get(self, name: str) -> dict | None:
        """
        Busca um lugar por nome (parcial, case-insensitive).
        Retorna o dict com x, y, yaw_deg, frame_id ou None se não encontrado.
        """
        key = name.lower().strip()

        # Exact match
        if key in self._data:
            return self._data[key]

        # Partial match — retorna o primeiro que contém o termo
        for k, v in self._data.items():
            if key in k or k in key:
                return v

        return None

    def delete(self, name: str) -> bool:
        key = name.lower().strip()
        if key in self._data:
            del self._data[key]
            self._persist()
            return True
        return False

    def list_all(self) -> list[dict]:
        return sorted(self._data.values(), key=lambda v: v.get("saved_at", 0))

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _persist(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
