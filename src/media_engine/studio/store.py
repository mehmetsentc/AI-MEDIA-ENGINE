"""Persistent projects, scenes, and assets. Generation jobs stay in the job store."""
from __future__ import annotations

import json
import uuid
from typing import Optional

from media_engine.db import connect, init_schema

ASSET_TYPES = ("text", "voice", "music", "image", "video")
TRACKS = ASSET_TYPES


class StudioError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class StudioStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_schema(db_path)

    def create_project(self, name: str, now: str) -> dict:
        title = _name(name)
        project_id = "proj_" + uuid.uuid4().hex
        self._write(
            "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (project_id, title, now, now),
        )
        return self.get_project(project_id)

    def list_projects(self) -> list[dict]:
        rows = self._query("SELECT * FROM projects ORDER BY updated_at DESC")
        return [_project(row) for row in rows]

    def get_project(self, project_id: str) -> dict:
        row = self._one("SELECT * FROM projects WHERE id = ?", (project_id,))
        if row is None:
            raise StudioError("PROJECT_NOT_FOUND")
        project = _project(row)
        project["scenes"] = self.list_scenes(project_id)
        return project

    def update_project(self, project_id: str, name: str, now: str) -> dict:
        self.get_project(project_id)
        self._write(
            "UPDATE projects SET name = ?, updated_at = ? WHERE id = ?",
            (_name(name), now, project_id),
        )
        return self.get_project(project_id)

    def create_scene(self, project_id: str, name: str, now: str) -> dict:
        self.get_project(project_id)
        title = _name(name or "Scene")
        count = self._one("SELECT COUNT(*) AS n FROM scenes WHERE project_id = ?", (project_id,))
        scene_id = "scn_" + uuid.uuid4().hex
        self._write(
            """
            INSERT INTO scenes (id, project_id, name, position, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (scene_id, project_id, title, int(count["n"]) if count else 0, now, now),
        )
        for asset_type in TRACKS:
            self.create_asset(scene_id, asset_type, "", now, placeholder=True)
        return self.get_scene(scene_id)

    def list_scenes(self, project_id: str) -> list[dict]:
        rows = self._query(
            "SELECT * FROM scenes WHERE project_id = ? ORDER BY position, created_at",
            (project_id,),
        )
        return [self.get_scene(row["id"]) for row in rows]

    def get_scene(self, scene_id: str) -> dict:
        row = self._one("SELECT * FROM scenes WHERE id = ?", (scene_id,))
        if row is None:
            raise StudioError("SCENE_NOT_FOUND")
        scene = _scene(row)
        scene["assets"] = self.list_assets(scene_id)
        return scene

    def update_scene(self, scene_id: str, name: str, now: str) -> dict:
        self.get_scene(scene_id)
        self._write(
            "UPDATE scenes SET name = ?, updated_at = ? WHERE id = ?",
            (_name(name), now, scene_id),
        )
        return self.get_scene(scene_id)

    def delete_scene(self, scene_id: str) -> None:
        self.get_scene(scene_id)
        self._write("DELETE FROM studio_assets WHERE scene_id = ?", (scene_id,))
        self._write("DELETE FROM scenes WHERE id = ?", (scene_id,))

    def create_asset(self, scene_id: str, asset_type: str, prompt: str, now: str,
                     *, settings: Optional[dict] = None, placeholder: bool = False) -> dict:
        scene = self._one("SELECT * FROM scenes WHERE id = ?", (scene_id,))
        if scene is None:
            raise StudioError("SCENE_NOT_FOUND")
        kind = _type(asset_type)
        text = "" if placeholder else _prompt(prompt, required=kind == "image")
        asset_id = "ast_" + uuid.uuid4().hex
        stored = _settings(settings or {})
        engine = "qwen-image" if kind == "image" else None
        model = "Qwen/Qwen-Image" if kind == "image" else None
        self._write(
            """
            INSERT INTO studio_assets (
                id, project_id, scene_id, asset_type, engine, model, prompt, status,
                artifact_id, job_id, created_at, updated_at, settings_json, history_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
            """,
            (
                asset_id, scene["project_id"], scene_id, kind, engine, model, text, "draft",
                now, now, json.dumps(stored, separators=(",", ":")), "[]",
            ),
        )
        return self.get_asset(asset_id)

    def list_assets(self, scene_id: str) -> list[dict]:
        rows = self._query(
            "SELECT * FROM studio_assets WHERE scene_id = ? ORDER BY created_at",
            (scene_id,),
        )
        return [_asset(row) for row in rows]

    def get_asset(self, asset_id: str) -> dict:
        row = self._one("SELECT * FROM studio_assets WHERE id = ?", (asset_id,))
        if row is None:
            raise StudioError("ASSET_NOT_FOUND")
        return _asset(row)

    def update_asset(self, asset_id: str, now: str, *, prompt: Optional[str] = None,
                     settings: Optional[dict] = None, job_id: Optional[str] = None,
                     artifact_id: Optional[str] = None, status: Optional[str] = None) -> dict:
        current = self.get_asset(asset_id)
        text = current["prompt"] if prompt is None else _prompt(prompt, required=False)
        stored = current["settings"] if settings is None else _settings({**current["settings"], **settings})
        history = list(current["history"])
        current_job = current["job_id"]
        if job_id:
            current_job = job_id
            if job_id not in history:
                history.append(job_id)
        current_artifact = current["artifact_id"] if artifact_id is None else artifact_id
        current_status = current["status"] if status is None else status
        if job_id and status is None:
            current_status = "queued"
        self._write(
            """
            UPDATE studio_assets
               SET prompt = ?, settings_json = ?, job_id = ?, artifact_id = ?,
                   status = ?, history_json = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                text, json.dumps(stored, separators=(",", ":")), current_job, current_artifact,
                current_status, json.dumps(history), now, asset_id,
            ),
        )
        return self.get_asset(asset_id)

    def _write(self, sql: str, params: tuple) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _query(self, sql: str, params: tuple = ()) -> list:
        conn = connect(self.db_path)
        try:
            return list(conn.execute(sql, params))
        finally:
            conn.close()

    def _one(self, sql: str, params: tuple):
        rows = self._query(sql, params)
        return rows[0] if rows else None


def _name(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
        raise StudioError("NAME_REQUIRED")
    return value.strip()


def _prompt(value: object, *, required: bool) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str) or len(value) > 4000:
        raise StudioError("PROMPT_REQUIRED")
    text = value.strip()
    if required and not text:
        raise StudioError("PROMPT_REQUIRED")
    return text


def _type(value: object) -> str:
    if not isinstance(value, str) or value not in ASSET_TYPES:
        raise StudioError("ASSET_TYPE_UNSUPPORTED")
    return value


def _settings(value: dict) -> dict:
    width = value.get("width", 1024)
    height = value.get("height", 1024)
    seed = value.get("seed")
    if isinstance(width, bool) or not isinstance(width, int) or width != 1024:
        raise StudioError("RESOLUTION_UNSUPPORTED")
    if isinstance(height, bool) or not isinstance(height, int) or height != 1024:
        raise StudioError("RESOLUTION_UNSUPPORTED")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**31):
        raise StudioError("SEED_INVALID")
    stored = {"width": width, "height": height}
    if seed is not None:
        stored["seed"] = seed
    return stored


def _project(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _scene(row) -> dict:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "name": row["name"],
        "position": row["position"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _asset(row) -> dict:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "scene_id": row["scene_id"],
        "type": row["asset_type"],
        "engine": row["engine"],
        "model": row["model"],
        "prompt": row["prompt"],
        "status": row["status"],
        "artifact_id": row["artifact_id"],
        "job_id": row["job_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "settings": json.loads(row["settings_json"]),
        "history": json.loads(row["history_json"]),
    }
