"""Studio HTTP routes. Image bytes still move through the existing image API."""
from __future__ import annotations

import json
import re
from typing import Optional

from media_engine.jobs.model import JobNotFound
from media_engine.orchestrator.controller import MediaController
from media_engine.studio.present import present_job
from media_engine.studio.store import StudioError, StudioStore

_PROJECTS = re.compile(r"^/v1/projects$")
_PROJECT = re.compile(r"^/v1/projects/([A-Za-z0-9_-]+)$")
_SCENES = re.compile(r"^/v1/projects/([A-Za-z0-9_-]+)/scenes$")
_SCENE = re.compile(r"^/v1/scenes/([A-Za-z0-9_-]+)$")
_ASSETS = re.compile(r"^/v1/scenes/([A-Za-z0-9_-]+)/assets$")
_ASSET = re.compile(r"^/v1/assets/([A-Za-z0-9_-]+)$")


def try_studio(controller: MediaController, method: str, path: str,
               body: Optional[bytes]) -> Optional[tuple[int, dict]]:
    if not path.startswith("/v1/projects") and not path.startswith("/v1/scenes") and not path.startswith("/v1/assets"):
        return None
    store = StudioStore(controller.jobs.db_path)
    now = controller.clock.iso()
    try:
        return _route(controller, store, method, path, body, now)
    except StudioError as exc:
        status = 404 if exc.code.endswith("_NOT_FOUND") else 400
        return status, {"error": exc.code}


def _route(controller, store: StudioStore, method: str, path: str,
           body: Optional[bytes], now: str) -> tuple[int, dict]:
    if method == "GET" and _PROJECTS.fullmatch(path):
        return 200, {"projects": [_summary(store, controller, item) for item in store.list_projects()]}
    if method == "POST" and _PROJECTS.fullmatch(path):
        payload = _object(body)
        project = store.create_project(str(payload.get("name", "")), now)
        return 201, _summary(store, controller, project)
    found = _PROJECT.fullmatch(path)
    if found and method == "GET":
        return 200, _summary(store, controller, store.get_project(found.group(1)))
    if found and method == "PATCH":
        payload = _object(body)
        project = store.update_project(found.group(1), str(payload.get("name", "")), now)
        return 200, _summary(store, controller, project)
    scenes = _SCENES.fullmatch(path)
    if scenes and method == "POST":
        payload = _object(body)
        scene = store.create_scene(scenes.group(1), str(payload.get("name", "Scene")), now)
        return 201, _scene_view(store, controller, scene)
    scene = _SCENE.fullmatch(path)
    if scene and method == "GET":
        return 200, _scene_view(store, controller, store.get_scene(scene.group(1)))
    if scene and method == "PATCH":
        payload = _object(body)
        updated = store.update_scene(scene.group(1), str(payload.get("name", "")), now)
        return 200, _scene_view(store, controller, updated)
    if scene and method == "DELETE":
        store.delete_scene(scene.group(1))
        return 200, {"deleted": True}
    assets = _ASSETS.fullmatch(path)
    if assets and method == "POST":
        payload = _object(body)
        asset = store.create_asset(
            assets.group(1), str(payload.get("type", "")), str(payload.get("prompt", "")), now,
            settings=_image_settings(payload),
        )
        return 201, _asset_view(controller, asset)
    asset = _ASSET.fullmatch(path)
    if asset and method == "GET":
        return 200, _asset_view(controller, store.get_asset(asset.group(1)))
    if asset and method == "PATCH":
        payload = _object(body)
        updated = store.update_asset(
            asset.group(1), now,
            prompt=payload.get("prompt") if "prompt" in payload else None,
            settings=_image_settings(payload) if _has_settings(payload) else None,
            job_id=payload.get("job_id") if "job_id" in payload else None,
        )
        return 200, _asset_view(controller, updated)
    return 404, {"error": "NOT_FOUND"}


def _summary(store: StudioStore, controller, project: dict) -> dict:
    full = project if "scenes" in project else store.get_project(project["id"])
    scenes = [_scene_view(store, controller, scene) for scene in full["scenes"]]
    costs = [asset.get("actual_cost") for scene in scenes for asset in scene["assets"]]
    numbers = [item for item in costs if item]
    return {
        "id": full["id"],
        "name": full["name"],
        "created_at": full["created_at"],
        "updated_at": full["updated_at"],
        "scenes": scenes,
        "actual_cost": numbers[-1] if numbers else None,
        "estimated_cost": None,
    }


def _scene_view(store: StudioStore, controller, scene: dict) -> dict:
    assets = scene.get("assets")
    if assets is None:
        assets = store.list_assets(scene["id"])
    return {
        "id": scene["id"],
        "project_id": scene["project_id"],
        "name": scene["name"],
        "position": scene["position"],
        "created_at": scene["created_at"],
        "updated_at": scene["updated_at"],
        "assets": [_asset_view(controller, asset) for asset in assets],
    }


def _asset_view(controller, asset: dict) -> dict:
    job = _job(controller, asset.get("job_id"))
    view = present_job(job)
    artifact_id = asset.get("artifact_id")
    status = asset.get("status")
    if job and job.get("status") == "completed" and job.get("artifact_id"):
        artifact_id = job["artifact_id"]
        status = "completed"
    elif job and view["phase"] == "waiting_capacity":
        status = "waiting_capacity"
    elif job and job.get("status") == "failed":
        status = "failed"
    elif job:
        status = view["phase"]
    estimated, actual = _costs(controller, asset.get("job_id"))
    history = []
    for job_id in asset.get("history") or []:
        earlier = _job(controller, job_id)
        history.append({
            "job_id": job_id,
            "status": None if earlier is None else earlier.get("status"),
            "artifact_id": None if earlier is None else earlier.get("artifact_id"),
            "phase": present_job(earlier)["phase"],
        })
    return {
        "id": asset["id"],
        "project_id": asset["project_id"],
        "scene_id": asset["scene_id"],
        "type": asset["type"],
        "engine": asset["engine"],
        "model": asset["model"] if asset["type"] == "image" else None,
        "prompt": asset["prompt"],
        "status": status,
        "artifact_id": artifact_id,
        "job_id": asset.get("job_id"),
        "created_at": asset["created_at"],
        "updated_at": asset["updated_at"],
        "settings": asset["settings"],
        "history": history,
        "phase": view["phase"],
        "title": view["title"],
        "tone": view["tone"],
        "progress": view["progress"],
        "estimated_cost": estimated,
        "actual_cost": actual,
    }


def _job(controller, job_id: Optional[str]) -> Optional[dict]:
    if not job_id:
        return None
    try:
        job = controller.get_job(job_id)
    except JobNotFound:
        return None
    if job.public_status is not None:
        return job.to_image_dict()
    return {"job_id": job.id, "status": job.status.lower(), "error_code": job.error_code, "progress": 0}


def _costs(controller, job_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not job_id:
        return None, None
    event = controller.usage.for_job(job_id)
    if event is None:
        return None, None
    return event.estimated_cost_usd, event.actual_cost_usd


def _object(body: Optional[bytes]) -> dict:
    if not body:
        raise StudioError("BODY_REQUIRED")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StudioError("INVALID_JSON") from exc
    if not isinstance(payload, dict):
        raise StudioError("BODY_REQUIRED")
    return payload


def _has_settings(payload: dict) -> bool:
    return any(key in payload for key in ("width", "height", "seed"))


def _image_settings(payload: dict) -> dict:
    settings = {}
    if "width" in payload:
        settings["width"] = payload["width"]
    if "height" in payload:
        settings["height"] = payload["height"]
    if "seed" in payload:
        settings["seed"] = payload["seed"]
    return settings
