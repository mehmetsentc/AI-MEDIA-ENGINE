"""Admin snapshot. Credentials are not part of the payload."""
from __future__ import annotations

_SECRET_KEYS = {"api_key", "authorization", "secret", "token", "database_url"}


def snapshot_for(manager, *, cache=None, worker=None, queue_length: int = 0) -> dict:
    body = manager.snapshot() if manager is not None else _empty_manager()
    cache_health = cache.health() if cache is not None else {"state": "UNKNOWN", "problem": "NOT_ATTACHED"}
    worker_status = worker.status() if worker is not None else {
        "state": "STOPPED", "model_loaded": False, "revision": "", "memory_mib": 0, "queue": 0,
    }
    payload = {
        "paid_actions_enabled": False,
        "instances": [] if not body.get("instance_id") else [{
            "id": body["instance_id"],
            "machine_id": body["machine_id"],
            "offer_id": body["offer_id"],
            "status": body["state"],
            "hourly_usd": body["hourly_usd"],
        }],
        "gpu_status": body["state"],
        "loaded_models": [worker_status["revision"]] if worker_status.get("model_loaded") else [],
        "model_loaded": bool(worker_status.get("model_loaded")),
        "memory_mib": worker_status.get("memory_mib", 0),
        "cache": {key: value for key, value in cache_health.items() if key != "path"},
        "queue_length": queue_length,
        "hourly_usd": body["hourly_usd"],
        "estimated_daily_usd": body["estimated_daily_usd"],
        "estimated_monthly_usd": body["estimated_monthly_usd"],
        "session_remaining_seconds": body.get("session_remaining_seconds", 0),
        "shutdown": "automatic" if body["state"] in {"IDLE", "STOPPING", "STOPPED"} else "armed",
        "alerts": [_public_alert(item) for item in body.get("alerts", [])],
        "candidates": body.get("candidates", 0),
        "poll_backoff_seconds": body.get("poll_backoff_seconds", 0),
    }
    return _redact(payload)


def _empty_manager() -> dict:
    return {
        "state": "STOPPED",
        "instance_id": "",
        "machine_id": 0,
        "offer_id": 0,
        "hourly_usd": "0",
        "estimated_daily_usd": "0",
        "estimated_monthly_usd": "0",
        "alerts": [],
        "candidates": 0,
        "poll_backoff_seconds": 0,
        "paid_actions_enabled": False,
    }


def _public_alert(item: dict) -> dict:
    return {key: value for key, value in item.items() if key.lower() not in _SECRET_KEYS}


def _redact(value):
    if isinstance(value, dict):
        return {
            key: _redact(item)
            for key, item in value.items()
            if key.lower() not in _SECRET_KEYS
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
