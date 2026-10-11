"""Multimodal registry and gates. Importing this package does not download weights."""
from media_engine.models.registry import ModelSpec, get_model, models_for, primary_for

__all__ = ["ModelSpec", "get_model", "models_for", "primary_for"]
