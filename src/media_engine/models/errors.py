"""Adapter failures. Importing this module does not load a model."""
from __future__ import annotations


class AdapterError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
