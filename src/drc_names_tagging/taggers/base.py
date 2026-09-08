from __future__ import annotations

from typing import Protocol

from drc_names_tagging.models import Name


class Tagger(Protocol):
    """Minimal contract required by the experiment runner."""

    @property
    def name(self) -> str: ...

    def tag(self, name: str) -> Name: ...
