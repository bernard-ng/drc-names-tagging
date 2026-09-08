from __future__ import annotations

from typing import Protocol, runtime_checkable

from drc_names_tagging.models import Name, Token


class Tagger(Protocol):
    """Minimal contract required by the experiment runner."""

    @property
    def name(self) -> str: ...

    def tag(self, name: str) -> Name: ...


@runtime_checkable
class BatchTagger(Protocol):
    """Optional efficient annotation of independent lexical tokens."""

    def tag_batch(self, source: list[str]) -> tuple[Token, ...]: ...
