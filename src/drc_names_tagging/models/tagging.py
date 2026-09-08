from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Tag(StrEnum):
    """Labels assigned to name tokens."""

    NATIVE = "native"
    FOREIGN = "foreign"


@dataclass(frozen=True, slots=True)
class Token:
    """One position-preserving token annotation."""

    index: int
    text: str
    tag: Tag
    score: float | None = None


@dataclass(frozen=True, slots=True)
class Name:
    """A name and its ordered token annotations."""

    text: str
    tokens: tuple[Token, ...]


def tokens(name: str) -> list[str]:
    """Normalize whitespace without changing token order."""

    normalized = " ".join(name.strip().split())
    return normalized.split(" ") if normalized else []
