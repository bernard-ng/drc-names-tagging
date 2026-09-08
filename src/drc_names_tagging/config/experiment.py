from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Definition of one comparable token-tagging experiment."""

    name: str
    tagger_type: str = "ollama"
    description: str = ""
    tags: tuple[str, ...] = ()
    tagger_params: dict[str, Any] = field(default_factory=dict)
    sample_fraction: float = 0.01
    limit: int | None = 1_000

    def __post_init__(self) -> None:
        name = self.name.strip()
        tagger_type = self.tagger_type.strip().lower()
        if not name:
            raise ValueError("Experiment name must not be empty")
        if not tagger_type:
            raise ValueError("Experiment tagger_type must not be empty")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when configured")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tagger_type", tagger_type)
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "tagger_params", dict(self.tagger_params))

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["tags"] = list(self.tags)
        return values

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        values = dict(data)
        if "tags" in values:
            values["tags"] = tuple(values["tags"])
        return cls(**values)
