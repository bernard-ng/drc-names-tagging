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
    token_sample_fraction: float = 1.0
    token_limit: int | None = None
    cpu_workers: int = 1
    concurrency: int = 1
    batch_size: int = 1
    retries: int = 2
    resume: bool = True

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
        if not 0 < self.token_sample_fraction <= 1:
            raise ValueError("token_sample_fraction must be between zero and one")
        if self.token_limit is not None and self.token_limit <= 0:
            raise ValueError("token_limit must be positive when configured")
        for field_name in ("cpu_workers", "concurrency", "batch_size"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be a boolean")
        if self.cpu_workers > 1 and self.concurrency > 1:
            raise ValueError("Select CPU workers or request concurrency, not both")

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
