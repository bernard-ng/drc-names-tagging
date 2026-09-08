from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drc_names_tagging.config.defaults import (
    DEFAULT_DATASET_PATH,
    DEFAULT_EXPERIMENT_OUTPUTS_DIR,
    DEFAULT_EXPERIMENT_TEMPLATES_PATH,
)


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """Shared dataset, template, and artifact locations."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    templates_path: Path = DEFAULT_EXPERIMENT_TEMPLATES_PATH
    outputs_dir: Path = DEFAULT_EXPERIMENT_OUTPUTS_DIR

    def __post_init__(self) -> None:
        for field_name in ("dataset_path", "templates_path", "outputs_dir"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))

    @property
    def tagging_dir(self) -> Path:
        return self.outputs_dir / "tagging"
