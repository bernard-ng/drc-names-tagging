from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drc_names_tagging.config import ExperimentConfig, ExperimentSettings

_SECTIONS = {
    "baseline": "baseline_experiments",
    "advanced": "advanced_experiments",
}


class ExperimentBuilder:
    """Load and validate comparable tagger definitions."""

    def __init__(self, settings: ExperimentSettings) -> None:
        self.settings = settings

    def load_templates(self, templates: str | Path | None = None) -> dict[str, Any]:
        path = (
            Path(templates) if templates is not None else self.settings.templates_path
        )
        if not path.is_absolute() and not path.is_file():
            path = self.settings.templates_path.parent / path
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"Template file must contain a mapping: {path}")
        return loaded

    def templates(self, experiment_type: str = "baseline") -> list[dict[str, Any]]:
        section_name = _SECTIONS.get(experiment_type)
        if section_name is None:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        values = self.load_templates().get(section_name, [])
        if not isinstance(values, list):
            raise TypeError(f"Template section '{section_name}' must be a list")
        return values

    def token_reference(self) -> str:
        """Return the configured preferred annotator for token training data."""

        workflow = self.load_templates().get("token_workflow", {})
        if not isinstance(workflow, dict):
            raise TypeError("Template section 'token_workflow' must be a mapping")
        reference = workflow.get("reference_tagger", "ollama_mistral_7b")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                "token_workflow.reference_tagger must be a non-empty string"
            )
        return reference.strip()

    def build(
        self,
        name: str,
        *,
        experiment_type: str = "baseline",
        sample_fraction: float | None = None,
        limit: int | None = None,
    ) -> ExperimentConfig:
        template = self.find_template(self.load_templates(), name, experiment_type)
        values = dict(template)
        if sample_fraction is not None:
            values["sample_fraction"] = sample_fraction
        if limit is not None:
            values["limit"] = limit
        return ExperimentConfig.from_dict(values)

    @staticmethod
    def find_template(
        templates: dict[str, Any],
        name: str,
        experiment_type: str = "baseline",
    ) -> dict[str, Any]:
        section_name = _SECTIONS.get(experiment_type)
        if section_name is None:
            available = ", ".join(_SECTIONS)
            raise ValueError(
                f"Unknown experiment type '{experiment_type}'. Available: {available}"
            )
        experiments = templates.get(section_name, [])
        for experiment in experiments:
            if experiment.get("name") == name:
                return experiment
        available = [experiment.get("name", "unknown") for experiment in experiments]
        raise ValueError(f"Experiment '{name}' not found. Available: {available}")
