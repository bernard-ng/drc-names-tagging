from __future__ import annotations

from pathlib import Path
from typing import Any

from drc_names_tagging.config import ExperimentConfig
from drc_names_tagging.taggers.base import Tagger
from drc_names_tagging.taggers.markov import Markov
from drc_names_tagging.taggers.ollama import Ollama


class Registry:
    """Construct taggers from experiment templates."""

    def create(self, experiment: ExperimentConfig) -> Tagger:
        params: dict[str, Any] = experiment.tagger_params
        if experiment.tagger_type == "markov":
            native = params.get("native_transition", "data/native_transition.csv")
            foreign = params.get("foreign_transition", "data/foreign_transition.csv")
            return Markov.from_paths(Path(str(native)), Path(str(foreign)))
        if experiment.tagger_type == "ollama":
            return Ollama(
                model=str(params.get("ollama_model", "mistral:7b")),
                url=str(params.get("ollama_url", "http://localhost:11434")),
                timeout=float(params.get("ollama_timeout", 120.0)),
            )
        raise ValueError(f"Unknown tagger type '{experiment.tagger_type}'.")
