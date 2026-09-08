from __future__ import annotations

from drc_names_tagging.taggers.base import Tagger
from drc_names_tagging.taggers.markov import Markov
from drc_names_tagging.taggers.ollama import Ollama
from drc_names_tagging.taggers.registry import Registry

__all__ = ["Markov", "Ollama", "Registry", "Tagger"]
