from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenClassification:
    token_normalized: str
    native_score: float
    surname_score: float
    predicted_component: str
    score_margin: float
