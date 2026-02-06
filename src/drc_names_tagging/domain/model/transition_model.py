from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

START_TOKEN = "^"
END_TOKEN = "$"


@dataclass(frozen=True)
class TransitionModel:
    component: str
    transitions: dict[tuple[str, str], float]
    epsilon: float = 1e-6

    @classmethod
    def from_csv(
        cls,
        component: str,
        path: Path,
        *,
        epsilon: float = 1e-6,
    ) -> TransitionModel:
        table = pl.read_csv(path)
        required = {"from_char", "to_char", "probability"}
        missing = required.difference(table.columns)
        if missing:
            missing_label = ", ".join(sorted(missing))
            raise KeyError(f"Missing required transition columns: {missing_label}.")

        transitions: dict[tuple[str, str], float] = {}
        for row in table.select(["from_char", "to_char", "probability"]).iter_rows():
            from_char, to_char, probability = row
            transitions[(str(from_char), str(to_char))] = float(probability)

        return cls(component=component, transitions=transitions, epsilon=epsilon)

    def log_likelihood(self, token: str) -> float:
        sequence = f"{START_TOKEN}{token}{END_TOKEN}"
        total = 0.0
        for from_char, to_char in zip(sequence, sequence[1:]):
            prob = self.transitions.get((from_char, to_char), self.epsilon)
            total += math.log(max(prob, self.epsilon))
        return total

    def average_log_likelihood(self, token: str) -> float:
        transitions = max(1, len(token) + 1)
        return self.log_likelihood(token) / transitions
