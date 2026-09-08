from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import polars as pl

START = "^"
END = "$"


@dataclass(frozen=True, slots=True)
class Spec:
    """Input and output metadata for one transition matrix."""

    tag: str
    source: str
    filename: str


@dataclass(frozen=True, slots=True)
class Matrix:
    spec: Spec
    table: pl.DataFrame
    path: Path

    def summary(self) -> str:
        count = int(self.table.get_column("count").sum()) if self.table.height else 0
        return f"{self.spec.tag} -> {self.path} (rows={self.table.height}, transitions={count})"


@dataclass(frozen=True, slots=True)
class Model:
    """Character transition probabilities used by the Markov tagger."""

    tag: str
    transitions: dict[tuple[str, str], float]
    epsilon: float = 1e-6

    @classmethod
    def from_csv(cls, tag: str, path: Path, *, epsilon: float = 1e-6) -> Model:
        table = pl.read_csv(path)
        if {"from_letter", "to_letter", "probability"}.issubset(table.columns):
            source, target = "from_letter", "to_letter"
        elif {"from_char", "to_char", "probability"}.issubset(table.columns):
            source, target = "from_char", "to_char"
        else:
            raise KeyError(
                "Transition CSV must contain from_letter/to_letter/probability columns."
            )

        values = {
            (str(source), str(target)): float(probability)
            for source, target, probability in table.select(
                [source, target, "probability"]
            ).iter_rows()
        }
        return cls(tag=tag, transitions=values, epsilon=epsilon)

    def average_log_likelihood(self, token: str) -> float:
        sequence = f"{START}{token}{END}"
        total = sum(
            math.log(max(self.transitions.get(pair, self.epsilon), self.epsilon))
            for pair in pairwise(sequence)
        )
        return total / max(1, len(token) + 1)
