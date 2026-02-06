from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class TransitionMatrixSpec:
    component: str
    source_column: str
    output_filename: str


@dataclass(frozen=True)
class TransitionMatrix:
    spec: TransitionMatrixSpec
    table: pl.DataFrame
    output_path: Path

    @property
    def row_count(self) -> int:
        return self.table.height

    @property
    def total_transitions(self) -> int:
        if "count" not in self.table.columns or self.table.is_empty():
            return 0
        return int(self.table.get_column("count").sum())

    def summary(self) -> str:
        return (
            f"{self.spec.component} -> {self.output_path} "
            f"(rows={self.row_count}, transitions={self.total_transitions})"
        )
