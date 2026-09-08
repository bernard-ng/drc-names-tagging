from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import polars as pl
from tqdm import tqdm

from drc_names_tagging.dataset import NameDataset
from drc_names_tagging.models import END, START, Matrix, Spec
from drc_names_tagging.utils.paths import data_path


class TransitionBuilder:
    """Build native and foreign character-transition matrices."""

    specs = (
        Spec("native", "probable_native", "native_transition.csv"),
        Spec("foreign", "probable_foreign", "foreign_transition.csv"),
    )

    def __init__(self, dataset: str | Path | None = None) -> None:
        self.dataset = dataset

    def run(self) -> list[Matrix]:
        table = NameDataset(self.dataset).load()
        results = []
        for spec in self.specs:
            names = self._names(table, spec.source)
            matrix = self._matrix(names)
            path = data_path(spec.filename)
            matrix.write_csv(path, float_precision=8)
            results.append(Matrix(spec, matrix, path))
        return results

    @staticmethod
    def _names(table: pl.DataFrame, column: str) -> list[str]:
        if column not in table.columns:
            raise KeyError(
                f"Expected '{column}' in the dataset. The Markov baseline needs "
                "probable_native and probable_foreign."
            )
        names: list[str] = []
        for value in tqdm(table.get_column(column), desc=f"Normalizing {column}", unit="name"):
            if value is not None:
                normalized = " ".join(str(value).strip().split()).lower()
                if normalized:
                    names.append(normalized)
        return names

    @staticmethod
    def _matrix(names: list[str]) -> pl.DataFrame:
        pairs = [
            pair
            for name in tqdm(names, desc="Building transition pairs", unit="name")
            for pair in pairwise(f"{START}{name}{END}")
        ]
        if not pairs:
            return pl.DataFrame(
                schema={
                    "from_letter": pl.String,
                    "to_letter": pl.String,
                    "count": pl.UInt64,
                    "probability": pl.Float64,
                }
            )
        counts = pl.DataFrame(pairs, schema=["from_letter", "to_letter"])
        return (
            counts.group_by(["from_letter", "to_letter"])
            .len()
            .rename({"len": "count"})
            .with_columns(
                (pl.col("count") / pl.col("count").sum().over("from_letter")).alias(
                    "probability"
                )
            )
            .sort(["from_letter", "to_letter"])
            .select(["from_letter", "to_letter", "count", "probability"])
        )
