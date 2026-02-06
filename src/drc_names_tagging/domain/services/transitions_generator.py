from __future__ import annotations

from pathlib import Path

import polars as pl
from tqdm import tqdm

from drc_names_tagging.core import assert_file_exists
from drc_names_tagging.core import get_dataset_path
from drc_names_tagging.domain.model import TransitionMatrix
from drc_names_tagging.domain.model import TransitionMatrixSpec

START_TOKEN = "^"
END_TOKEN = "$"


class TransitionMatrixGenerator:
    def __init__(self, dataset_filename: str = "names.csv") -> None:
        self._dataset_filename = dataset_filename

    def create_transition_matrices(self) -> list[TransitionMatrix]:
        dataset = self._load_names_dataset()
        specs = [
            TransitionMatrixSpec(
                component="probable_native",
                source_column="probable_native",
                output_filename="native_transition.csv",
            ),
            TransitionMatrixSpec(
                component="probable_surname",
                source_column="probable_surname",
                output_filename="surname_transition.csv",
            ),
        ]

        results: list[TransitionMatrix] = []
        for spec in specs:
            names = self._extract_component_names(dataset, spec.source_column)
            table = self._build_transition_matrix(names)
            output_path = get_dataset_path("sliver", spec.output_filename)
            self._save_transition_matrix(table, output_path)
            results.append(
                TransitionMatrix(spec=spec, table=table, output_path=output_path)
            )

        return results

    def _load_names_dataset(self) -> pl.DataFrame:
        dataset_path = assert_file_exists(
            get_dataset_path("bronze", self._dataset_filename)
        )
        return pl.read_csv(dataset_path)

    def _extract_component_names(self, dataset: pl.DataFrame, column: str) -> list[str]:
        if column not in dataset.columns:
            raise KeyError(f"Expected column '{column}' to exist in dataset.")

        series = dataset.get_column(column)
        names: list[str] = []
        for value in tqdm(
            series,
            desc=f"Normalizing {column} values",
            unit="name",
        ):
            if value is None:
                continue
            normalized = self._normalize_name_component(str(value))
            if normalized:
                names.append(normalized)
        return names

    def _normalize_name_component(self, value: str) -> str:
        collapsed = " ".join(value.strip().split())
        return collapsed.lower()

    def _build_transition_matrix(self, names: list[str]) -> pl.DataFrame:
        pairs = self._build_transition_pairs(names)
        if not pairs:
            return pl.DataFrame(
                schema={
                    "from_char": pl.String,
                    "to_char": pl.String,
                    "count": pl.UInt64,
                    "probability": pl.Float64,
                }
            )

        pairs_df = pl.DataFrame(pairs, schema=["from_char", "to_char"])
        counts = pairs_df.group_by(["from_char", "to_char"]).len().rename(
            {"len": "count"}
        )
        return (
            counts.with_columns(
                (pl.col("count") / pl.col("count").sum().over("from_char")).alias(
                    "probability"
                )
            )
            .sort(["from_char", "to_char"])
            .select(["from_char", "to_char", "count", "probability"])
        )

    def _build_transition_pairs(self, names: list[str]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for name in tqdm(names, desc="Building transition pairs", unit="name"):
            sequence = f"{START_TOKEN}{name}{END_TOKEN}"
            pairs.extend(zip(sequence, sequence[1:]))
        return pairs

    def _save_transition_matrix(self, table: pl.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(output_path, float_precision=8)


def create_transition_matrices() -> list[TransitionMatrix]:
    return TransitionMatrixGenerator().create_transition_matrices()
