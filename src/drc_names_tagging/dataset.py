from __future__ import annotations

from pathlib import Path

import polars as pl

from drc_names_tagging.utils.paths import assert_file, resolve_dataset


class DatasetSchemaError(ValueError):
    """Raised when names.csv does not contain the published name column."""


class NameDataset:
    """Load and normalize the shared names.csv dataset."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_dataset(path)

    def load(self) -> pl.DataFrame:
        table = pl.read_csv(assert_file(self.path))
        if "name" not in table.columns:
            raise DatasetSchemaError(f"Dataset '{self.path}' is missing the 'name' column.")
        table = table.with_row_index("row_index").with_columns(
            pl.col("name")
            .fill_null("")
            .cast(pl.String)
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
            .alias("name")
        )
        return table.filter(pl.col("name") != "")
