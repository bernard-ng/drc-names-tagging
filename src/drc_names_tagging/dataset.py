from __future__ import annotations

from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

import polars as pl

from drc_names_tagging.models import tokens
from drc_names_tagging.utils.paths import assert_file, resolve_dataset


class DatasetSchemaError(ValueError):
    """Raised when names.csv does not contain the published name column."""


class NameDataset:
    """Load and normalize the published name dataset."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_dataset(path)

    def load(self) -> pl.DataFrame:
        table = pl.read_csv(assert_file(self.path))
        if "name" not in table.columns:
            raise DatasetSchemaError(
                f"Dataset '{self.path}' is missing the 'name' column."
            )
        table = table.with_row_index("row_index").with_columns(
            pl.col("name")
            .fill_null("")
            .cast(pl.String)
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
            .alias("name")
        )
        return table.filter(pl.col("name") != "")


class TokenDataset:
    """Load reusable lexical annotations for later model training."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pl.DataFrame:
        table = pl.read_csv(assert_file(self.path))
        required = {"token", "tag"}
        missing = sorted(required.difference(table.columns))
        if missing:
            fields = ", ".join(missing)
            raise DatasetSchemaError(
                f"Token dataset '{self.path}' is missing required column(s): {fields}."
            )
        return table.filter(pl.col("token").fill_null("").cast(pl.String) != "")


class Vocabulary:
    """Cache the unique vocabulary beside its source dataset."""

    schema: ClassVar = {
        "token_key": pl.String,
        "token": pl.String,
        "frequency": pl.Int64,
    }

    def __init__(self, dataset: str | Path | None = None) -> None:
        self.source = resolve_dataset(dataset)
        self.path = self.source.with_name(f"{self.source.stem}_tokens.csv")

    @staticmethod
    def extract(dataset: pl.DataFrame) -> pl.DataFrame:
        """Count case-insensitive tokens while preserving their first spelling."""

        counts: Counter[str] = Counter()
        display: dict[str, str] = {}
        for name in dataset.get_column("name").to_list():
            for value in tokens(str(name)):
                key = value.casefold()
                counts[key] += 1
                display.setdefault(key, value)
        return pl.DataFrame(
            {
                "token_key": list(counts),
                "token": [display[key] for key in counts],
                "frequency": list(counts.values()),
            },
            schema=Vocabulary.schema,
        ).sort("token_key")

    def load(self, *, refresh: bool = False) -> pl.DataFrame:
        """Reuse the cache; extract only when missing or explicitly refreshed."""

        if self.path.exists() and not refresh:
            table = pl.read_csv(self.path, schema_overrides=self.schema)
            if not set(self.schema).issubset(table.columns):
                raise DatasetSchemaError(
                    f"Vocabulary '{self.path}' has invalid columns."
                )
            return table

        table = self.extract(NameDataset(self.source).load())
        # Publish a complete CSV so an interrupted extraction cannot leave a
        # partial cache that a later experiment would mistake for the vocabulary.
        with NamedTemporaryFile(
            dir=self.path.parent, suffix=".csv", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            table.write_csv(temporary)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return table
