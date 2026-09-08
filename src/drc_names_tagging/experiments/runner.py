from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import polars as pl
from tqdm import tqdm

from drc_names_tagging.config import ExperimentConfig, ExperimentSettings
from drc_names_tagging.dataset import NameDataset
from drc_names_tagging.models import Name
from drc_names_tagging.taggers import Registry, Tagger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Run:
    """Output and timing information for one tagger run."""

    tagger: str
    table: pl.DataFrame
    unique_names: int
    elapsed: float


class Runner:
    """Apply a tagger to one shared dataset."""

    def run(
        self,
        tagger: Tagger,
        dataset: pl.DataFrame,
        *,
        limit: int | None = None,
        sample_fraction: float = 1.0,
    ) -> Run:
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        selected = dataset.head(limit) if limit is not None else dataset
        if sample_fraction < 1:
            selected = selected.head(max(1, int(selected.height * sample_fraction)))
        names = selected.get_column("name").unique(maintain_order=True).to_list()

        # Repeated names are common; caching here avoids paying for duplicate
        # Ollama calls while retaining every source row in the output.
        started = time.perf_counter()
        tagged = {
            name: tagger.tag(name)
            for name in tqdm(names, desc=f"Tagging with {tagger.name}", unit="name")
        }
        output = self._flatten(selected, tagged, tagger.name)
        return Run(
            tagger=tagger.name,
            table=output,
            unique_names=len(names),
            elapsed=time.perf_counter() - started,
        )

    @staticmethod
    def _flatten(
        dataset: pl.DataFrame,
        tagged: dict[str, Name],
        tagger: str,
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for row in dataset.iter_rows(named=True):
            for token in tagged[str(row["name"])].tokens:
                rows.append(
                    {
                        "tagger": tagger,
                        "row_index": row["row_index"],
                        "id": row.get("id"),
                        "name": row["name"],
                        "token_index": token.index,
                        "component": token.text,
                        "tag": token.tag.value,
                        "score": token.score,
                    }
                )
        return pl.DataFrame(rows)


def compare_runs(runs: Iterable[Run]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return pairwise token agreement and throughput summaries."""

    completed = list(runs)
    if not completed:
        return pl.DataFrame(), pl.DataFrame()
    summary = pl.DataFrame(
        [
            {
                "tagger": run.tagger,
                "tokens": run.table.height,
                "unique_names": run.unique_names,
                "elapsed_seconds": run.elapsed,
                "tokens_per_second": run.table.height / run.elapsed if run.elapsed else 0.0,
            }
            for run in completed
        ]
    )
    if len(completed) < 2:
        return pl.DataFrame(), summary

    pairs: list[pl.DataFrame] = []
    for left, right in combinations(completed, 2):
        # Source row and token position keep comparisons aligned across taggers.
        pairs.append(
            left.table.select(
                ["row_index", "token_index", pl.col("tag").alias("left_tag")]
            )
            .join(
                right.table.select(
                    ["row_index", "token_index", pl.col("tag").alias("right_tag")]
                ),
                on=["row_index", "token_index"],
                how="inner",
            )
            .with_columns(
                [
                    pl.lit(left.tagger).alias("left_tagger"),
                    pl.lit(right.tagger).alias("right_tagger"),
                    (pl.col("left_tag") == pl.col("right_tag")).alias("agrees"),
                ]
            )
            .select(
                [
                    "left_tagger",
                    "right_tagger",
                    "row_index",
                    "token_index",
                    "left_tag",
                    "right_tag",
                    "agrees",
                ]
            )
        )
    return pl.concat(pairs), summary


def save_table(table: pl.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(destination, float_precision=8)
    return destination


def run_one(
    experiment: ExperimentConfig,
    settings: ExperimentSettings,
    *,
    output: str | Path | None = None,
) -> Path:
    """Run one configured experiment and persist its token table."""

    table = NameDataset(settings.dataset_path).load()
    result = Runner().run(
        Registry().create(experiment),
        table,
        limit=experiment.limit,
        sample_fraction=experiment.sample_fraction,
    )
    destination = Path(output) if output else settings.tagging_dir / f"{experiment.name}.csv"
    save_table(result.table, destination)
    logger.info(
        "Saved %d token tags from %d unique names to %s",
        result.table.height,
        result.unique_names,
        destination,
    )
    return destination


def compare(
    experiments: list[ExperimentConfig],
    settings: ExperimentSettings,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Run configured experiments on shared rows and persist agreement metrics."""

    if len(experiments) < 2:
        raise ValueError("Compare requires at least two configured experiments.")
    table = NameDataset(settings.dataset_path).load()
    runner = Runner()
    results = []
    destination = Path(output_dir) if output_dir else settings.tagging_dir
    registry = Registry()
    for experiment in experiments:
        result = runner.run(
            registry.create(experiment),
            table,
            limit=experiment.limit,
            sample_fraction=experiment.sample_fraction,
        )
        results.append(result)
        save_table(result.table, destination / f"{experiment.name}.csv")

    agreement, summary = compare_runs(results)
    agreement_path = save_table(agreement, destination / "comparison.csv")
    agreement_rate = None
    if "agrees" in agreement.columns and agreement.height:
        agrees = agreement.get_column("agrees").to_list()
        agreement_rate = sum(bool(value) for value in agrees) / len(agrees)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(
        json.dumps(
            {
                "experiments": [experiment.to_dict() for experiment in experiments],
                "taggers": summary.to_dicts(),
                "agreement_rate": agreement_rate,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Saved tagging comparison to %s", agreement_path)
    return agreement_path
