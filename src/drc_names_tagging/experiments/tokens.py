from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import polars as pl
from tqdm import tqdm

from drc_names_tagging.config import ExperimentConfig, ExperimentSettings
from drc_names_tagging.dataset import NameDataset
from drc_names_tagging.models import tokens
from drc_names_tagging.taggers import Registry, Tagger


@dataclass(frozen=True, slots=True)
class TokenRun:
    """Output and timing information for one unique-token annotation run."""

    tagger: str
    table: pl.DataFrame
    unique_tokens: int
    occurrences: int
    elapsed: float


def vocabulary(dataset: pl.DataFrame) -> pl.DataFrame:
    """Extract one case-insensitive row per token and retain its corpus frequency."""

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
            "frequency": [counts[key] for key in counts],
        }
    )


class TokenRunner:
    """Annotate each unique lexical token once, independently of name position."""

    def run(
        self,
        tagger: Tagger,
        dataset: pl.DataFrame,
        *,
        limit: int | None = None,
        sample_fraction: float = 1.0,
        label: str | None = None,
    ) -> TokenRun:
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        selected = dataset.head(limit) if limit is not None else dataset
        if sample_fraction < 1:
            selected = selected.head(max(1, int(selected.height * sample_fraction)))
        words = vocabulary(selected)
        run_name = label or tagger.name

        started = time.perf_counter()
        rows: list[dict[str, object]] = []
        for row in tqdm(
            words.iter_rows(named=True),
            total=words.height,
            desc=f"Tagging tokens with {run_name}",
            unit="token",
        ):
            token = str(row["token"])
            result = tagger.tag(token)
            if len(result.tokens) != 1:
                raise ValueError(
                    f"Tagger '{tagger.name}' returned {len(result.tokens)} components "
                    f"for the isolated token '{token}'."
                )
            annotation = result.tokens[0]
            if annotation.text.casefold() != str(row["token_key"]):
                raise ValueError(
                    f"Tagger '{tagger.name}' changed token '{token}' to "
                    f"'{annotation.text}'."
                )
            rows.append(
                {
                    "tagger": run_name,
                    "token_key": row["token_key"],
                    "token": row["token"],
                    "frequency": row["frequency"],
                    "tag": annotation.tag.value,
                    "score": annotation.score,
                }
            )

        return TokenRun(
            tagger=run_name,
            table=pl.DataFrame(rows),
            unique_tokens=words.height,
            occurrences=int(words.get_column("frequency").sum() or 0),
            elapsed=time.perf_counter() - started,
        )


def compare_token_runs(runs: Iterable[TokenRun]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare annotators on the same unique token rather than token position."""

    completed = list(runs)
    if not completed:
        return pl.DataFrame(), pl.DataFrame()
    summary = pl.DataFrame(
        [
            {
                "tagger": run.tagger,
                "unique_tokens": run.unique_tokens,
                "occurrences": run.occurrences,
                "elapsed_seconds": run.elapsed,
                "tokens_per_second": (
                    run.unique_tokens / run.elapsed if run.elapsed else 0.0
                ),
            }
            for run in completed
        ]
    )
    if len(completed) < 2:
        return pl.DataFrame(), summary

    pairs: list[pl.DataFrame] = []
    for left, right in combinations(completed, 2):
        pairs.append(
            left.table.select(
                ["token_key", "token", "frequency", pl.col("tag").alias("left_tag")]
            )
            .join(
                right.table.select(
                    ["token_key", pl.col("tag").alias("right_tag")]
                ),
                on="token_key",
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
                    "token_key",
                    "token",
                    "frequency",
                    "left_tag",
                    "right_tag",
                    "agrees",
                ]
            )
        )
    return pl.concat(pairs), summary


def training_table(run: TokenRun) -> pl.DataFrame:
    """Return the preferred lexical labels in a reusable training-data format."""

    return run.table.select(
        ["token_key", "token", "frequency", "tag", "score"]
    ).with_columns(pl.lit(run.tagger).alias("source_tagger"))


def run_tokens(
    experiment: ExperimentConfig,
    settings: ExperimentSettings,
    *,
    output: str | Path | None = None,
) -> Path:
    """Run one token experiment and persist its lexical annotation table."""

    table = NameDataset(settings.dataset_path).load()
    result = TokenRunner().run(
        Registry().create(experiment),
        table,
        limit=experiment.token_limit,
        sample_fraction=experiment.token_sample_fraction,
        label=experiment.name,
    )
    destination = Path(output) if output else settings.token_dir / f"{experiment.name}.csv"
    save_table(result.table, destination)
    return destination


def compare_tokens(
    experiments: list[ExperimentConfig],
    settings: ExperimentSettings,
    *,
    reference_tagger: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Compare token annotators and save preferred labels for later modelling."""

    if len(experiments) < 2:
        raise ValueError("Token comparison requires at least two configured experiments.")
    table = NameDataset(settings.dataset_path).load()
    runner = TokenRunner()
    results: list[TokenRun] = []
    destination = Path(output_dir) if output_dir else settings.token_dir
    registry = Registry()
    for experiment in experiments:
        result = runner.run(
            registry.create(experiment),
            table,
            limit=experiment.token_limit,
            sample_fraction=experiment.token_sample_fraction,
            label=experiment.name,
        )
        results.append(result)
        save_table(result.table, destination / f"{experiment.name}.csv")

    reference = next(
        (result for result in results if result.tagger == reference_tagger), None
    )
    if reference is None:
        available = ", ".join(result.tagger for result in results)
        raise ValueError(
            f"Reference tagger '{reference_tagger}' was not selected. Available: {available}"
        )

    agreement, summary = compare_token_runs(results)
    agreement_path = save_table(agreement, destination / "comparison.csv")
    training_path = save_table(training_table(reference), destination / "training.csv")
    agreement_rate = None
    if "agrees" in agreement.columns and agreement.height:
        agrees = agreement.get_column("agrees").to_list()
        agreement_rate = sum(bool(value) for value in agrees) / len(agrees)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(
        json.dumps(
            {
                "experiments": [experiment.to_dict() for experiment in experiments],
                "reference_tagger": reference_tagger,
                "training_data": str(training_path),
                "taggers": summary.to_dicts(),
                "agreement_rate": agreement_rate,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return agreement_path


def save_table(table: pl.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(destination, float_precision=8)
    return destination
