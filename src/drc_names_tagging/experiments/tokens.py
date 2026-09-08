from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import polars as pl
from tqdm import tqdm

from drc_names_tagging.config import (
    DEFAULT_CHECKPOINT_PATH,
    ExperimentConfig,
    ExperimentSettings,
)
from drc_names_tagging.dataset import Vocabulary
from drc_names_tagging.experiments.checkpoint import Checkpoint, signature
from drc_names_tagging.experiments.execution import execute
from drc_names_tagging.taggers import Registry, Tagger


@dataclass(frozen=True, slots=True)
class TokenRun:
    """Output and timing information for one unique-token annotation run."""

    tagger: str
    table: pl.DataFrame
    unique_tokens: int
    occurrences: int
    elapsed: float
    cached_tokens: int = 0
    processed_tokens: int = 0
    batch_attempts: int = 0
    checkpoint_identity: str = ""


class TokenRunner:
    """Annotate a precomputed vocabulary once per unique token."""

    def run(
        self,
        tagger: Tagger,
        dataset: pl.DataFrame,
        *,
        limit: int | None = None,
        sample_fraction: float = 1.0,
        label: str | None = None,
        cpu_workers: int = 1,
        concurrency: int = 1,
        batch_size: int = 1,
        retries: int = 2,
        checkpoint: Path = DEFAULT_CHECKPOINT_PATH,
        resume: bool = True,
        checkpoint_key: str | None = None,
    ) -> TokenRun:
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when configured")
        if min(cpu_workers, concurrency, batch_size) < 1 or retries < 0:
            raise ValueError(
                "Workers and batch size must be positive; retries cannot be negative"
            )
        if cpu_workers > 1 and concurrency > 1:
            raise ValueError("Select CPU workers or request concurrency, not both")
        # Sample the vocabulary after counting the entire corpus. Sorting and a
        # fixed seed give each annotator the same sample and retain full frequencies.
        words = dataset.sort("token_key")
        if sample_fraction < 1 and words.height:
            words = words.sample(
                n=max(1, int(words.height * sample_fraction)), seed=42, shuffle=True
            )
        if limit is not None:
            words = words.head(limit)
        run_name = label or tagger.name

        started = time.perf_counter()
        identity = signature(checkpoint_key or tagger.name)
        if not resume:
            # Fresh measurements still persist their labels, in a separate namespace.
            # Existing resumable annotations remain available and unchanged.
            identity = f"{identity}:{uuid4().hex}"
        store = Checkpoint(checkpoint, identity)
        processed = attempts = 0
        failures: list[tuple[str, str]] = []
        try:
            pending = words.join(store.load(), on=["token_key", "token"], how="anti")
            cached = words.height - pending.height
            with tqdm(
                total=words.height,
                initial=cached,
                desc=f"Tagging tokens with {run_name}",
                unit="token",
            ) as progress:
                for result in execute(
                    tagger,
                    pending.get_column("token").to_list(),
                    batch_size=batch_size,
                    cpu_workers=cpu_workers,
                    concurrency=concurrency,
                    retries=retries,
                ):
                    store.save(result.annotations, result.failures)
                    processed += len(result.annotations)
                    attempts += result.attempts
                    failures.extend(result.failures)
                    progress.update(len(result.annotations))
            if failures:
                raise RuntimeError(
                    f"{len(failures)} tokens failed after bounded retries. Checkpoint: {checkpoint}. "
                    f"First failure: {failures[0]}. Rerun to retry unfinished tokens."
                )
            output = (
                words.join(
                    store.load(),
                    on=["token_key", "token"],
                    how="left",
                    maintain_order="left",
                )
                .with_columns(pl.lit(run_name).alias("tagger"))
                .select("tagger", "token_key", "token", "frequency", "tag", "score")
            )
        finally:
            store.close()

        return TokenRun(
            tagger=run_name,
            table=output,
            unique_tokens=words.height,
            occurrences=int(words.get_column("frequency").sum() or 0),
            elapsed=time.perf_counter() - started,
            cached_tokens=cached,
            processed_tokens=processed,
            batch_attempts=attempts,
            checkpoint_identity=identity,
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
                "cached_tokens": run.cached_tokens,
                "processed_tokens": run.processed_tokens,
                "batch_attempts": run.batch_attempts,
                "checkpoint_identity": run.checkpoint_identity,
                "tokens_per_second": (
                    run.processed_tokens / run.elapsed if run.elapsed else 0.0
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
                right.table.select(["token_key", pl.col("tag").alias("right_tag")]),
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

    table = Vocabulary(settings.dataset_path).load()
    destination = (
        Path(output) if output else settings.token_dir / f"{experiment.name}.csv"
    )
    result = TokenRunner().run(
        Registry().create(experiment),
        table,
        limit=experiment.token_limit,
        sample_fraction=experiment.token_sample_fraction,
        label=experiment.name,
        cpu_workers=experiment.cpu_workers,
        concurrency=experiment.concurrency,
        batch_size=experiment.batch_size,
        retries=experiment.retries,
        checkpoint=settings.checkpoint_path,
        resume=experiment.resume,
        checkpoint_key=experiment.checkpoint_key or experiment.name,
    )
    save_table(result.table, destination)
    _, summary = compare_token_runs([result])
    destination.with_suffix(".json").write_text(
        json.dumps(
            {"experiment": experiment.to_dict(), "metrics": summary.to_dicts()[0]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
        raise ValueError(
            "Token comparison requires at least two configured experiments."
        )
    if reference_tagger not in {experiment.name for experiment in experiments}:
        raise ValueError(f"Reference experiment '{reference_tagger}' was not selected")
    table = Vocabulary(settings.dataset_path).load()
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
            cpu_workers=experiment.cpu_workers,
            concurrency=experiment.concurrency,
            batch_size=experiment.batch_size,
            retries=experiment.retries,
            checkpoint=settings.checkpoint_path,
            resume=experiment.resume,
            checkpoint_key=experiment.checkpoint_key or experiment.name,
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
