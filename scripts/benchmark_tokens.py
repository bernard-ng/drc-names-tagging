"""Small local throughput benchmark: uv run python scripts/benchmark_tokens.py.

OLLAMA_HOST may point to an isolated server configured for parallel requests.
Results are model agreement and structural validity measurements, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

from drc_names_tagging.config import ExperimentSettings
from drc_names_tagging.dataset import Vocabulary
from drc_names_tagging.experiments.builder import ExperimentBuilder
from drc_names_tagging.experiments.tokens import TokenRunner
from drc_names_tagging.taggers import Registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["markov", "ollama_mistral_7b"])
    parser.add_argument(
        "--batch-size", type=int, help="Restrict the benchmark to this batch size."
    )
    arguments = parser.parse_args()
    selected = arguments.model
    settings = ExperimentSettings()
    builder = ExperimentBuilder(settings)
    words = Vocabulary(settings.dataset_path).load()
    directory = settings.outputs_dir / "benchmarks" / str(time.time_ns())
    directory.mkdir(parents=True)
    results: list[dict[str, object]] = []
    for model, count, grid in (
        ("markov", words.height, [(1, 1, 4096), (10, 1, 4096)]),
        (
            "ollama_mistral_7b",
            32,
            [
                (1, 1, 1),
                (1, 2, 1),
                (1, 4, 1),
                (1, 1, 8),
                (1, 2, 8),
                (1, 4, 8),
                (1, 2, 16),
            ],
        ),
    ):
        if selected and selected != model:
            continue
        experiment = builder.build(model)
        params = dict(experiment.tagger_params)
        if model.startswith("ollama") and "OLLAMA_HOST" in os.environ:
            params["ollama_url"] = os.environ["OLLAMA_HOST"]
        tagger = Registry().create(replace(experiment, tagger_params=params))
        sample = words.sample(n=min(count, words.height), seed=42, shuffle=True)
        # Warm model loading is excluded from timing for each implementation.
        tagger.tag("bope")
        baseline: list[str] | None = None
        for workers, concurrency, batch in grid:
            if arguments.batch_size is not None and arguments.batch_size != batch:
                continue
            label = f"{model}-w{workers}-c{concurrency}-b{batch}"
            started = time.perf_counter()
            row: dict[str, object] = {
                "experiment": label,
                "tokens": sample.height,
                "cpu_workers": workers,
                "concurrency": concurrency,
                "batch_size": batch,
            }
            try:
                run = TokenRunner().run(
                    tagger,
                    sample,
                    cpu_workers=workers,
                    concurrency=concurrency,
                    batch_size=batch,
                    retries=1,
                    label=label,
                    checkpoint=settings.checkpoint_path,
                    resume=False,
                )
                tags = run.table["tag"].to_list()
                baseline = tags if baseline is None else baseline
                row.update(
                    elapsed_seconds=run.elapsed,
                    tokens_per_second=run.processed_tokens / run.elapsed,
                    batch_attempts=run.batch_attempts,
                    checkpoint_identity=run.checkpoint_identity,
                    agreement_with_serial=sum(
                        a == b for a, b in zip(tags, baseline, strict=True)
                    )
                    / len(tags),
                    error=None,
                )
                run.table.write_csv(directory / f"{label}.csv")
            except (ValueError, RuntimeError) as error:
                row.update(
                    elapsed_seconds=time.perf_counter() - started, error=str(error)
                )
            results.append(row)
            print(json.dumps(row), flush=True)
            (directory / "summary.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
    print(f"Saved benchmark to {directory}")


if __name__ == "__main__":
    main()
