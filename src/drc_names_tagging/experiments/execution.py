from __future__ import annotations

import math
import multiprocessing
from collections.abc import Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from itertools import batched

from drc_names_tagging.models import Token
from drc_names_tagging.taggers.base import BatchTagger, Tagger


@dataclass
class Outcome:
    annotations: list[Token] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    attempts: int = 0


def annotate(tagger: Tagger, source: list[str], retries: int) -> Outcome:
    """Validate a whole batch before accepting it; retry failures a bounded number of times."""

    result = Outcome()
    message = "Annotation failed"
    for _ in range(retries + 1):
        result.attempts += 1
        try:
            if isinstance(tagger, BatchTagger):
                annotations = tagger.tag_batch(source)
            else:
                values: list[Token] = []
                for text in source:
                    name = tagger.tag(text)
                    if len(name.tokens) != 1:
                        raise ValueError(f"Expected one annotation for {text!r}")
                    values.append(name.tokens[0])
                annotations = tuple(values)
            if len(annotations) != len(source):
                raise ValueError("Annotation count does not match batch size")
            for text, token in zip(source, annotations, strict=True):
                if token.text != text:
                    raise ValueError(f"Expected {text!r}, received {token.text!r}")
                if token.score is not None and not math.isfinite(token.score):
                    raise ValueError(f"Non-finite score for {text!r}")
            result.annotations.extend(annotations)
            return result
        except (RuntimeError, OSError, ValueError, TypeError) as error:
            message = str(error)
    result.failures = [(text, message) for text in source]
    return result


_worker: Tagger | None = None


def initialize(tagger: Tagger) -> None:
    global _worker
    _worker = tagger


def process(source: list[str], retries: int) -> Outcome:
    if _worker is None:
        raise RuntimeError("CPU worker was not initialized")
    return annotate(_worker, source, retries)


def execute(
    tagger: Tagger,
    source: Iterable[str],
    *,
    batch_size: int,
    cpu_workers: int,
    concurrency: int,
    retries: int,
) -> Iterator[Outcome]:
    """Keep a bounded number of batches in flight while the caller saves results."""

    batches = (list(batch) for batch in batched(source, batch_size))
    workers = max(cpu_workers, concurrency)
    if workers == 1:
        for batch in batches:
            result = annotate(tagger, batch, retries)
            yield result
            if result.failures:
                return
        return

    pool: Executor
    if cpu_workers > 1:
        # Spawn avoids forking Polars' thread pool; initialize the model once per process.
        pool = ProcessPoolExecutor(
            max_workers=cpu_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(tagger,),
        )
    else:
        pool = ThreadPoolExecutor(max_workers=concurrency)

    def submit(batch: list[str]) -> Future[Outcome]:
        if cpu_workers > 1:
            return pool.submit(process, batch, retries)
        return pool.submit(annotate, tagger, batch, retries)

    pending: set[Future[Outcome]] = set()
    failed = False
    try:
        for batch in batches:
            pending.add(submit(batch))
            if len(pending) == workers:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                failed |= bool(result.failures)
                yield result
            # On failure, drain already running work so its successful labels are saved.
            if not failed:
                for _ in done:
                    batch = next(batches, None)
                    if batch is not None:
                        pending.add(submit(batch))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
