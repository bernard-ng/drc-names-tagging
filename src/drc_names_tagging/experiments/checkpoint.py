from __future__ import annotations

import hashlib
import inspect
import sqlite3
from pathlib import Path

import polars as pl

from drc_names_tagging.models import Token
from drc_names_tagging.taggers.base import Tagger

SCHEMA = {
    "token_key": pl.String,
    "token": pl.String,
    "tag": pl.String,
    "score": pl.Float64,
}


def signature(tagger: Tagger, batch_size: int) -> str:
    """Invalidate labels when model settings, matrices, or annotation code change."""

    digest = hashlib.sha256(f"v1:{tagger!r}:{batch_size}".encode())
    revision = getattr(tagger, "revision", None)
    if callable(revision):
        digest.update(str(revision()).encode())
    # Include the module (prompt/schema) and shared scoring/validation logic.
    paths = [Path(__file__).with_name("execution.py")]
    source = inspect.getsourcefile(type(tagger))
    if source:
        paths.append(Path(source))
    paths.extend((Path(__file__).parents[1] / "models").glob("*.py"))
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


class Checkpoint:
    """Commit completed batches while other workers continue computing."""

    def __init__(self, path: Path | None, identity: str) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path) if path else ":memory:")
        self.identity = identity
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS annotations ("
            "identity TEXT, token_key TEXT, token TEXT, tag TEXT, score REAL, "
            "PRIMARY KEY(identity, token_key, token))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS failures ("
            "identity TEXT, token TEXT, error TEXT, PRIMARY KEY(identity, token))"
        )
        self.connection.commit()

    def load(self) -> pl.DataFrame:
        rows = self.connection.execute(
            "SELECT token_key, token, tag, score FROM annotations WHERE identity = ?",
            (self.identity,),
        ).fetchall()
        return pl.DataFrame(rows, schema=SCHEMA, orient="row")

    def save(self, annotations: list[Token], failures: list[tuple[str, str]]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?, ?)",
                [
                    (self.identity, t.text.casefold(), t.text, t.tag.value, t.score)
                    for t in annotations
                ],
            )
            self.connection.executemany(
                "DELETE FROM failures WHERE identity = ? AND token = ?",
                [(self.identity, t.text) for t in annotations],
            )
            self.connection.executemany(
                "INSERT OR REPLACE INTO failures VALUES (?, ?, ?)",
                [(self.identity, token, error) for token, error in failures],
            )

    def close(self) -> None:
        self.connection.close()
