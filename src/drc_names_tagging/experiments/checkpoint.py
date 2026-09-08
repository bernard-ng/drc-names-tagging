from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from drc_names_tagging.models import Token

SCHEMA = {
    "token_key": pl.String,
    "token": pl.String,
    "tag": pl.String,
    "score": pl.Float64,
}


def signature(checkpoint_key: str) -> str:
    """Return the user-controlled namespace for persisted annotations."""

    key = checkpoint_key.strip()
    if not key:
        raise ValueError("checkpoint_key must not be empty")
    return key


class Checkpoint:
    """Commit completed batches while other workers continue computing."""

    def __init__(self, path: Path, identity: str) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
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
