"""Configuration-driven native/foreign tagging experiments."""

from __future__ import annotations

from drc_names_tagging.dataset import DatasetSchemaError, NameDataset

__version__ = "0.2.0"


def main() -> None:
    """Launch the command-line interface."""

    from drc_names_tagging.cli import app

    app()


__all__ = ["DatasetSchemaError", "NameDataset", "main"]
