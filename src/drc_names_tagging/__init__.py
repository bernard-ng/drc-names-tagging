from __future__ import annotations

import logging

import typer

from drc_names_tagging.workflow.classify import classify
from drc_names_tagging.workflow.transitions import transitions

app = typer.Typer(no_args_is_help=True)


@app.command("transitions")
def transitions_command() -> None:
    """Generate Markov transitions."""
    transitions()


@app.command("classify")
def classify_command() -> None:
    """Classify each name component using Markov transitions."""
    classify()


def main() -> None:
    configure_logging()
    app()


__all__ = [
    "app",
    "main",
]


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s : %(message)s",
    )
