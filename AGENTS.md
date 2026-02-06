# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/drc_names_tagging/`, using a standard Python package layout.
- Domain and core logic are organized under `src/drc_names_tagging/domain/` and `src/drc_names_tagging/core/`.
- Workflows are in `src/drc_names_tagging/workflow/` (for example, `collect.py`).
- Project metadata and build configuration are in `pyproject.toml`; the entry point is the `drc-names-corpus` script.

## Build, Test, and Development Commands
- If you use uv (see `uv.lock`), `uv sync` installs dependencies and `uv run drc-names-corpus` runs the entry point.
- Run `uv run pyright` and `uv run ruff check .` to validate changes end-to-end.

## Coding Style & Naming Conventions
- Use Python 3.13 syntax and type hints when introducing new functions.
- Indentation: 4 spaces; line length: keep lines reasonably short (around 88–100 chars).
- Use `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Prefer small, focused functions; keep workflow steps in `workflow/` and reusable logic in `core/` or `domain/`.
