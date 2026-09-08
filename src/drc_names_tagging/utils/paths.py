from __future__ import annotations

from pathlib import Path


def get_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def dataset_path(stage: str, *parts: str | Path) -> Path:
    target = get_root() / "dataset" / stage
    if parts:
        target = target.joinpath(*[str(part) for part in parts])
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def data_path(*parts: str | Path) -> Path:
    """Return a versioned project-data path and create its parent directory."""

    target = get_root() / "data" / Path(*[str(part) for part in parts])
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def resolve_dataset(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    root = get_root()
    # Prefer the local copy, then fall back to the sibling corpus projects.
    candidates = (
        root / "data" / "dataset" / "names.csv",
        root / "dataset" / "gold" / "names.csv",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def assert_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    return path
