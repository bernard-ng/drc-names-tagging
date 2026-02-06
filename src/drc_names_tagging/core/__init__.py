from __future__ import annotations

from pathlib import Path


def get_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def get_dataset_path(stage: str, *parts: str | Path) -> Path:
    base = get_root() / "dataset" / stage
    target = base.joinpath(*[str(part) for part in parts]) if parts else base
    dir_target = target if parts and not target.suffix else target.parent
    dir_target.mkdir(parents=True, exist_ok=True)
    return target


def get_report_path(*parts: str | Path) -> Path:
    target = get_root() / "reports"
    if parts:
        target = target.joinpath(*[str(part) for part in parts])
    dir_target = target if parts and not target.suffix else target.parent
    dir_target.mkdir(parents=True, exist_ok=True)
    return target


def assert_file_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    return path


class DirectoryNotFoundError(FileNotFoundError):
    """Raised when a required directory is missing."""


def assert_dir_exists(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise DirectoryNotFoundError(f"The directory '{path}' does not exist.")
    return path