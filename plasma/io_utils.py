from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import shutil


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_directory(path: str | Path) -> Path:
    """Remove generated contents while preserving the directory itself."""
    path = Path(path)
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_audio_files(
    data_dir: str | Path,
    extensions: Iterable[str] = (".wav", ".flac"),
    recursive: bool = False,
) -> List[Path]:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {data_dir.resolve()}"
        )

    normalized_extensions = {
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in extensions
    }

    iterator = data_dir.rglob("*") if recursive else data_dir.iterdir()
    files = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )

    if not files:
        raise FileNotFoundError(
            f"No supported audio files found in: {data_dir.resolve()}"
        )

    return files


def relative_recording_id(file_path: str | Path, dataset_root: str | Path) -> str:
    """Return a stable POSIX-style recording identifier relative to a dataset root."""
    return Path(file_path).resolve().relative_to(Path(dataset_root).resolve()).as_posix()
