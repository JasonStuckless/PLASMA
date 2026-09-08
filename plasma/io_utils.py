from pathlib import Path
from typing import List


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_audio_files(
    data_dir: str | Path,
    extensions: tuple[str, ...] = (".wav", ".flac"),
    recursive: bool = False,
) -> List[Path]:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {data_dir.resolve()}"
        )

    if recursive:
        files = sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        )
    else:
        files = sorted(
            path
            for path in data_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in extensions
        )

    if not files:
        raise FileNotFoundError(
            f"No supported audio files found in: "
            f"{data_dir.resolve()}"
        )

    return files