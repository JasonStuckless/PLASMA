from pathlib import Path
from typing import List


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_wav_files(data_dir: str | Path) -> List[Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in: {data_dir.resolve()}")
    return files