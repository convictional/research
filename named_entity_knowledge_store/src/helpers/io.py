import pickle
from pathlib import Path
from typing import Any

from ..settings import settings


def dump_to_pickle_file(data: Any, path: Path = settings.output_path / "data.pkl"):
    print(f"Dumping data to pickle file {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle_file(path: Path = settings.output_path / "data.pkl") -> Any:
    print(f"Loading data from pickle file {path}...")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_checkpoint_path(name: str) -> Path:
    """Get the path for a specific checkpoint file."""
    return settings.output_path / "checkpoints" / f"{name}.pkl"


def save_checkpoint(data: Any, name: str):
    """Save a checkpoint with a specific name."""
    path = get_checkpoint_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_to_pickle_file(data, path)


def load_checkpoint(name: str, default=None) -> Any:
    """Load a checkpoint if it exists, otherwise return default."""
    path = get_checkpoint_path(name)
    if path.exists():
        return load_pickle_file(path)
    return default
