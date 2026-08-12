import pickle
from pathlib import Path
from typing import Any

from ..config.experiment_settings import settings


def dump_to_pickle_file(data: Any, path: Path = settings.output_path / "data.pkl"):
    print(f"Dumping data to pickle file {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle_file(path: Path = settings.output_path / "data.pkl") -> Any:
    print(f"Loading data from pickle file {path}...")
    with open(path, "rb") as f:
        return pickle.load(f)
