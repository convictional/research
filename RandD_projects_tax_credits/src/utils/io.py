import pickle
from pathlib import Path
from typing import Any
import pandas as pd

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


def dump_list_of_objects_to_csv(objects: list[Any], file_path: Path):
    """
    This function takes a list of objects (classes) and dumps them to a csv file.
    An intermediate step is to convert the objects to a pandas DataFrame.
    """
    print(f"Dumping data to csv file: {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([x.model_dump() for x in objects])
    df.to_csv(file_path, index=False)
