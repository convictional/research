from pathlib import Path
from typing import Any
import pandas as pd


def dump_list_of_objects_to_csv(objects: list[Any], file_path: Path):
    print(f"Dumping data to csv file: {file_path}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([x.model_dump() for x in objects])
    df.to_csv(file_path, index=False)
