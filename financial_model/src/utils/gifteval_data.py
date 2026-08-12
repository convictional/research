from huggingface_hub import hf_hub_download
import pandas as pd
from typing import Dict, Optional
import numpy as np
import pyarrow as pa
import os


class GiftEvalDataLoader:
    """
    Data loader for GiftEval time series datasets.
    Focuses on M4 daily time series data.
    """

    def __init__(self):
        """Initialize the data loader for the GiftEval dataset, focusing on M4 daily data."""
        try:
            # Download the arrow file
            self.file_path = hf_hub_download(
                repo_id="Salesforce/GiftEval", filename="m4_daily/data-00000-of-00001.arrow", repo_type="dataset"
            )

            print(f"Downloaded file to: {self.file_path}")
            print(f"File exists: {os.path.exists(self.file_path)}")
            print(f"File size: {os.path.getsize(self.file_path)} bytes")

            # Try to load as a streaming Arrow file
            with open(self.file_path, "rb") as f:
                # Read all record batches from the stream
                reader = pa.ipc.open_stream(f)
                self.arrow_table = pa.Table.from_batches(reader)
                print("Successfully loaded with pa.ipc.open_stream")

            self.metadata = {"file_path": self.file_path}

        except Exception as e:
            print(f"Warning: Failed to load dataset with error: {e}")
            self.arrow_table = None
            self.metadata = {}

    def load_timeseries(self, n_series: Optional[int] = 1, random_seed: Optional[int] = 42) -> pd.DataFrame:
        """
        Load M4 daily time series data from the dataset.

        Args:
            n_series (Optional[int]): Number of time series to load. If None, loads all series.
            random_seed (Optional[int]): Random seed for reproducible sampling.

        Returns:
            pd.DataFrame: DataFrame containing the loaded time series data
        """
        if self.arrow_table is None:
            raise RuntimeError("Dataset failed to load")

        # Convert to pandas DataFrame
        df = self.arrow_table.to_pandas()

        # Extract target values from list column
        series_data = pd.DataFrame([list(row) for row in df["target"]])

        if n_series is not None:
            if random_seed is not None:
                np.random.seed(random_seed)

            # Sample n_series randomly if specified
            total_series = len(series_data)
            if n_series > total_series:
                raise ValueError(f"Requested {n_series} series but dataset only contains {total_series}")

            indices = np.random.choice(total_series, size=n_series, replace=False)
            series_data = series_data.iloc[indices].reset_index(drop=True)

        return series_data

    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.

        Returns:
            Dict: Dictionary containing dataset metadata
        """
        return self.metadata

    def get_series_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate basic statistics about the time series.

        Args:
            df (Optional[pd.DataFrame]): DataFrame containing time series data.
                                       If None, loads all data first.

        Returns:
            Dict: Dictionary containing statistics about the time series
        """
        if df is None:
            df = self.load_timeseries()

        stats = {
            "n_series": len(df),
            "mean_length": df.notna().sum(axis=1).mean(),
            "min_length": df.notna().sum(axis=1).min(),
            "max_length": df.notna().sum(axis=1).max(),
            "total_points": df.notna().sum().sum(),
        }

        return stats
