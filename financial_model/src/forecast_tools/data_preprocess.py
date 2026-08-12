import pandas as pd
import numpy as np
from ..helpers.print_section import print_section, Colors


class DataPreprocessor:
    """Agent responsible for cleaning and preparing data for analysis"""

    def __init__(self, data_source: str = "csv"):
        self.data_source = data_source

    def _clean_datetime_column(self, series: pd.Series) -> pd.Series:
        """Standardize datetime column by removing timezone and standardizing format"""
        try:
            # First try to parse the dates with timezone awareness
            clean_dates = pd.to_datetime(series, utc=True)
            # Then remove the timezone
            return clean_dates.dt.tz_localize(None)
        except Exception:
            # If that fails, try simple parsing without timezone handling
            return pd.to_datetime(series)

    def _prepare_gifteval_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare GiftEval data for forecasting"""
        if target_column is None:
            target_column = "value"

        # Create date index
        dates = pd.date_range(
            start="2020-01-01",  # Arbitrary start date
            periods=data.shape[1],
            freq="D",
        )

        # Melt the dataframe to get it in the right format
        melted_df = data.melt(var_name="date_id", value_name=target_column)
        melted_df["date_id"] = np.tile(dates, len(data))

        # Remove any NaN values
        melted_df = melted_df.dropna()

        # Sort and set index
        melted_df = melted_df.sort_values("date_id")
        melted_df = melted_df.set_index("date_id")

        return melted_df

    def _prepare_csv_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare CSV data for forecasting"""
        working_data = data.copy()
        working_data["date_id"] = self._clean_datetime_column(data["date_id"])
        working_data = working_data.set_index("date_id")
        working_data = working_data.sort_index()
        working_data = working_data.groupby(level=0)[target_column].sum().to_frame()
        return working_data

    async def prepare_data(self, data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for analysis and split into train/validation sets"""
        print_section("Data Preprocessing", f"Data Source: {self.data_source}", Colors.GREEN)

        # Process data based on source type
        if self.data_source == "csv":
            working_data = self._prepare_csv_data(data, target_column)
        elif self.data_source == "gifteval":
            working_data = self._prepare_gifteval_data(data, target_column)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

        # Split into train/validation sets
        train_data, val_data = self.split_train_validation(working_data)

        print_section(
            "Preprocessed Data Summary",
            f"Rows: {len(working_data)}\nDate Range: {working_data.index.min()} to {working_data.index.max()}\n"
            f"Target Column Stats:\n{working_data[working_data.columns[0]].describe()}",
            Colors.GREEN,
        )

        return train_data, val_data

    def split_train_validation(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets using last 20% for validation"""
        total_rows = len(data)
        val_size = int(round(total_rows * 0.2))  # 20% for validation

        train_data = data.iloc[:-val_size].copy()
        val_data = data.iloc[-val_size:].copy()

        print_section(
            "Train/Validation Split",
            f"Total rows: {total_rows}\n"
            f"Training rows: {len(train_data)}\n"
            f"Validation rows: {len(val_data)}\n"
            f"Training date range: {train_data.index.min()} to {train_data.index.max()}\n"
            f"Validation date range: {val_data.index.min()} to {val_data.index.max()}",
            Colors.GREEN,
        )

        return train_data, val_data
