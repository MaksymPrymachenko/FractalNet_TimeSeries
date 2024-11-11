import lightning.pytorch as L
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
import polars.selectors as cs
from typing import Optional
import numpy as np
from src.utils.normalizer import BaseNormalizer
import numpy as np


class TimeSeriesDataSet(Dataset):

    def __init__(
        self,
        df: pl.DataFrame,
        input_column: str,
        window_size: int = 10,
        predict_size: int = 1,
    ) -> None:

        df = (
            df.with_columns(pl.col(input_column).cast(pl.Float32))
            .with_columns(
                pl.col(input_column).shift(i).alias(f"lag_{i}")
                for i in reversed(range(1, window_size + 1))
            )
            .drop_nulls()
            .with_columns(
                pl.concat_list(cs.starts_with("lag_")).alias("historical_data")
            )
            .with_columns(
                pl.col(input_column).shift(-i).alias(f"pr_lag_{i}")
                for i in range(0, predict_size)
            )
            .drop_nulls()
            .with_columns(
                pl.concat_list(cs.starts_with("pr_lag_")).alias(
                    "historical_data_predict"
                )
            )
        )

        self.X = torch.from_numpy(np.vstack(df["historical_data"].to_numpy()))

        self.y = torch.from_numpy(np.vstack(df["historical_data_predict"].to_numpy()))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


class TimeSeriesDataModule(L.LightningDataModule):

    def __init__(
        self,
        input_path: str,
        input_column: str,
        predict_size: int = 1,
        window_size: int = 10,
        normalizer: Optional[BaseNormalizer] = None,
        batch_size: int = 128,
        num_workes: int = 5,
    ) -> None:
        super().__init__()

        self.input_path = input_path
        self.batch_size = batch_size
        self.num_workers = num_workes
        self.window_size = window_size
        self.input_column = input_column
        self.predict_size = predict_size

        self.normalizer = normalizer

        df = pl.read_csv(self.input_path)

        if self.normalizer is not None:
            df = self.normalizer.fit_transform(df, self.input_column)

            self.input_column = "transformed"

        self.df = df

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TimeSeriesDataSet(
                self.df.filter(pl.col("split") == "train"),
                input_column=self.input_column,
                window_size=self.window_size,
                predict_size=self.predict_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            TimeSeriesDataSet(
                self.df.filter(pl.col("split") == "valid"),
                input_column=self.input_column,
                window_size=self.window_size,
                predict_size=self.predict_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            TimeSeriesDataSet(
                self.df.filter(pl.col("split") == "test"),
                input_column=self.input_column,
                window_size=self.window_size,
                predict_size=self.predict_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
