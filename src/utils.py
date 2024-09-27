import lightning.pytorch as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
import polars.selectors as cs
from typing import Union
import numpy as np


class TimeSeriesDataSet(Dataset):

    def __init__(
        self, df: pl.DataFrame, input_column: str, window_size: int = 10
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
        )

        self.X = torch.from_numpy(np.vstack(df["historical_data"].to_numpy()))

        self.y = df[input_column].to_torch()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


class TimeSeriesDataModule(L.LightningDataModule):

    def __init__(
        self,
        input_path: str,
        input_column: str,
        window_size: int = 10,
        batch_size: int = 128,
        num_workes: int = 5,
    ) -> None:
        super().__init__()
        print(input_path)
        self.df = pl.read_csv(input_path)

        self.batch_size = batch_size
        self.num_workers = num_workes
        self.window_size = window_size
        self.input_column = input_column

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TimeSeriesDataSet(
                self.df.filter(pl.col("split_type") == "train"),
                input_column=self.input_column,
                window_size=self.window_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            TimeSeriesDataSet(
                self.df.filter(pl.col("split_type") == "val"),
                input_column=self.input_column,
                window_size=self.window_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            TimeSeriesDataSet(
                self.df.filter(pl.col("split_type") == "test"),
                input_column=self.input_column,
                window_size=self.window_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
