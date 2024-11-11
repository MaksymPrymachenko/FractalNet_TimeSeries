from abc import ABC, abstractmethod
from typing import Optional
import polars as pl
import torch


class BaseNormalizer(ABC):

    @abstractmethod
    def transform(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def transform_tensor(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def inverse_transform(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def inverse_transform_tensor(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def fit_transform(self):
        raise NotImplementedError("Method not implemented")


class ZScoreNormalizer(BaseNormalizer):

    def __init__(
        self, mean: Optional[float] = None, std: Optional[float] = None
    ) -> None:
        self.mean = mean
        self.std = std

    def transform(self, df: pl.DataFrame, feature_column: str) -> pl.DataFrame:
        return df.with_columns(
            ((pl.col(feature_column) - self.mean) / self.std).alias("transformed")
        )

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def inverse_transform(self, df: pl.DataFrame, feature_column: str) -> pl.DataFrame:
        return df.with_columns(
            (pl.col(feature_column) * self.std + self.mean).alias("inverse_transformed")
        )

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

    def fit(self, df: pl.DataFrame, feature_column: str) -> None:
        self.mean = df.filter(pl.col("split") == "train")[feature_column].mean()
        self.std = df.filter(pl.col("split") == "train")[feature_column].std()

    def fit_transform(self, df: pl.DataFrame, feature_column: str) -> pl.DataFrame:
        self.fit(df, feature_column)
        return self.transform(df, feature_column)


class MaxMinNormalizer(BaseNormalizer):

    def __init__(
        self, min: Optional[float] = None, max: Optional[float] = None
    ) -> None:
        self.min = min
        self.max = max

    def transform(self, df: pl.DataFrame, feature_column: str) -> pl.DataFrame:

        return df.with_columns(
            ((pl.col(feature_column) - self.min) / (self.max - self.min)).alias(
                "transformed"
            )
        )

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:

        return (tensor - self.min) / (self.max - self.min)

    def inverse_transform(self, df: pl.DataFrame, feature_column: str) -> pl.DataFrame:

        return df.with_columns(
            (pl.col(feature_column) * (self.max - self.min) + self.min).alias(
                "inverse_transformed"
            )
        )

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:

        return tensor * (self.max - self.min) + self.min

    def fit(self, df: pl.DataFrame, feature_column: str) -> None:

        train_data = df.filter(pl.col("split") == "train")
        self.min = train_data[feature_column].min()
        self.max = train_data[feature_column].max()

    def fit_transform(self, df: pl.DataFrame, feature_column: str) -> pl.DataFrame:

        self.fit(df, feature_column)
        return self.transform(df, feature_column)
