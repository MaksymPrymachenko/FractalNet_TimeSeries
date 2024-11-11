import lightning.pytorch as L
from typing import Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD, AdamW
from torch.optim import Optimizer
import torch
from torch import nn

import abc
from typing import Optional
from src.utils.normalizer import BaseNormalizer
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


class BaseModelPL(L.LightningModule):

    def __init__(
        self,
        lr: float = 1e-2,
        normalizer: Optional[BaseNormalizer] = None,
    ) -> None:
        super(BaseModelPL, self).__init__()
        self.criterion = nn.HuberLoss()  # nn.MSELoss()  # nn.L1Loss()
        self.optimizer = Adam
        self.normalizer = normalizer

        self.save_hyperparameters()

    @abc.abstractmethod
    def forward(self, X: torch.tensor) -> torch.Tensor:
        """Abstract forward method with same input and output for all models"""
        return

    def calculate_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, prefix=""
    ) -> Dict[str, float]:

        return {
            f"{prefix}loss": self.criterion(y_hat, y),
            f"{prefix}mae": mean_absolute_error(y_hat, y),
            f"{prefix}rmse": torch.sqrt(mean_squared_error(y_hat, y)),
            f"{prefix}mape": mean_absolute_percentage_error(y_hat, y),
            f"{prefix}r2": r2_score(y_hat, y),
        }

    def shared_step(
        self, batch: torch.Tensor, split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        X, y = batch
        y_hat = self.forward(X)

        metrics = self.calculate_metrics(y_hat, y, prefix=f"{split}/")

        inversed_y_hat = self.normalizer.inverse_transform_tensor(y_hat)
        inversed_y = self.normalizer.inverse_transform_tensor(y)

        inversed_metrics = self.calculate_metrics(
            inversed_y_hat, inversed_y, prefix=f"{split}/inversed_"
        )

        metrics.update(inversed_metrics)

        self.log_dict({k: v.item() for k, v in metrics.items()})

        return metrics[f"{split}/loss"]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, split="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, split="test")

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        X, y = batch
        return (self.forward(X), y)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }
