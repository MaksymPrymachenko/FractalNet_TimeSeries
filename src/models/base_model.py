import lightning.pytorch as L
from typing import Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD
import torch
from torch import nn

import abc


class BaseModelPL(L.LightningModule):

    def __init__(self, lr: float = 1e-2) -> None:
        super(BaseModelPL, self).__init__()

        self.save_hyperparameters()

        self.criterion = nn.L1Loss()  # nn.MSELoss()

    @abc.abstractmethod
    def forward(self, X: torch.tensor) -> torch.Tensor:
        """Abstract forward method with same input and output for all models"""
        return

    def shared_step(
        self, batch: torch.Tensor, split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        X, y = batch

        y_hat = self.forward(X)

        loss = self.criterion(y_hat, y.unsqueeze(-1))

        self.log_dict({f"{split}/loss": loss.item()})

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, split="val")

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        X, y = batch
        return (self.forward(X), y)

    def configure_optimizers(self) -> Dict[str, Any]:
        print(self.hparams)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
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
