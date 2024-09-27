from lightning.pytorch import cli_lightning_logo
from lightning.pytorch.cli import LightningCLI

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import lightning.pytorch as pl

from src.models.base_model import BaseModelPL
from src.utils import TimeSeriesDataModule


def cli_main():
    cli = LightningCLI(
        BaseModelPL,
        TimeSeriesDataModule,
        subclass_mode_model=True,
        seed_everything_default=42,
        run=True,
        trainer_defaults={
            "max_epochs": 1_000,
            "logger": {
                "class_path": "lightning.pytorch.loggers.MLFlowLogger",
                "init_args": {"save_dir": "./mlruns/", "log_model": "all"},
            },
            "callbacks": [
                EarlyStopping(
                    monitor="val/loss",
                    patience=50,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    filename="{epoch}-{val/loss:.4f}", save_top_k=3, monitor="val/loss"
                ),
            ],
        },
        save_config_kwargs={"overwrite": True},
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
