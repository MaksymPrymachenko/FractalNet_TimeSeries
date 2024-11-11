from lightning.pytorch import cli_lightning_logo
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.models.base_model import BaseModelPL
from src.utils.datamodule import TimeSeriesDataModule


class CustomLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "model.init_args.normalizer", "data.normalizer", apply_on="instantiate"
        )


def cli_main():
    early_stopping = EarlyStopping(monitor="val/loss", patience=50, mode="min")
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{val/loss:.4f}", save_top_k=1, monitor="val/loss", mode="min"
    )

    cli = CustomLightningCLI(
        BaseModelPL,
        TimeSeriesDataModule,
        auto_configure_optimizers=False,
        subclass_mode_model=True,
        seed_everything_default=42,
        run=True,
        trainer_defaults={
            "max_epochs": 1_000,
            "logger": {
                "class_path": "lightning.pytorch.loggers.MLFlowLogger",
                "init_args": {"save_dir": "./mlruns/", "log_model": True},
            },
            "callbacks": [early_stopping, learning_rate_monitor, model_checkpoint],
            # "log_every_n_steps": 1,
        },
        save_config_kwargs={"overwrite": True},
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    cli.trainer.test(
        cli.model, datamodule=cli.datamodule, ckpt_path=model_checkpoint.best_model_path
    )


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
