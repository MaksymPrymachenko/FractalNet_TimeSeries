from functools import partial
import argparse
import yaml

import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

import optuna

from typing import Dict, Any

from src.models.nets import LSTM_Net, LSTM_GRU_Net, CNN_LSTM_Net, FractalNet

from src.utils import TimeSeriesDataModule

MODEL_CLASSES = {
    "LSTM_Net": LSTM_Net,
    "LSTM_GRU_Net": LSTM_GRU_Net,
    "CNN_LSTM_Net": CNN_LSTM_Net,
    "FractalNet": FractalNet,
}


def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f_i:
        config = yaml.safe_load(f_i)

    return config


def get_hyperparameters(trial, hyperparams_config):

    params = {}
    for param, details in hyperparams_config.items():
        param_type = details["type"]
        if param_type == "int":
            low = details["range"]["low"]
            high = details["range"]["high"]
            params[param] = trial.suggest_int(param, low, high)
        elif param_type == "float":
            low = details["range"]["low"]
            high = details["range"]["high"]
            params[param] = trial.suggest_float(param, low, high)
        elif param_type == "loguniform":
            low = details["range"]["low"]
            high = details["range"]["high"]
            params[param] = trial.suggest_loguniform(param, low, high)
        elif param_type == "categorical":
            choices = details["choices"]
            params[param] = trial.suggest_categorical(param, choices)
        else:
            raise ValueError(f"Unsupported hyperparameter type: {param_type}")

    return params


def objective(
    trial: optuna.trial.Trial, model_config: Dict[str, Any], data_config: Dict[str, Any]
) -> float:
    model_class = MODEL_CLASSES[model_config["model"]["class"]]

    hyperparams = get_hyperparameters(trial, model_config["hyperparameters"])

    hyperparams.update(model_config["model"]["constant_parameters"])

    model = model_class(**hyperparams)

    data_module = TimeSeriesDataModule(
        data_config["data_path"],
        input_column=data_config["input_column"],
        window_size=data_config["window_size"],
        batch_size=data_config["batch_size"],
    )

    model_checkpoint_callback = ModelCheckpoint(monitor="val/loss")

    trainer = L.Trainer(
        max_epochs=model_config["trainer"]["max_epochs"],
        accelerator="gpu",
        enable_progress_bar=True,
        callbacks=[
            model_checkpoint_callback,
            EarlyStopping(
                monitor="val/loss",
                patience=model_config["trainer"]["early_stopping_patience"],
            ),
        ],
    )

    trainer.fit(model, datamodule=data_module)

    return model_checkpoint_callback.best_model_score.item()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optuna tune arguments")

    parser.add_argument("--model_config", "-mc", type=str)

    parser.add_argument("--data_config", "-dc", type=str)

    parser.add_argument("--study_name", "-sn", type=str)

    args = parser.parse_args()

    model_config = read_config(args.model_config)
    data_config = read_config(args.data_config)

    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///db.sqlite3",
        study_name=args.study_name,
    )

    objective_with_configs = partial(
        objective, model_config=model_config, data_config=data_config
    )

    study.optimize(
        objective_with_configs,
        n_trials=model_config["optuna"]["n_trials"],
        timeout=model_config["optuna"]["timeout"],
    )

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")

    trial = study.best_trial

    print(f"Value: {trial.value}")

    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
