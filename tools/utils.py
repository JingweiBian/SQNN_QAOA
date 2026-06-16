# -*- coding: utf-8 -*-

"""Shared training utilities for the original SQNN classification workflow."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from matplotlib.ticker import MaxNLocator

try:
    from config.optimizer_config import SCHEDULER_CONFIGS, get_optimizer_config
except ModuleNotFoundError as exc:
    if exc.name != "config":
        raise

    _OPTIMIZER_CONFIGS = {
        "adam": {"lr": 1e-3},
        "adamw": {"lr": 1e-3},
        "sgd": {"lr": 1e-2, "momentum": 0.9},
        "rmsprop": {"lr": 1e-3},
        "adagrad": {"lr": 1e-2},
    }
    SCHEDULER_CONFIGS = {
        "step": {"step_size": 30, "gamma": 0.1},
        "cosine": {"T_max": 100},
        "plateau": {"mode": "min", "factor": 0.1, "patience": 10},
    }

    def get_optimizer_config(optimizer_type):
        return _OPTIMIZER_CONFIGS.get(str(optimizer_type).lower(), {}).copy()


def save_training_history(history, save_path):
    if not history.get("epoch"):
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=history.keys())
        writer.writeheader()
        for row_values in zip(*history.values()):
            writer.writerow(dict(zip(history.keys(), row_values)))
    return save_path


def plot_training_metrics(history, save_path, title="Training Metrics"):
    if not history.get("epoch"):
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.8)
    if "test_loss" in history and any(value is not None for value in history["test_loss"]):
        axes[0].plot(epochs, history["test_loss"], label="Test Loss", linewidth=1.8)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=1.8)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=1.8)
    if "test_acc" in history and any(value is not None for value in history["test_acc"]):
        axes[1].plot(epochs, history["test_acc"], label="Test Acc", linewidth=1.8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)

    if "lr" in history and any(value is not None for value in history["lr"]):
        axes[2].plot(epochs, history["lr"], label="Learning Rate", linewidth=1.8)
        axes[2].set_yscale("log")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


class EfficientQuantumSampler(nn.Module):
    """Average repeated stochastic forward passes without materializing huge batches."""

    def __init__(self, quantum_net, training_numbers, evaluation_numbers, chunk_size=256):
        super().__init__()
        self.quantum_net = quantum_net
        self.training_numbers = int(training_numbers)
        self.evaluation_numbers = int(evaluation_numbers)
        self.chunk_size = int(chunk_size)

        if self.training_numbers <= 0:
            raise ValueError("training_numbers must be positive")
        if self.evaluation_numbers <= 0:
            raise ValueError("evaluation_numbers must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    @property
    def output_dim(self):
        if not hasattr(self.quantum_net, "layer_dims"):
            return None
        if not hasattr(self.quantum_net, "layers_size"):
            return None
        if self.quantum_net.layers_size <= 0:
            return None
        return self.quantum_net.layer_dims[self.quantum_net.layers_size - 1]

    def _fast_path(self, inputs_batch, sample_numbers):
        expanded = inputs_batch.repeat(sample_numbers, 1)
        outputs = self.quantum_net(expanded)
        return outputs.view(sample_numbers, inputs_batch.size(0), -1).mean(0)

    def _chunk_avg(self, inputs_batch, sample_numbers):
        batch_size = inputs_batch.size(0)
        output_dim = self.output_dim
        if output_dim is None:
            raise RuntimeError("Cannot infer quantum network output dimension")

        accumulator = torch.zeros(batch_size, output_dim, device=inputs_batch.device)
        chunk_count = (sample_numbers + self.chunk_size - 1) // self.chunk_size
        for chunk_index in range(chunk_count):
            chunk_samples = min(
                self.chunk_size,
                sample_numbers - chunk_index * self.chunk_size,
            )
            expanded = inputs_batch.repeat(chunk_samples, 1)
            outputs = self.quantum_net(expanded)
            accumulator += outputs.view(chunk_samples, batch_size, -1).sum(0)
        return accumulator / sample_numbers

    def forward(self, inputs_batch):
        sample_numbers = (
            self.training_numbers if self.training else self.evaluation_numbers
        )
        if sample_numbers <= self.chunk_size:
            return self._fast_path(inputs_batch, sample_numbers)
        return self._chunk_avg(inputs_batch, sample_numbers)


class OptimizerManager:
    """Create and step optimizers/schedulers from the project config format."""

    def __init__(self, model_params, config=None):
        self.config = config or {}
        self.model_params = model_params
        self.optimizer = None
        self.scheduler = None

    def create_optimizer(self, optimizer_type=None, **overrides):
        optimizer_type = str(
            optimizer_type or self.config.get("optimizer_type", "adam")
        ).lower()

        params = get_optimizer_config(optimizer_type)
        config_params = self.config.get("optimizer", {})
        if isinstance(config_params, dict):
            params.update(config_params.get("params", {}))
        params.update(overrides)

        optimizer_map = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad,
        }
        if optimizer_type not in optimizer_map:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        self.optimizer = optimizer_map[optimizer_type](self.model_params, **params)
        return self.optimizer

    def create_scheduler(self, scheduler_type=None, **overrides):
        if self.optimizer is None:
            raise ValueError("Create the optimizer before the scheduler")

        scheduler_type = scheduler_type or self.config.get("scheduler_type")
        if not scheduler_type:
            return None
        scheduler_type = str(scheduler_type).lower()

        params = SCHEDULER_CONFIGS.get(scheduler_type, {}).copy()
        config_params = self.config.get("scheduler", {})
        if isinstance(config_params, dict):
            params.update(config_params.get("params", {}))
        params.update(overrides)

        if scheduler_type == "step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **params)
        elif scheduler_type == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, **params)
        elif scheduler_type == "plateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return self.scheduler

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def step_scheduler(self, metrics=None):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            if metrics is None:
                raise ValueError("ReduceLROnPlateau requires a metrics value")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def get_lr(self):
        if self.optimizer is None:
            return []
        return [group["lr"] for group in self.optimizer.param_groups]
