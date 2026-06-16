# -*- coding: utf-8 -*-

__all__ = [
    "EfficientQuantumSampler",
    "OptimizerManager",
    "create_mnist_loaders",
    "filter_dataset",
    "plot_training_metrics",
    "save_training_history",
]


def __getattr__(name):
    if name in {"create_mnist_loaders", "filter_dataset"}:
        from .dataloader import create_mnist_loaders, filter_dataset

        return {
            "create_mnist_loaders": create_mnist_loaders,
            "filter_dataset": filter_dataset,
        }[name]

    if name in {
        "EfficientQuantumSampler",
        "OptimizerManager",
        "plot_training_metrics",
        "save_training_history",
    }:
        from .utils import (
            EfficientQuantumSampler,
            OptimizerManager,
            plot_training_metrics,
            save_training_history,
        )

        return {
            "EfficientQuantumSampler": EfficientQuantumSampler,
            "OptimizerManager": OptimizerManager,
            "plot_training_metrics": plot_training_metrics,
            "save_training_history": save_training_history,
        }[name]

    raise AttributeError(f"module 'tools' has no attribute {name!r}")
