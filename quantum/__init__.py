# -*- coding: utf-8 -*-

import os

_repo_root = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_repo_root, ".matplotlib"))

from .core.layers import (
    InputEncodedLayerAngleEncoding,
    InputEncodedLayerParallelEncoding,
    MultiBasisQuantumNeuronLayer,
    QuantumNeuronLayer,
)
from .encoders.group_encoder import (
    GroupSequentialQuantumEncoder,
    MultiBasisReadoutGroupEncoder,
    RingMultiBasisGroupEncoder,
    SequentialMeasurementGroupEncoder,
)
from .warmstart import (
    QUBOProblem,
    QUBOWarmStartSQNN,
    best_sample_from_probabilities,
    entropy_regularized_qubo_loss,
    qaoa_ry_angles_from_probabilities,
    qubo_expected_energy_loss,
    sample_qubo_solutions,
)


__all__ = [
    "DataReuploadingSoftQuantumNeuralNetwork",
    "EncodingSkipSoftQuantumNeuralNetwork",
    "InputEncodedLayerAngleEncoding",
    "InputEncodedLayerParallelEncoding",
    "GroupSequentialQuantumEncoder",
    "MultiBasisReadoutGroupEncoder",
    "MultiBasisQuantumNeuronLayer",
    "QuantumLoss",
    "QuantumNeuronLayer",
    "QuantumTrainer",
    "QUBOProblem",
    "QUBOWarmStartSQNN",
    "RingMultiBasisGroupEncoder",
    "SequentialMeasurementGroupEncoder",
    "SoftQuantumNeuralNetwork",
    "best_sample_from_probabilities",
    "entropy_regularized_qubo_loss",
    "qaoa_ry_angles_from_probabilities",
    "qubo_expected_energy_loss",
    "sample_qubo_solutions",
]


def __getattr__(name):
    if name == "QuantumTrainer":
        from .training.trainer import QuantumTrainer

        return QuantumTrainer
    if name in {
        "DataReuploadingSoftQuantumNeuralNetwork",
        "EncodingSkipSoftQuantumNeuralNetwork",
        "QuantumLoss",
        "SoftQuantumNeuralNetwork",
    }:
        from .classifiers.networkmodels import (
            DataReuploadingSoftQuantumNeuralNetwork,
            EncodingSkipSoftQuantumNeuralNetwork,
            QuantumLoss,
            SoftQuantumNeuralNetwork,
        )

        return {
            "DataReuploadingSoftQuantumNeuralNetwork": DataReuploadingSoftQuantumNeuralNetwork,
            "EncodingSkipSoftQuantumNeuralNetwork": EncodingSkipSoftQuantumNeuralNetwork,
            "QuantumLoss": QuantumLoss,
            "SoftQuantumNeuralNetwork": SoftQuantumNeuralNetwork,
        }[name]
    raise AttributeError(f"module 'quantum' has no attribute {name!r}")
