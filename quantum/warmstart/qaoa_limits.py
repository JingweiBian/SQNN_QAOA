# -*- coding: utf-8 -*-

"""Practical QAOA resource estimates for large QUBO warm-start planning."""

import math


def statevector_memory_bytes(num_qubits, complex_bytes=8):
    return int(complex_bytes) * (1 << int(num_qubits))


def max_statevector_qubits(memory_gb=12.0, safety_fraction=0.55, complex_bytes=8):
    usable = float(memory_gb) * (1024**3) * float(safety_fraction)
    return int(math.floor(math.log2(usable / int(complex_bytes))))


def qaoa_two_qubit_gate_count(num_edges, layers):
    return int(num_edges) * int(layers)


def qaoa_resource_summary(num_variables, num_edges, layers, gpu_memory_gb=12.0):
    max_qubits = max_statevector_qubits(gpu_memory_gb)
    statevector_possible = int(num_variables) <= max_qubits
    memory = (
        statevector_memory_bytes(num_variables) / (1024**3)
        if int(num_variables) <= 62
        else float("inf")
    )
    return {
        "num_variables": int(num_variables),
        "num_edges": int(num_edges),
        "layers": int(layers),
        "estimated_two_qubit_gates": qaoa_two_qubit_gate_count(num_edges, layers),
        "statevector_memory_gb_complex64": memory,
        "estimated_max_statevector_qubits_on_gpu": max_qubits,
        "full_statevector_possible_on_gpu": statevector_possible,
        "large_qaoa_note": (
            "full-state QAOA is not realistic; use SQNN to warm-start, "
            "fix confident variables, and run QAOA only on small subproblems"
            if not statevector_possible
            else "small enough for statevector experiments"
        ),
    }
