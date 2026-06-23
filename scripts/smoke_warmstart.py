# -*- coding: utf-8 -*-

"""Small smoke checks for the QUBO warm-start stack."""

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantum.warmstart import (
    QUBOProblem,
    QUBOQuantumDataWarmStartSQNN,
    QUBOSynchronousLocalFieldSQNN,
    calibrate_probabilities_with_assignment,
    componentwise_qaoa_resource_summary,
    make_planted_parity_qubo,
    optimize_qaoa_statevector,
    qubo_component_subproblems,
    qubo_connected_components,
    reduce_by_fixing_isolated_variables,
    residual_qaoa_active_summary,
)


def assert_close(value, expected, tol=1e-5):
    if abs(float(value) - float(expected)) > tol:
        raise AssertionError(f"expected {expected}, got {value}")


def main():
    problem = QUBOProblem.from_terms(
        num_variables=4,
        linear=torch.tensor([1.0, -2.0, 0.5, -0.25]),
        edge_index=torch.tensor([[0], [1]]),
        edge_weight=torch.tensor([-1.0]),
    )
    reduced, kept_indices, isolated_mask, fixed_values = reduce_by_fixing_isolated_variables(problem)
    if reduced is None:
        raise AssertionError("expected a non-empty active problem")
    if kept_indices.detach().cpu().tolist() != [0, 1]:
        raise AssertionError(f"unexpected kept indices: {kept_indices}")
    if isolated_mask.detach().cpu().tolist() != [False, False, True, True]:
        raise AssertionError(f"unexpected isolated mask: {isolated_mask}")
    if fixed_values.detach().cpu().tolist() != [0.0, 1.0, 0.0, 1.0]:
        raise AssertionError(f"unexpected fixed values: {fixed_values}")
    assert_close(reduced.constant, -0.25)
    assert_close(reduced.energy(torch.tensor([0.0, 1.0])), -2.25)
    if problem.quantum_node_features().shape != (problem.num_variables, 3):
        raise AssertionError(
            f"unexpected quantum-node feature shape: {problem.quantum_node_features().shape}"
        )

    quantum_data_model = QUBOQuantumDataWarmStartSQNN(
        num_variables=problem.num_variables,
        message_rounds=1,
    )
    quantum_data_probabilities = quantum_data_model(problem)
    if quantum_data_probabilities.shape != (problem.num_variables,):
        raise AssertionError(
            f"unexpected quantum-data probabilities shape: {quantum_data_probabilities.shape}"
        )
    if not bool(((quantum_data_probabilities >= 0.0) & (quantum_data_probabilities <= 1.0)).all()):
        raise AssertionError(f"invalid quantum-data probabilities: {quantum_data_probabilities}")
    quantum_data_loss = problem.expected_energy(quantum_data_probabilities)
    quantum_data_loss.backward()
    if quantum_data_model.node_local_angles.grad is None:
        raise AssertionError("quantum-data model did not receive gradients")

    sync_model = QUBOSynchronousLocalFieldSQNN(
        num_variables=problem.num_variables,
        message_rounds=3,
        monotone_accept=True,
    )
    sync_result = sync_model(problem, return_state=True)
    sync_probabilities = sync_result["probabilities"]
    if sync_probabilities.shape != (problem.num_variables,):
        raise AssertionError(f"unexpected sync-local probabilities shape: {sync_probabilities.shape}")
    energy_trace = sync_result["energy_trace"].detach()
    if bool((energy_trace[1:] > energy_trace[:-1] + 1e-6).any()):
        raise AssertionError(f"sync-local energy increased: {energy_trace}")

    sync_grad_model = QUBOSynchronousLocalFieldSQNN(
        num_variables=problem.num_variables,
        message_rounds=3,
        monotone_accept=False,
    )
    sync_loss = problem.expected_energy(sync_grad_model(problem))
    sync_loss.backward()
    if sync_grad_model.field_steps.grad is None:
        raise AssertionError("sync-local model did not receive gradients")

    positive_field_problem = QUBOProblem.from_terms(
        num_variables=1,
        linear=torch.tensor([2.0]),
    )
    negative_field_problem = QUBOProblem.from_terms(
        num_variables=1,
        linear=torch.tensor([-2.0]),
    )
    direction_model = QUBOSynchronousLocalFieldSQNN(
        num_variables=1,
        message_rounds=1,
        monotone_accept=True,
    )
    positive_probability = direction_model(positive_field_problem)[0]
    negative_probability = direction_model(negative_field_problem)[0]
    if not bool(positive_probability < 0.5):
        raise AssertionError(
            f"positive local field should push P(x=1) below 0.5, got {positive_probability}"
        )
    if not bool(negative_probability > 0.5):
        raise AssertionError(
            f"negative local field should push P(x=1) above 0.5, got {negative_probability}"
        )

    all_isolated = QUBOProblem.from_terms(
        num_variables=2,
        linear=torch.tensor([3.0, -4.0]),
    )
    empty_reduced, empty_indices, _, empty_values = reduce_by_fixing_isolated_variables(all_isolated)
    if empty_reduced is not None:
        raise AssertionError("all-isolated problem should have no active QAOA core")
    if empty_indices.numel() != 0:
        raise AssertionError("all-isolated kept indices should be empty")
    if empty_values.detach().cpu().tolist() != [0.0, 1.0]:
        raise AssertionError(f"unexpected all-isolated fixed values: {empty_values}")

    calibrated = calibrate_probabilities_with_assignment(
        torch.tensor([0.9, 0.8, 0.2]),
        torch.tensor([0.0, 1.0, 1.0]),
    )
    expected = torch.tensor([0.1, 0.8, 0.8])
    if not torch.allclose(calibrated, expected, atol=1e-6):
        raise AssertionError(f"unexpected calibrated probabilities: {calibrated}")

    active = residual_qaoa_active_summary(problem)
    if active["active_variables_after_isolated_fixing"] != 2:
        raise AssertionError(f"unexpected active summary: {active}")

    split_problem = QUBOProblem.from_terms(
        num_variables=5,
        linear=torch.zeros(5),
        edge_index=torch.tensor([[0, 2], [1, 3]]),
        edge_weight=torch.tensor([-1.0, -2.0]),
    )
    components = qubo_connected_components(split_problem)
    if [int(component.numel()) for component in components] != [2, 2, 1]:
        raise AssertionError(f"unexpected components: {components}")
    subproblems = qubo_component_subproblems(split_problem)
    if [item[0].num_edges for item in subproblems] != [1, 1, 0]:
        raise AssertionError(f"unexpected component subproblems: {subproblems}")
    component_summary = componentwise_qaoa_resource_summary(split_problem)
    if component_summary["max_component_variables"] != 2:
        raise AssertionError(f"unexpected component summary: {component_summary}")

    parity = make_planted_parity_qubo(16, average_degree=4, seed=11)
    planted_ratio = parity.approximation_ratio(parity.planted_assignment)
    assert_close(planted_ratio, 1.0)
    planted_energy = parity.problem.energy(parity.planted_assignment)
    assert_close(-planted_energy, parity.known_optimum)

    qaoa_result = optimize_qaoa_statevector(
        reduced,
        initial_probabilities=torch.tensor([0.5, 0.5]),
        layers=1,
        steps=3,
        lr=0.01,
        device="cpu",
    )
    if qaoa_result["num_states"] != 4:
        raise AssertionError(f"unexpected qaoa num_states: {qaoa_result}")
    if qaoa_result["best"]["expected_energy"] > 10:
        raise AssertionError(f"unexpected qaoa energy: {qaoa_result}")

    print("warmstart smoke checks passed")


if __name__ == "__main__":
    main()
