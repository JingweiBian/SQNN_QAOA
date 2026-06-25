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
    QUBOPairAwarePhaseSQNN,
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
    sample_pair_guided,
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

    pair_model = QUBOPairAwarePhaseSQNN(
        num_variables=problem.num_variables,
        message_rounds=3,
        pair_energy_weight=1.0,
        corr_regularization=0.0,
        monotone_accept=True,
    )
    raw_corr0 = pair_model._initial_raw_corr(problem)
    p0 = torch.full((problem.num_variables,), 0.5)
    pair_energy0 = pair_model.pair_expected_energy(
        problem,
        p0,
        raw_corr0,
        include_regularization=False,
    )
    product_energy0 = problem.expected_energy(p0)
    if not torch.allclose(pair_energy0, product_energy0, atol=1e-6):
        raise AssertionError(
            f"corr=0 pair energy should match product energy: {pair_energy0} vs {product_energy0}"
        )
    pair_result = pair_model(problem, return_state=True)
    pair_probabilities = pair_result["probabilities"]
    if pair_probabilities.shape != (problem.num_variables,):
        raise AssertionError(f"unexpected pair-aware probabilities shape: {pair_probabilities.shape}")
    pair_belief = pair_result["pair_belief"].detach()
    if pair_belief.numel() and not torch.allclose(
        pair_belief.sum(dim=(-1, -2)),
        torch.ones(pair_belief.shape[0], dtype=pair_belief.dtype, device=pair_belief.device),
        atol=1e-5,
    ):
        raise AssertionError(f"invalid pair-belief normalization: {pair_belief}")
    pair_energy_trace = pair_result["energy_trace"].detach()
    if bool((pair_energy_trace[1:] > pair_energy_trace[:-1] + 1e-6).any()):
        raise AssertionError(f"pair-aware energy increased: {pair_energy_trace}")

    pair_grad_model = QUBOPairAwarePhaseSQNN(
        num_variables=problem.num_variables,
        message_rounds=3,
        pair_energy_weight=1.0,
        corr_regularization=0.0,
        monotone_accept=False,
    )
    pair_grad_state = pair_grad_model(problem, return_state=True)
    pair_grad_state["loss_energy"].backward()
    if pair_grad_model.raw_corr_steps.grad is None:
        raise AssertionError("pair-aware model did not receive corr-step gradients")

    positive_edge_problem = QUBOProblem.from_terms(
        num_variables=2,
        linear=torch.zeros(2),
        edge_index=torch.tensor([[0], [1]]),
        edge_weight=torch.tensor([1.0]),
    )
    negative_edge_problem = QUBOProblem.from_terms(
        num_variables=2,
        linear=torch.zeros(2),
        edge_index=torch.tensor([[0], [1]]),
        edge_weight=torch.tensor([-1.0]),
    )
    pair_direction_model = QUBOPairAwarePhaseSQNN(
        num_variables=2,
        message_rounds=1,
        pair_energy_weight=1.0,
        corr_regularization=0.0,
        monotone_accept=False,
    )
    midpoint = torch.full((2,), 0.5)
    raw_zero = torch.zeros(1)
    positive_next_corr = torch.tanh(
        pair_direction_model._propose_raw_corr(positive_edge_problem, midpoint, raw_zero, 0)
    )[0]
    negative_next_corr = torch.tanh(
        pair_direction_model._propose_raw_corr(negative_edge_problem, midpoint, raw_zero, 0)
    )[0]
    if not bool(positive_next_corr < 0.0):
        raise AssertionError(f"positive edge should push corr negative, got {positive_next_corr}")
    if not bool(negative_next_corr > 0.0):
        raise AssertionError(f"negative edge should push corr positive, got {negative_next_corr}")

    consistency_model = QUBOPairAwarePhaseSQNN(
        num_variables=2,
        message_rounds=1,
        corr_consistency_weight=1.0,
        pair_message_weight=1.0,
    )
    unsupported_corr_energy = consistency_model.corr_consistency_energy(
        positive_edge_problem,
        torch.tensor([0.5, 0.5]),
        torch.tensor([2.0]),
    )
    supported_corr_energy = consistency_model.corr_consistency_energy(
        positive_edge_problem,
        torch.tensor([0.99, 0.99]),
        torch.tensor([2.0]),
    )
    if not bool(unsupported_corr_energy > supported_corr_energy):
        raise AssertionError(
            "unsupported edge corr should be penalized more than node-supported corr: "
            f"{unsupported_corr_energy} <= {supported_corr_energy}"
        )
    consistency_field = consistency_model._corr_consistency_field(
        positive_edge_problem,
        torch.tensor([0.8, 0.2]),
        torch.tensor([2.0]),
    )
    if not bool(consistency_field[0] > 0.0 and consistency_field[1] < 0.0):
        raise AssertionError(f"consistency field has wrong direction: {consistency_field}")

    relation_model = QUBOPairAwarePhaseSQNN(
        num_variables=2,
        message_rounds=1,
        pair_relation_center=False,
    )
    same_relation = relation_model._pair_relation_signal(
        positive_edge_problem,
        torch.tensor([0.8, 0.2]),
        torch.tensor([2.0]),
    )
    if not bool(same_relation[0] < 0.0 and same_relation[1] > 0.0):
        raise AssertionError(f"same-corr relation signal has wrong direction: {same_relation}")
    anti_relation = relation_model._pair_relation_signal(
        positive_edge_problem,
        torch.tensor([0.8, 0.2]),
        torch.tensor([-2.0]),
    )
    if not bool(anti_relation.abs().max() < 1e-5):
        raise AssertionError(f"anti-corr satisfied relation should be near zero: {anti_relation}")

    anti_pair_belief = torch.tensor([[[0.0, 0.5], [0.5, 0.0]]])
    anti_samples = sample_pair_guided(
        positive_edge_problem,
        torch.full((2,), 0.5),
        anti_pair_belief,
        num_samples=16,
    )
    if not bool((anti_samples[:, 0] != anti_samples[:, 1]).all().item()):
        raise AssertionError(f"pair-guided readout ignored anti-correlated pair belief: {anti_samples}")
    rooted_anti_samples = sample_pair_guided(
        positive_edge_problem,
        torch.tensor([0.9, 0.5]),
        anti_pair_belief,
        num_samples=16,
        root_strategy="confidence",
        root_mode="round",
    )
    if not bool(((rooted_anti_samples[:, 0] == 1.0) & (rooted_anti_samples[:, 1] == 0.0)).all().item()):
        raise AssertionError(f"high-confidence root did not propagate anti-correlation: {rooted_anti_samples}")

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
