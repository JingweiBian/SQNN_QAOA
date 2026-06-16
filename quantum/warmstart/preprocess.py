# -*- coding: utf-8 -*-

"""Preprocessing helpers for residual QUBO instances."""

import torch

from .qaoa_limits import qaoa_resource_summary
from .qubo import QUBOProblem


def isolated_variable_fixing(problem):
    """Return the exact optimum assignment for isolated QUBO variables."""

    degree = problem.node_degrees(weighted=False)
    isolated_mask = degree == 0
    fixed_values = (problem.linear < 0).to(dtype=problem.linear.dtype)
    return isolated_mask, fixed_values


def reduce_by_fixing_isolated_variables(problem):
    """Eliminate degree-zero variables exactly.

    Returns:
        ``(reduced_problem, kept_indices, isolated_mask, fixed_values)``.

    ``reduced_problem`` is ``None`` when all variables were isolated and can be
    fixed classically, so no QAOA subproblem remains.
    """

    isolated_mask, fixed_values = isolated_variable_fixing(problem)
    if not bool(isolated_mask.any().item()):
        kept_indices = torch.arange(
            problem.num_variables,
            dtype=torch.long,
            device=problem.linear.device,
        )
        return problem, kept_indices, isolated_mask, fixed_values

    if bool(isolated_mask.all().item()):
        kept_indices = torch.empty(
            (0,),
            dtype=torch.long,
            device=problem.linear.device,
        )
        return None, kept_indices, isolated_mask, fixed_values

    reduced, kept_indices = problem.reduce_by_fixed_assignments(
        isolated_mask,
        fixed_values,
    )
    return reduced, kept_indices, isolated_mask, fixed_values


def qubo_connected_components(problem, include_isolated=True):
    """Return connected components of the QUBO interaction graph."""

    n = int(problem.num_variables)
    parent = list(range(n))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    if problem.edge_weight.numel():
        edges = problem.edge_index.detach().cpu()
        for edge_pos in range(edges.shape[1]):
            union(int(edges[0, edge_pos]), int(edges[1, edge_pos]))

    groups = {}
    degree = problem.node_degrees(weighted=False).detach().cpu()
    for index in range(n):
        if not include_isolated and float(degree[index]) == 0.0:
            continue
        groups.setdefault(find(index), []).append(index)

    components = [
        torch.tensor(indices, dtype=torch.long, device=problem.linear.device)
        for indices in groups.values()
    ]
    components.sort(key=lambda item: (-int(item.numel()), int(item[0].detach().cpu()) if item.numel() else 0))
    return components


def qubo_component_subproblems(problem, include_isolated=True):
    """Split a QUBO into induced connected-component subproblems."""

    components = qubo_connected_components(problem, include_isolated=include_isolated)
    subproblems = []
    for indices in components:
        old_to_new = torch.full(
            (problem.num_variables,),
            -1,
            dtype=torch.long,
            device=problem.linear.device,
        )
        old_to_new[indices] = torch.arange(
            indices.numel(),
            dtype=torch.long,
            device=problem.linear.device,
        )

        new_edges = []
        new_weights = []
        if problem.edge_weight.numel():
            index_set = set(indices.detach().cpu().tolist())
            for edge_pos in range(problem.edge_weight.numel()):
                src = int(problem.edge_index[0, edge_pos].detach().cpu())
                dst = int(problem.edge_index[1, edge_pos].detach().cpu())
                if src in index_set and dst in index_set:
                    new_edges.append(
                        [
                            int(old_to_new[src].detach().cpu()),
                            int(old_to_new[dst].detach().cpu()),
                        ]
                    )
                    new_weights.append(problem.edge_weight[edge_pos])

        if new_edges:
            edge_index = torch.tensor(
                new_edges,
                dtype=torch.long,
                device=problem.linear.device,
            ).t()
            edge_weight = torch.stack(new_weights)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=problem.linear.device)
            edge_weight = torch.empty((0,), dtype=problem.linear.dtype, device=problem.linear.device)

        subproblem = QUBOProblem.from_terms(
            num_variables=int(indices.numel()),
            linear=problem.linear[indices].clone(),
            edge_index=edge_index,
            edge_weight=edge_weight,
            constant=0.0,
            coalesce=True,
        )
        subproblems.append((subproblem, indices))
    return subproblems


def componentwise_qaoa_resource_summary(problem, layers=(1, 2, 3), gpu_memory_gb=12.0):
    """Summarize QAOA resources when disconnected components are solved separately."""

    if problem is None:
        return {
            "num_components": 0,
            "max_component_variables": 0,
            "max_component_edges": 0,
            "component_stats_preview": [],
            "qaoa_limits_largest_component": {
                f"p{layer}": qaoa_resource_summary(
                    0,
                    0,
                    layers=layer,
                    gpu_memory_gb=gpu_memory_gb,
                )
                for layer in layers
            },
        }

    components = qubo_component_subproblems(problem, include_isolated=True)
    component_stats = [
        {
            "variables": subproblem.num_variables,
            "edges": subproblem.num_edges,
            "indices_preview": indices.detach().cpu().tolist()[:20],
        }
        for subproblem, indices in components
    ]
    max_component_variables = max((item["variables"] for item in component_stats), default=0)
    max_component_edges = max((item["edges"] for item in component_stats), default=0)

    return {
        "num_components": len(component_stats),
        "max_component_variables": int(max_component_variables),
        "max_component_edges": int(max_component_edges),
        "component_stats_preview": component_stats[:10],
        "qaoa_limits_largest_component": {
            f"p{layer}": qaoa_resource_summary(
                max_component_variables,
                max_component_edges,
                layers=layer,
                gpu_memory_gb=gpu_memory_gb,
            )
            for layer in layers
        },
    }


def residual_qaoa_active_summary(problem, layers=(1, 2, 3), gpu_memory_gb=12.0):
    """Summarize the active QAOA subproblem after exact isolated-variable fixing."""

    reduced, kept_indices, isolated_mask, fixed_values = reduce_by_fixing_isolated_variables(problem)
    active_variables = 0 if reduced is None else reduced.num_variables
    active_edges = 0 if reduced is None else reduced.num_edges
    component_summary = componentwise_qaoa_resource_summary(
        reduced,
        layers=layers,
        gpu_memory_gb=gpu_memory_gb,
    )

    return {
        "isolated_variables_fixed_exactly": int(isolated_mask.sum().detach().cpu()),
        "active_variables_after_isolated_fixing": int(active_variables),
        "active_edges_after_isolated_fixing": int(active_edges),
        "active_indices_preview": kept_indices.detach().cpu().tolist()[:20],
        "componentwise_qaoa": component_summary,
        "qaoa_limits_after_isolated_fixing": {
            f"p{layer}": qaoa_resource_summary(
                active_variables,
                active_edges,
                layers=layer,
                gpu_memory_gb=gpu_memory_gb,
            )
            for layer in layers
        },
    }
