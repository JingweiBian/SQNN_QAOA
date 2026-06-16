# -*- coding: utf-8 -*-

"""Sparse QUBO representation used by SQNN warm-start models."""

from dataclasses import dataclass

import torch


NODE_FEATURE_DIM = 5
QUANTUM_NODE_FEATURE_DIM = 3
EDGE_FEATURE_DIM = 7


def _empty_edges(device=None):
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _empty_weights(dtype=None, device=None):
    return torch.empty((0,), dtype=dtype or torch.get_default_dtype(), device=device)


def _coalesce_quadratic_terms(num_variables, linear, edge_index, edge_weight):
    if edge_index.numel() == 0:
        return linear, _empty_edges(linear.device), _empty_weights(linear.dtype, linear.device)

    src = edge_index[0].to(dtype=torch.long, device=linear.device)
    dst = edge_index[1].to(dtype=torch.long, device=linear.device)
    weight = edge_weight.to(dtype=linear.dtype, device=linear.device)

    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)

    diagonal_mask = lo == hi
    if diagonal_mask.any():
        linear = linear.clone()
        linear.index_add_(0, lo[diagonal_mask], weight[diagonal_mask])

    off_mask = ~diagonal_mask
    if not off_mask.any():
        return linear, _empty_edges(linear.device), _empty_weights(linear.dtype, linear.device)

    lo = lo[off_mask]
    hi = hi[off_mask]
    weight = weight[off_mask]

    keys = lo * int(num_variables) + hi
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    coalesced_weight = torch.zeros(
        unique_keys.numel(),
        dtype=linear.dtype,
        device=linear.device,
    )
    coalesced_weight.index_add_(0, inverse, weight)

    keep = coalesced_weight != 0
    unique_keys = unique_keys[keep]
    coalesced_weight = coalesced_weight[keep]

    coalesced_src = torch.div(unique_keys, int(num_variables), rounding_mode="floor")
    coalesced_dst = unique_keys.remainder(int(num_variables))
    coalesced_edges = torch.stack((coalesced_src, coalesced_dst), dim=0)
    return linear, coalesced_edges, coalesced_weight


@dataclass
class QUBOProblem:
    """A sparse QUBO objective.

    The represented objective is:

        E(x) = constant + sum_i linear[i] * x_i
             + sum_k edge_weight[k] * x[src_k] * x[dst_k]

    with binary variables x_i in {0, 1}.  Edges are stored once, canonically
    with src < dst.  Use ``directed_edges`` when building graph message passing.
    """

    num_variables: int
    linear: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    constant: torch.Tensor | float = 0.0

    def __post_init__(self):
        self.num_variables = int(self.num_variables)
        if self.num_variables <= 0:
            raise ValueError("num_variables must be positive")

        self.linear = torch.as_tensor(self.linear)
        if self.linear.ndim != 1:
            raise ValueError("linear must be a 1D tensor")
        if self.linear.numel() != self.num_variables:
            raise ValueError(
                f"linear has {self.linear.numel()} entries, "
                f"expected {self.num_variables}"
            )
        if not torch.is_floating_point(self.linear):
            self.linear = self.linear.to(dtype=torch.get_default_dtype())

        self.edge_index = torch.as_tensor(
            self.edge_index,
            dtype=torch.long,
            device=self.linear.device,
        )
        if (
            self.edge_index.ndim == 2
            and self.edge_index.shape[0] != 2
            and self.edge_index.shape[1] == 2
        ):
            self.edge_index = self.edge_index.t().contiguous()
        if self.edge_index.numel() == 0:
            self.edge_index = _empty_edges(self.linear.device)
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        self.edge_weight = torch.as_tensor(
            self.edge_weight,
            dtype=self.linear.dtype,
            device=self.linear.device,
        )
        if self.edge_weight.ndim != 1:
            raise ValueError("edge_weight must be a 1D tensor")
        if self.edge_weight.numel() != self.edge_index.shape[1]:
            raise ValueError("edge_weight length must match edge_index columns")
        if self.edge_index.numel() and (
            self.edge_index.min() < 0 or self.edge_index.max() >= self.num_variables
        ):
            raise ValueError("edge_index contains an out-of-range variable index")

        self.constant = torch.as_tensor(
            self.constant,
            dtype=self.linear.dtype,
            device=self.linear.device,
        )

    @classmethod
    def from_terms(
        cls,
        num_variables,
        linear=None,
        edge_index=None,
        edge_weight=None,
        constant=0.0,
        coalesce=True,
        dtype=None,
        device=None,
    ):
        dtype = dtype or torch.get_default_dtype()
        linear_tensor = (
            torch.zeros(int(num_variables), dtype=dtype, device=device)
            if linear is None
            else torch.as_tensor(linear, dtype=dtype, device=device)
        )

        if edge_index is None:
            edge_index_tensor = _empty_edges(linear_tensor.device)
            edge_weight_tensor = _empty_weights(linear_tensor.dtype, linear_tensor.device)
        else:
            if edge_weight is None:
                raise ValueError("edge_weight is required when edge_index is provided")
            edge_index_tensor = torch.as_tensor(
                edge_index,
                dtype=torch.long,
                device=linear_tensor.device,
            )
            if (
                edge_index_tensor.ndim == 2
                and edge_index_tensor.shape[0] != 2
                and edge_index_tensor.shape[1] == 2
            ):
                edge_index_tensor = edge_index_tensor.t().contiguous()
            edge_weight_tensor = torch.as_tensor(
                edge_weight,
                dtype=linear_tensor.dtype,
                device=linear_tensor.device,
            )

        if coalesce:
            linear_tensor, edge_index_tensor, edge_weight_tensor = _coalesce_quadratic_terms(
                int(num_variables),
                linear_tensor,
                edge_index_tensor,
                edge_weight_tensor,
            )

        return cls(
            num_variables=int(num_variables),
            linear=linear_tensor,
            edge_index=edge_index_tensor,
            edge_weight=edge_weight_tensor,
            constant=constant,
        )

    @classmethod
    def from_dense(cls, matrix, convention="upper", zero_tol=0.0):
        """Build a QUBO from a dense square matrix.

        ``convention="upper"`` interprets the strict upper triangle as the
        pair coefficients in E(x). ``convention="matrix"`` interprets the
        dense matrix as x^T Q x, so pair coefficients become Q_ij + Q_ji.
        """

        q = torch.as_tensor(matrix)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise ValueError("matrix must be square")
        if not torch.is_floating_point(q):
            q = q.to(dtype=torch.get_default_dtype())

        n = q.shape[0]
        linear = torch.diagonal(q).clone()
        if convention == "upper":
            pair_matrix = torch.triu(q, diagonal=1)
        elif convention == "matrix":
            pair_matrix = torch.triu(q + q.transpose(0, 1), diagonal=1)
        else:
            raise ValueError("convention must be 'upper' or 'matrix'")

        if zero_tol > 0:
            edge_positions = (pair_matrix.abs() > zero_tol).nonzero(as_tuple=False)
        else:
            edge_positions = (pair_matrix != 0).nonzero(as_tuple=False)

        if edge_positions.numel() == 0:
            edge_index = _empty_edges(q.device)
            edge_weight = _empty_weights(q.dtype, q.device)
        else:
            edge_index = edge_positions.t().contiguous()
            edge_weight = pair_matrix[edge_index[0], edge_index[1]]

        return cls.from_terms(
            num_variables=n,
            linear=linear,
            edge_index=edge_index,
            edge_weight=edge_weight,
            dtype=q.dtype,
            device=q.device,
        )

    @property
    def num_edges(self):
        return int(self.edge_weight.numel())

    def to(self, *args, **kwargs):
        linear = self.linear.to(*args, **kwargs)
        device = linear.device
        return QUBOProblem(
            num_variables=self.num_variables,
            linear=linear,
            edge_index=self.edge_index.to(device=device),
            edge_weight=self.edge_weight.to(device=device, dtype=linear.dtype),
            constant=self.constant.to(device=device, dtype=linear.dtype),
        )

    def coefficient_scale(self, eps=1e-12):
        values = [self.linear.abs().max()]
        if self.edge_weight.numel():
            values.append(self.edge_weight.abs().max())
        scale = torch.stack(values).max()
        return scale.clamp_min(float(eps))

    def directed_edges(self):
        if self.edge_index.numel() == 0:
            return _empty_edges(self.linear.device), _empty_weights(
                self.linear.dtype,
                self.linear.device,
            )
        directed_index = torch.cat((self.edge_index, self.edge_index.flip(0)), dim=1)
        directed_weight = torch.cat((self.edge_weight, self.edge_weight), dim=0)
        return directed_index, directed_weight

    def node_degrees(self, weighted=False, absolute=True):
        degree = torch.zeros(
            self.num_variables,
            dtype=self.linear.dtype,
            device=self.linear.device,
        )
        if self.edge_index.numel() == 0:
            return degree

        src, dst = self.edge_index
        if weighted:
            values = self.edge_weight.abs() if absolute else self.edge_weight
        else:
            values = torch.ones_like(self.edge_weight)
        degree.index_add_(0, src, values)
        degree.index_add_(0, dst, values)
        return degree

    def node_features(self, normalize=True):
        """Return per-variable features for SQNN warm-start models.

        Columns are:
            linear, abs(linear), signed incident sum, abs incident sum, degree
        """

        signed_incident = self.node_degrees(weighted=True, absolute=False)
        abs_incident = self.node_degrees(weighted=True, absolute=True)
        degree = self.node_degrees(weighted=False)

        scale = self.coefficient_scale() if normalize else self.linear.new_tensor(1.0)
        degree_scale = degree.max().clamp_min(1.0) if normalize else self.linear.new_tensor(1.0)

        return torch.stack(
            (
                self.linear / scale,
                self.linear.abs() / scale,
                signed_incident / scale,
                abs_incident / scale,
                degree / degree_scale,
            ),
            dim=-1,
        )

    def quantum_node_features(self, normalize=True):
        """Return SQNN-native three-basis node features.

        Columns are:
            linear, signed incident sum, abs incident sum

        These three values fit one multi-basis SQNN neuron without relying on
        an MLP to compress a wider feature vector.
        """

        signed_incident = self.node_degrees(weighted=True, absolute=False)
        abs_incident = self.node_degrees(weighted=True, absolute=True)
        scale = self.coefficient_scale() if normalize else self.linear.new_tensor(1.0)

        return torch.stack(
            (
                self.linear / scale,
                signed_incident / scale,
                abs_incident / scale,
            ),
            dim=-1,
        )

    def directed_edge_features(self, normalize=True):
        """Return features for both directions of every QUBO interaction.

        Columns are:
            weight, abs(weight), sign(weight), src linear, dst linear,
            src degree, dst degree
        """

        directed_index, directed_weight = self.directed_edges()
        if directed_weight.numel() == 0:
            return torch.empty(
                (0, EDGE_FEATURE_DIM),
                dtype=self.linear.dtype,
                device=self.linear.device,
            )

        src, dst = directed_index
        degree = self.node_degrees(weighted=False)
        scale = self.coefficient_scale() if normalize else self.linear.new_tensor(1.0)
        degree_scale = degree.max().clamp_min(1.0) if normalize else self.linear.new_tensor(1.0)

        return torch.stack(
            (
                directed_weight / scale,
                directed_weight.abs() / scale,
                torch.sign(directed_weight),
                self.linear[src] / scale,
                self.linear[dst] / scale,
                degree[src] / degree_scale,
                degree[dst] / degree_scale,
            ),
            dim=-1,
        )

    def energy(self, assignments):
        x = torch.as_tensor(assignments, dtype=self.linear.dtype, device=self.linear.device)
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != self.num_variables:
            raise ValueError(
                f"assignments must have shape [num_variables] or [batch, num_variables], "
                f"got {tuple(x.shape)}"
            )

        energy = x @ self.linear
        if self.edge_weight.numel():
            src, dst = self.edge_index
            energy = energy + (x[:, src] * x[:, dst] * self.edge_weight).sum(dim=-1)
        energy = energy + self.constant
        return energy.squeeze(0) if squeeze else energy

    def expected_energy(self, probabilities):
        p = torch.as_tensor(
            probabilities,
            dtype=self.linear.dtype,
            device=self.linear.device,
        )
        p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if p.ndim == 2 and p.shape[-1] == 1:
            p = p.squeeze(-1)

        squeeze = p.ndim == 1
        if squeeze:
            p = p.unsqueeze(0)
        if p.ndim != 2 or p.shape[-1] != self.num_variables:
            raise ValueError(
                f"probabilities must have shape [num_variables] or [batch, num_variables], "
                f"got {tuple(p.shape)}"
            )

        energy = p @ self.linear
        if self.edge_weight.numel():
            src, dst = self.edge_index
            energy = energy + (p[:, src] * p[:, dst] * self.edge_weight).sum(dim=-1)
        energy = energy + self.constant
        return energy.squeeze(0) if squeeze else energy

    def reduce_by_fixed_assignments(self, fixed_mask, fixed_values):
        """Return the remaining QUBO after fixing a subset of variables.

        ``fixed_mask`` is a boolean vector of length n. ``fixed_values`` can be
        either a length-n vector or a vector containing only the fixed values.
        The returned tuple is ``(reduced_problem, free_indices)``.
        """

        mask = torch.as_tensor(fixed_mask, dtype=torch.bool, device=self.linear.device)
        if mask.ndim != 1 or mask.numel() != self.num_variables:
            raise ValueError("fixed_mask must be a boolean vector of length num_variables")

        values = torch.as_tensor(
            fixed_values,
            dtype=self.linear.dtype,
            device=self.linear.device,
        )
        if values.ndim != 1:
            raise ValueError("fixed_values must be a 1D tensor")
        if values.numel() == self.num_variables:
            full_values = values
        elif values.numel() == int(mask.sum().item()):
            full_values = torch.zeros_like(self.linear)
            full_values[mask] = values
        else:
            raise ValueError(
                "fixed_values must have length num_variables or fixed_mask.sum()"
            )

        free_indices = (~mask).nonzero(as_tuple=False).flatten()
        old_to_new = torch.full(
            (self.num_variables,),
            -1,
            dtype=torch.long,
            device=self.linear.device,
        )
        old_to_new[free_indices] = torch.arange(
            free_indices.numel(),
            dtype=torch.long,
            device=self.linear.device,
        )

        reduced_linear = self.linear[free_indices].clone()
        reduced_constant = self.constant + (self.linear[mask] * full_values[mask]).sum()
        new_edges = []
        new_weights = []

        if self.edge_weight.numel():
            for edge_pos in range(self.edge_weight.numel()):
                i = self.edge_index[0, edge_pos]
                j = self.edge_index[1, edge_pos]
                w = self.edge_weight[edge_pos]
                i_fixed = bool(mask[i].item())
                j_fixed = bool(mask[j].item())

                if i_fixed and j_fixed:
                    reduced_constant = reduced_constant + w * full_values[i] * full_values[j]
                elif i_fixed and not j_fixed:
                    reduced_linear[old_to_new[j]] = (
                        reduced_linear[old_to_new[j]] + w * full_values[i]
                    )
                elif j_fixed and not i_fixed:
                    reduced_linear[old_to_new[i]] = (
                        reduced_linear[old_to_new[i]] + w * full_values[j]
                    )
                else:
                    new_edges.append([int(old_to_new[i].item()), int(old_to_new[j].item())])
                    new_weights.append(w)

        if new_edges:
            edge_index = torch.tensor(
                new_edges,
                dtype=torch.long,
                device=self.linear.device,
            ).t()
            edge_weight = torch.stack(new_weights)
        else:
            edge_index = _empty_edges(self.linear.device)
            edge_weight = _empty_weights(self.linear.dtype, self.linear.device)

        return (
            QUBOProblem.from_terms(
                num_variables=int(free_indices.numel()),
                linear=reduced_linear,
                edge_index=edge_index,
                edge_weight=edge_weight,
                constant=reduced_constant,
                coalesce=True,
            ),
            free_indices,
        )
