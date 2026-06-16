# -*- coding: utf-8 -*-

"""Utilities that convert SQNN warm-start probabilities into QAOA inputs."""

import torch


def qaoa_ry_angles_from_probabilities(probabilities, eps=1e-7):
    """Return angles theta where RY(theta)|0> has probability p of |1>."""

    p = torch.nan_to_num(
        torch.as_tensor(probabilities),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(float(eps), 1.0 - float(eps))
    return 2.0 * torch.asin(torch.sqrt(p))


def probabilities_from_qaoa_ry_angles(angles):
    """Inverse of qaoa_ry_angles_from_probabilities."""

    theta = torch.as_tensor(angles)
    return torch.sin(theta * 0.5).square().clamp(0.0, 1.0)


def calibrate_probabilities_with_assignment(
    probabilities,
    assignment,
    min_probability=0.01,
    min_confidence=0.0,
):
    """Use SQNN confidence magnitudes with signs from a repaired assignment.

    This keeps ``abs(p - 0.5)`` as the confidence signal, but replaces the side
    of 0.5 with the provided binary assignment. It is useful when raw SQNN
    signs are overconfident but local repair finds a better discrete solution.
    """

    p = torch.nan_to_num(
        torch.as_tensor(probabilities),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    values = torch.as_tensor(assignment, device=p.device)
    values = (values >= 0.5).to(dtype=p.dtype)

    max_confidence = 0.5 - float(min_probability)
    confidence = (p - 0.5).abs().clamp(float(min_confidence), max_confidence)
    calibrated = torch.where(values > 0.5, 0.5 + confidence, 0.5 - confidence)
    return calibrated.clamp(float(min_probability), 1.0 - float(min_probability))
