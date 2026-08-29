"""Geometry ingestion and deterministic profile discretization."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_profile(path: str | Path) -> np.ndarray:
    """Load a two-column one-wavelength profile and normalize its origin."""
    points = np.loadtxt(path, skiprows=1, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Profile file must contain exactly two numeric columns")
    points = points[np.argsort(points[:, 0])]
    points[:, 0] -= points[:, 0].min()
    points[:, 1] -= points[:, 1].min()
    return points


def repeat_profile(
    profile: np.ndarray,
    repetitions: int,
    elements_per_wavelength: int,
    target_height: float | None = None,
) -> np.ndarray:
    """Resample and repeat a periodic profile without duplicate interface nodes."""
    if repetitions < 1 or elements_per_wavelength < 4:
        raise ValueError("At least one repetition and four elements per wavelength are required")
    pitch = float(profile[:, 0].max() - profile[:, 0].min())
    height = float(profile[:, 1].max() - profile[:, 1].min())
    if pitch <= 0 or height <= 0:
        raise ValueError("Profile pitch and height must be positive")
    scale_y = 1.0 if target_height is None else float(target_height) / height
    local_x = np.linspace(0.0, pitch, elements_per_wavelength + 1)
    local_y = np.interp(local_x, profile[:, 0], profile[:, 1]) * scale_y
    blocks = []
    for repetition in range(repetitions):
        start = 0 if repetition == 0 else 1
        blocks.append(
            np.column_stack(
                [local_x[start:] + repetition * pitch, local_y[start:]]
            )
        )
    return np.vstack(blocks)
