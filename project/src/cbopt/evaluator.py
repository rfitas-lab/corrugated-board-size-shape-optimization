"""Fast numerical reproduction of the public NURBS corrugated-board model.

The public implementation evaluates a quadratic rational B-spline, rotates the
paper compliance along the local tangent, homogenizes the section, and returns
inverse geometric inertia, section area, and an effective transverse modulus.
This module retains those definitions while making units and constraints
explicit.  It also reports both the legacy raw quantities and normalized
quantities suitable for unambiguous manuscript labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class BoardEvaluation:
    inverse_inertia_scaled: float
    section_area_m2: float
    area_per_wavelength_mm: float
    inertia_per_wavelength_mm3: float
    effective_transverse_modulus_mpa: float
    minimum_radius_1_mm: float
    minimum_radius_2_mm: float
    feasible: bool
    arc_length_mm: float
    wavelength_mm: float
    amplitude_mm: float

    @property
    def objectives(self) -> np.ndarray:
        """Legacy minimization pair used in the archived adapted optimizer."""

        return np.array(
            [self.inverse_inertia_scaled, self.section_area_m2],
            dtype=float,
        )


def _clamped_uniform_knots(n_control: int, degree: int) -> np.ndarray:
    n_knots = n_control + degree + 1
    n_grid = n_knots - 2 * (degree + 1) + 2
    interior = np.linspace(0.0, 1.0, n_grid)[1:-1]
    return np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]


def _control_points(design: np.ndarray) -> np.ndarray:
    d = np.asarray(design[:5], dtype=float)
    if np.any(d <= 0.0) or not np.isfinite(d).all():
        raise ValueError("The five spacing variables must be finite and positive.")
    d = d / d.sum()
    x = np.array(
        [
            0.0,
            d[0],
            d[:2].sum(),
            d[:3].sum(),
            d[:4].sum(),
            1.0 + d[0],
            1.0 + d[:2].sum(),
            1.0 + d[:3].sum(),
            1.0 + d[:4].sum(),
            2.0,
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    return np.column_stack((x, y))


def nurbs_profile(
    design: np.ndarray,
    *,
    sample_size: int = 800,
    wavelength_mm: float = 5.65,
    amplitude_mm: float = 2.65,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Return the full two-period rational quadratic B-spline in millimetres."""

    design = np.asarray(design, dtype=float)
    if design.shape != (7,):
        raise ValueError("Expected seven design variables: five spacings and two weights.")
    cp = _control_points(design)
    r1 = 0.0103 * np.exp(9.17 * design[5]) + 0.1
    r2 = 0.0103 * np.exp(9.17 * design[6]) + 0.1
    weights = np.array([1.0, r1, r1, r2, r2, r1, r1, r2, r2, 1.0])
    knots = _clamped_uniform_knots(len(cp), 2)
    u = np.linspace(0.0, 1.0, sample_size)
    numerator = BSpline(knots, cp * weights[:, None], 2)(u)
    denominator = BSpline(knots, weights, 2)(u)
    pts = numerator / denominator[:, None]

    d = design[:5] / design[:5].sum()
    t1 = int(((d[0] + d[1] + d[2] / 2.0) / 2.0) * sample_size)
    tm = int(0.5 * sample_size)
    t3 = int(((1.0 + d[0] + d[1] + d[2] / 2.0) / 2.0) * sample_size)
    t1 = np.clip(t1, 1, sample_size - 4)
    tm = np.clip(tm, t1 + 2, sample_size - 2)
    t3 = np.clip(t3, tm + 2, sample_size - 1)
    dx_reference = abs(pts[t3, 0] - pts[t1, 0])
    height_reference = np.ptp(pts[t1:t3, 1])
    if dx_reference <= 0 or height_reference <= 0:
        raise ValueError("Degenerate NURBS profile.")
    x = pts[:, 0] * wavelength_mm / dx_reference
    y = pts[:, 1] * amplitude_mm / height_reference
    return x, y, (int(t1), int(tm), int(t3))


def _zone_radius_mm(x: np.ndarray, y: np.ndarray, sl: slice) -> float:
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-14)
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    # The public implementation used the raw minimum.  Light smoothing makes
    # the result substantially less sensitive to sampling without changing
    # the intended manufacturing-radius screen.
    curvature = gaussian_filter1d(curvature, sigma=max(1.0, len(x) / 800.0))
    local = curvature[sl]
    local = local[np.isfinite(local) & (local > 1e-9)]
    return float(1.0 / np.max(local)) if local.size else float("inf")


def _material_compliance() -> np.ndarray:
    e1 = 1.0e9
    e2 = e1 / 2.0
    e3 = e1 / 190.0
    g12 = 0.387 * np.sqrt(e1 * e2)
    g13 = e1 / 55.0
    g23 = e2 / 35.0
    nu12 = 0.293 * np.sqrt(e1 / e2)
    nu13 = 0.001
    nu23 = 0.001
    return np.array(
        [
            [1 / e1, -nu12 / e1, -nu13 / e1, 0, 0, 0],
            [-nu12 / e1, 1 / e2, -nu23 / e2, 0, 0, 0],
            [-nu13 / e1, -nu23 / e2, 1 / e3, 0, 0, 0],
            [0, 0, 0, 1 / g12, 0, 0],
            [0, 0, 0, 0, 1 / g13, 0],
            [0, 0, 0, 0, 0, 1 / g23],
        ],
        dtype=float,
    )


def _rotated_stiffness(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.cos(theta)
    s = np.sin(theta)
    n = len(theta)
    te = np.zeros((n, 6, 6))
    ts = np.zeros((n, 6, 6))
    te[:, 0, 0], te[:, 0, 2], te[:, 0, 4] = c * c, s * s, -s * c
    te[:, 1, 1] = 1
    te[:, 2, 0], te[:, 2, 2], te[:, 2, 4] = s * s, c * c, s * c
    te[:, 3, 3], te[:, 3, 5] = c, -s
    te[:, 4, 0], te[:, 4, 2], te[:, 4, 4] = 2 * s * c, -2 * s * c, c * c - s * s
    te[:, 5, 3], te[:, 5, 5] = -s, c
    ts[:, 0, 0], ts[:, 0, 2], ts[:, 0, 4] = c * c, s * s, 2 * s * c
    ts[:, 1, 1] = 1
    ts[:, 2, 0], ts[:, 2, 2], ts[:, 2, 4] = s * s, c * c, -2 * s * c
    ts[:, 3, 3], ts[:, 3, 5] = c, s
    ts[:, 4, 0], ts[:, 4, 2], ts[:, 4, 4] = -s * c, s * c, c * c - s * s
    ts[:, 5, 3], ts[:, 5, 5] = -s, c
    cxyz = te @ _material_compliance()[None, :, :] @ ts
    q = np.linalg.inv(cxyz[:, :3, :3])
    g = np.linalg.inv(cxyz[:, 3:, 3:])[:, :2, :2]
    return q, g


def evaluate_design(
    design: np.ndarray,
    *,
    sample_size: int = 800,
    wavelength_mm: float = 5.65,
    amplitude_mm: float = 2.65,
    thickness_mm: float = 0.2,
    radius_limit_mm: float = 0.9,
) -> BoardEvaluation:
    """Evaluate one seven-variable design using the public model definitions."""

    x_all, y_all, (t1, tm, t3) = nurbs_profile(
        design,
        sample_size=sample_size,
        wavelength_mm=wavelength_mm,
        amplitude_mm=amplitude_mm,
    )
    r1 = _zone_radius_mm(x_all, y_all, slice(t1, tm + 1))
    r2 = _zone_radius_mm(x_all, y_all, slice(tm, t3 + 1))
    x_mm = x_all[t1:t3]
    y_mm = y_all[t1:t3]
    order = np.argsort(x_mm)
    x_mm, y_mm = x_mm[order], y_mm[order]
    # Duplicate abscissae can occur at nearly vertical NURBS portions.
    keep = np.r_[True, np.diff(x_mm) > 1e-10]
    x_mm, y_mm = x_mm[keep], y_mm[keep]
    if len(x_mm) < 20:
        raise ValueError("Insufficient distinct profile points.")

    x = x_mm * 1e-3
    y = y_mm * 1e-3
    wavelength = wavelength_mm * 1e-3
    thickness = thickness_mm * 1e-3
    theta = np.arctan(np.gradient(y, x))
    cos_theta = np.maximum(np.abs(np.cos(theta)), 1e-6)
    tv = thickness / cos_theta
    d_section = y * y * tv + tv**3 / 12.0
    q, g = _rotated_stiffness(theta)
    a = np.trapezoid(tv[:, None, None] * q, x=x, axis=0) / wavelength
    d = np.trapezoid(d_section[:, None, None] * q, x=x, axis=0) / wavelength
    f = np.trapezoid(tv[:, None, None] * g, x=x, axis=0) / wavelength
    ai = np.linalg.inv(a)
    di = np.linalg.inv(d)
    _ = np.linalg.inv(f)  # retained as a singularity/consistency check
    effective_thickness = np.sqrt(12.0 * np.trace(d) / np.trace(a))
    ez = 12.0 / (effective_thickness**3 * di[2, 2])
    e3 = 1.0e9 / 190.0
    ez_eff = (ez * effective_thickness + e3 * 2 * thickness) / (
        effective_thickness + 2 * thickness
    )

    ds = np.hypot(np.diff(x), np.diff(y))
    arc_length = float(ds.sum())
    projected = float(np.ptp(x))
    area_core = float(ds.sum() * thickness)
    y_bar_core = float(np.sum(y[:-1] * ds * thickness) / area_core)
    y_top = float(y.max() + thickness)
    y_bottom = float(y.min() - thickness)
    area_liner = projected * thickness
    area_total = area_core + 2.0 * area_liner
    y_bar = (area_core * y_bar_core + area_liner * (y_top + y_bottom)) / area_total
    inertia_core = float(np.sum((y[:-1] - y_bar) ** 2 * ds * thickness))
    inertia_top = projected * thickness**3 / 12.0 + projected * thickness * (y_top - y_bar) ** 2
    inertia_bottom = projected * thickness**3 / 12.0 + projected * thickness * (y_bottom - y_bar) ** 2
    inertia_per_length = (inertia_core + inertia_top + inertia_bottom) / projected
    inverse_scaled = 1.0e-6 / inertia_per_length

    return BoardEvaluation(
        inverse_inertia_scaled=float(inverse_scaled),
        section_area_m2=float(area_total),
        area_per_wavelength_mm=float(area_total / projected * 1e3),
        inertia_per_wavelength_mm3=float(inertia_per_length * 1e9),
        effective_transverse_modulus_mpa=float(ez_eff / 1e6),
        minimum_radius_1_mm=r1,
        minimum_radius_2_mm=r2,
        feasible=bool(r1 >= radius_limit_mm and r2 >= radius_limit_mm),
        arc_length_mm=float(arc_length * 1e3),
        wavelength_mm=float(wavelength_mm),
        amplitude_mm=float(amplitude_mm),
    )
