"""Paper A v2 geometry adapter of the archived compression protocol.
Derived from project/src/cbopt/mechanical_evaluator.py (MIT repository).
The constitutive model, attachment constraints, continuation, and acceptance
logic are unchanged. The solve entry point accepts an analytic Curve object.
The original source is preserved verbatim in its original location.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from time import perf_counter

import numpy as np

from cbfem import BeamMaterial, BeamSection, CorotationalBeamModel
from cbopt.evaluator import evaluate_design, nurbs_profile
from geometry_v2 import Curve, metrics as geometry_metrics


@dataclass(frozen=True)
class CompressionProtocol:
    elastic_modulus_MPa: float = 2899.0
    shear_ratio: float = 1.0 / 55.0
    yield_stress_MPa: float = 60.0
    hardening_ratio: float = 0.02
    width_mm: float = 25.0
    strains: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
    stress_power: int = 8
    projected_gradient_force_fraction: float = 5.0e-4


PROTOCOL_VERSION = "paper-a-v2-generic-analytic-geometry-attached-contact-v1"


def protocol_fingerprint(
    elements_per_wavelength: int,
    protocol: CompressionProtocol = CompressionProtocol(),
) -> str:
    """Return a stable identifier for cached mechanical evidence."""

    payload = {
        "version": PROTOCOL_VERSION,
        "elements_per_wavelength": int(elements_per_wavelength),
        "protocol": asdict(protocol),
        "contact": (
            "surface offset t/2; exact trough perfectly attached to lower rigid liner; "
            "tied periodic crowns perfectly attached to upper rigid liner; remaining "
            "nodes use unilateral bounds; surface reactions are signed root sums"
        ),
        "mesh": "arc-length segments split at exact periodic crowns and trough",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def paper_a_profile_nodes(
    position: np.ndarray,
    elements_per_wavelength: int,
) -> tuple[np.ndarray, float]:
    """Return one period resampled in curve order by arc length.

    Arc-length sampling retains near-vertical profile portions that would be
    lost by sorting and interpolating on a uniform x grid.
    """

    position = np.asarray(position, dtype=float)
    if position.shape != (10,):
        raise ValueError("Expected seven shape and three size variables")
    if elements_per_wavelength < 6:
        raise ValueError("At least six elements per wavelength are required")
    pitch, height = map(float, position[7:9])
    x, y, (start, _, stop) = nurbs_profile(
        position[:7],
        sample_size=max(1600, 80 * elements_per_wavelength),
        wavelength_mm=pitch,
        amplitude_mm=height,
    )
    curve = np.column_stack((x[start : stop + 1], y[start : stop + 1]))
    curve[:, 0] -= curve[0, 0]
    curve[:, 1] -= curve[:, 1].min()
    # The exact u=1/4 to u=3/4 extraction is a C1-periodic crown-to-crown cell.
    # Scale the same dense curve used by the geometric evaluator to the declared
    # pitch and height; the endpoint assignment removes only roundoff.
    curve[:, 0] *= pitch / max(curve[-1, 0], 1.0e-12)
    curve[:, 1] *= height / max(np.ptp(curve[:, 1]), 1.0e-12)
    periodic_gap = abs(float(curve[-1, 1] - curve[0, 1]))
    if periodic_gap > 1.0e-10 * max(height, 1.0):
        raise ValueError(f"Periodic profile ordinate mismatch: {periodic_gap:.3e} mm")
    endpoint_y = height
    curve[0, 1] = curve[-1, 1] = endpoint_y
    increments = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arclength = np.r_[0.0, np.cumsum(increments)]

    # Contact-conforming partition: the scaled dense curve contains points at
    # exactly y=0 and y=H.  Make both points nodes at every mesh, then allocate
    # the remaining elements to the intervening arc-length segments.  This
    # removes geometry-dependent free platen travel from coarse meshes.
    critical = np.unique(
        np.asarray([0, int(np.argmin(curve[:, 1])), int(np.argmax(curve[:, 1])), len(curve) - 1])
    )
    critical.sort()
    segment_lengths = np.diff(arclength[critical])
    if np.any(segment_lengths <= 0.0) or elements_per_wavelength < len(segment_lengths):
        raise ValueError("The requested mesh cannot resolve all contact-critical segments")
    allocation = np.ones(len(segment_lengths), dtype=int)
    remaining = int(elements_per_wavelength - allocation.sum())
    if remaining:
        ideal = remaining * segment_lengths / segment_lengths.sum()
        extra = np.floor(ideal).astype(int)
        allocation += extra
        for index in np.argsort(-(ideal - extra))[: remaining - int(extra.sum())]:
            allocation[index] += 1
    target_parts = []
    for segment, count in enumerate(allocation):
        values = np.linspace(
            arclength[critical[segment]],
            arclength[critical[segment + 1]],
            int(count) + 1,
        )
        target_parts.append(values if segment == 0 else values[1:])
    target = np.concatenate(target_parts)
    nodes = np.column_stack(
        (np.interp(target, arclength, curve[:, 0]), np.interp(target, arclength, curve[:, 1]))
    )
    nodes[0, 0], nodes[-1, 0] = 0.0, pitch
    nodes[0, 1] = nodes[-1, 1] = endpoint_y
    nodes[int(np.argmin(nodes[:, 1])), 1] = 0.0
    nodes[int(np.argmax(nodes[:, 1])), 1] = height
    return nodes, float(arclength[-1])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0.0:
        return float("nan")
    return float(np.interp(quantile * cumulative[-1], cumulative, values))


def dimensionless_geometry(position: np.ndarray, radius_limit_mm: float = 0.9) -> dict[str, float]:
    """Evaluate exact geometric quantities and dimensionless input groups."""

    position = np.asarray(position, dtype=float)
    pitch, height, thickness = map(float, position[7:10])
    detail = evaluate_design(
        position[:7],
        sample_size=800,
        wavelength_mm=pitch,
        amplitude_mm=height,
        thickness_mm=thickness,
        radius_limit_mm=radius_limit_mm,
    )
    minimum_radius = min(detail.minimum_radius_1_mm, detail.minimum_radius_2_mm)
    core_fraction = thickness * detail.arc_length_mm / (pitch * height)
    board_fraction = thickness * (detail.arc_length_mm + 2.0 * pitch) / (pitch * height)
    return {
        "pitch_mm": pitch,
        "height_mm": height,
        "thickness_mm": thickness,
        "arc_length_mm": detail.arc_length_mm,
        "radius_1_mm": detail.minimum_radius_1_mm,
        "radius_2_mm": detail.minimum_radius_2_mm,
        "radius_min_mm": minimum_radius,
        "aspect_ratio_H_over_P": height / pitch,
        "thickness_ratio_t_over_H": thickness / height,
        "thickness_ratio_t_over_P": thickness / pitch,
        "curvature_index_t_over_Rmin": thickness / minimum_radius,
        "core_material_fraction": core_fraction,
        "board_material_fraction": board_fraction,
        "geometry_feasible": float(detail.feasible),
    }


def surrogate_features(position: np.ndarray) -> np.ndarray:
    """Independent dimensionless geometry descriptor for the neural potential.

    The curvature index is deliberately excluded: it is a useful derived
    manufacturing diagnostic, but it is determined by the eight entries below
    and would otherwise reweight the trust-distance metric redundantly.
    """

    position = np.asarray(position, dtype=float)
    spacing = position[:5] / np.sum(position[:5])
    geometry = dimensionless_geometry(position)
    return np.r_[
        spacing[:4],
        position[5:7],
        geometry["aspect_ratio_H_over_P"],
        geometry["thickness_ratio_t_over_H"],
    ]


def solve_curve_path(
    curve: Curve,
    thickness: float,
    *,
    elements_per_wavelength: int = 12,
    protocol: CompressionProtocol = CompressionProtocol(),
    solver_max_iterations: int = 1800,
) -> tuple[list[dict[str, float | bool | str | np.ndarray]], dict[str, float | bool]]:
    """Solve a monotonic 0--target-strain compression path with continuation."""

    height = float(curve.height)
    thickness = float(thickness)
    nodes, discretized_arc = curve.nodes(elements_per_wavelength)
    model = CorotationalBeamModel(
        nodes,
        BeamMaterial(
            protocol.elastic_modulus_MPa,
            protocol.elastic_modulus_MPa * protocol.shear_ratio,
            protocol.yield_stress_MPa,
            protocol.hardening_ratio,
        ),
        BeamSection(protocol.width_mm, thickness),
    )
    last = len(nodes) - 1
    trough = int(np.argmin(nodes[:, 1]))
    tied_dofs = [(3 * last + 1, 1), (3 * last + 2, 2)]
    element_lengths = model.initial_lengths
    force_scale = protocol.yield_stress_MPa * protocol.width_mm * thickness
    optimizer_gradient_tolerance_N = max(
        1.0e-6,
        0.03 * protocol.projected_gradient_force_fraction * force_scale,
    )
    contact_offset = 0.5 * thickness
    lower_platen_surface = -contact_offset
    initial_upper_platen_surface = height + contact_offset
    lower_centerline_bound = lower_platen_surface + contact_offset
    previous = None
    previous_strain = 0.0
    states: list[dict[str, float | bool | str | np.ndarray]] = []
    start_time = perf_counter()

    def solve_increment(
        strain_value: float,
        initial_displacement: np.ndarray | None,
        *,
        subdivision: int,
        substep: int,
    ):
        upper_surface = initial_upper_platen_surface - height * float(strain_value)
        result = model.solve(
            {
                0: 0.0,
                1: -height * float(strain_value),
                3 * trough + 1: 0.0,
            },
            lower_y_mm=lower_centerline_bound,
            upper_y_mm=upper_surface - contact_offset,
            tied_dofs=tied_dofs,
            initial_displacement=initial_displacement,
            tolerance=optimizer_gradient_tolerance_N,
            max_iterations=solver_max_iterations,
        )
        normalized_gradient = float(result.gradient_norm) / force_scale
        fallback_normalized_gradient = (
            float(result.fallback_gradient_norm) / force_scale
            if np.isfinite(result.fallback_gradient_norm)
            else float("nan")
        )
        independent_kkt_confirmed = bool(
            result.fallback_used
            and result.fallback_success
            and np.isfinite(fallback_normalized_gradient)
            and fallback_normalized_gradient
            <= protocol.projected_gradient_force_fraction
        )
        accepted = bool(
            np.isfinite(result.potential_energy_Nmm)
            and normalized_gradient <= protocol.projected_gradient_force_fraction
            and result.success
        )
        diagnostic = {
            "strain": float(strain_value),
            "subdivision": int(subdivision),
            "substep": int(substep),
            "optimizer_method": result.optimizer_method,
            "selected_success": bool(result.success),
            "selected_message": str(result.message),
            "selected_iterations": int(result.iterations),
            "primary_success": bool(result.primary_success),
            "primary_message": str(result.primary_message),
            "primary_iterations": int(result.primary_iterations),
            "fallback_used": bool(result.fallback_used),
            "fallback_success": bool(result.fallback_success),
            "fallback_message": str(result.fallback_message),
            "fallback_iterations": int(result.fallback_iterations),
            "fallback_normalized_projected_gradient": fallback_normalized_gradient,
            "independent_kkt_confirmed": independent_kkt_confirmed,
            "projected_gradient_N": float(result.gradient_norm),
            "normalized_projected_gradient": normalized_gradient,
            "accepted": accepted,
        }
        return result, normalized_gradient, accepted, diagnostic

    for strain in protocol.strains:
        if np.isclose(strain, 0.0):
            states.append(
                {
                    "strain": 0.0,
                    "platen_travel_mm": 0.0,
                    "reaction_N": 0.0,
                    "lower_reaction_N": 0.0,
                    "potential_energy_Nmm": 0.0,
                    "stress_pnorm_MPa": 0.0,
                    "stress_q99_MPa": 0.0,
                    "maximum_stress_MPa": 0.0,
                    "yielded_length_fraction": 0.0,
                    "raw_solver_success": True,
                    "solver_accepted": True,
                    "success": True,
                    "gradient_norm": 0.0,
                    "normalized_projected_gradient": 0.0,
                    "iterations": 0.0,
                    "message": "undeformed reference",
                    "optimizer_method": "reference",
                    "primary_solver_success": True,
                    "fallback_used": False,
                    "independent_kkt_confirmed": False,
                    "subdivision_count": 1,
                    "solver_diagnostics": [],
                    "solver_diagnostics_json": "[]",
                    "lower_platen_surface_mm": lower_platen_surface,
                    "upper_platen_surface_mm": initial_upper_platen_surface,
                    "displacement": np.zeros(model.n_dof),
                    "element_stress_MPa": np.zeros(len(element_lengths)),
                }
            )
            continue
        upper_platen_surface = initial_upper_platen_surface - height * float(strain)
        attempts: list[dict[str, object]] = []
        result, normalized_gradient, accepted, diagnostic = solve_increment(
            float(strain), previous, subdivision=1, substep=1
        )
        attempts.append(diagnostic)
        selected_subdivision = 1
        # If the nominal continuation step does not satisfy both optimizer and
        # projected-KKT acceptance, restart from the previous accepted nominal
        # state with progressively finer load subdivision.
        if not accepted:
            best = (result, normalized_gradient, accepted, selected_subdivision)
            for divisions in (2, 4, 8):
                trial_previous = previous
                trial_result = None
                trial_gradient = float("inf")
                trial_accepted = False
                path_accepted = True
                substep_strains = np.linspace(
                    previous_strain, float(strain), divisions + 1
                )[1:]
                for substep_index, substep_strain in enumerate(substep_strains, start=1):
                    (
                        trial_result,
                        trial_gradient,
                        trial_accepted,
                        diagnostic,
                    ) = solve_increment(
                        float(substep_strain),
                        trial_previous,
                        subdivision=divisions,
                        substep=substep_index,
                    )
                    attempts.append(diagnostic)
                    trial_previous = trial_result.displacement
                    if not trial_accepted:
                        path_accepted = False
                        break
                reached_target = trial_result is not None and np.isclose(
                    float(diagnostic["strain"]), float(strain)
                )
                if reached_target and trial_gradient < best[1]:
                    best = (
                        trial_result,
                        trial_gradient,
                        bool(path_accepted and trial_accepted),
                        divisions,
                    )
                if reached_target and path_accepted and trial_accepted:
                    best = (trial_result, trial_gradient, True, divisions)
                    break
            result, normalized_gradient, accepted, selected_subdivision = best
        previous = result.displacement
        previous_strain = float(strain)
        stress_fields = np.vstack(
            [
                np.abs(values)
                for name, values in result.element_results.items()
                if name.startswith("stress_")
            ]
        )
        element_stress = np.max(stress_fields, axis=0)
        p = int(protocol.stress_power)
        stress_pnorm = float(
            (np.sum(element_lengths * element_stress**p) / np.sum(element_lengths))
            ** (1.0 / p)
        )
        stress_q99 = _weighted_quantile(element_stress, element_lengths, 0.99)
        yielded = float(
            np.sum(element_lengths[element_stress >= protocol.yield_stress_MPa])
            / np.sum(element_lengths)
        )
        normalized_gradient = float(result.gradient_norm) / force_scale
        accepted = bool(
            result.success
            and normalized_gradient <= protocol.projected_gradient_force_fraction
        )
        states.append(
            {
                "strain": float(strain),
                "platen_travel_mm": height * float(strain),
                "reaction_N": float(result.top_reaction_N),
                "lower_reaction_N": float(result.lower_reaction_N),
                "potential_energy_Nmm": float(result.potential_energy_Nmm),
                "stress_pnorm_MPa": stress_pnorm,
                "stress_q99_MPa": stress_q99,
                "maximum_stress_MPa": float(element_stress.max()),
                "yielded_length_fraction": yielded,
                "raw_solver_success": bool(result.success),
                "solver_accepted": accepted,
                "success": accepted,
                "gradient_norm": float(result.gradient_norm),
                "normalized_projected_gradient": normalized_gradient,
                "iterations": float(result.iterations),
                "message": str(result.message),
                "optimizer_method": str(result.optimizer_method),
                "primary_solver_success": bool(result.primary_success),
                "fallback_used": bool(result.fallback_used),
                "independent_kkt_confirmed": bool(
                    result.fallback_used
                    and result.fallback_success
                    and np.isfinite(result.fallback_gradient_norm)
                    and float(result.fallback_gradient_norm) / force_scale
                    <= protocol.projected_gradient_force_fraction
                ),
                "subdivision_count": int(selected_subdivision),
                "solver_diagnostics": attempts,
                "solver_diagnostics_json": json.dumps(attempts, separators=(",", ":")),
                "lower_platen_surface_mm": lower_platen_surface,
                "upper_platen_surface_mm": upper_platen_surface,
                "displacement": result.displacement,
                "element_stress_MPa": element_stress,
            }
        )
    elapsed = perf_counter() - start_time
    geometry = geometry_metrics(curve, thickness)
    travel = np.asarray([float(row["platen_travel_mm"]) for row in states])
    reaction = np.asarray([float(row["reaction_N"]) for row in states])
    work = float(np.trapezoid(reaction, x=travel))
    target = states[-1]
    normalizer_force = force_scale
    normalizer_work = normalizer_force * geometry["arc_length_mm"]
    if len(states) >= 2:
        d_force = float(states[-1]["reaction_N"]) - float(states[-2]["reaction_N"])
        d_travel = float(states[-1]["platen_travel_mm"]) - float(states[-2]["platen_travel_mm"])
        tangent_number = (d_force / max(d_travel, 1.0e-12)) * height / (
            protocol.elastic_modulus_MPa * protocol.width_mm * thickness
        )
    else:
        tangent_number = float("nan")
    metrics: dict[str, float | bool] = {
        **geometry,
        "solver_max_iterations": int(solver_max_iterations),
        "elements_per_wavelength": float(elements_per_wavelength),
        "discretized_arc_length_mm": discretized_arc,
        "protocol_fingerprint": protocol_fingerprint(elements_per_wavelength, protocol),
        "contact_offset_mm": contact_offset,
        "initial_upper_contact_gap_mm": float(height - nodes[:, 1].max()),
        "initial_lower_contact_gap_mm": float(nodes[:, 1].min()),
        "target_strain": float(protocol.strains[-1]),
        "target_reaction_N": float(target["reaction_N"]),
        "target_potential_energy_Nmm": float(target["potential_energy_Nmm"]),
        "external_work_Nmm": work,
        "target_stress_pnorm_MPa": float(target["stress_pnorm_MPa"]),
        "target_stress_q99_MPa": float(target["stress_q99_MPa"]),
        "target_maximum_stress_MPa": float(target["maximum_stress_MPa"]),
        "target_yielded_length_fraction": float(target["yielded_length_fraction"]),
        "crush_load_number": float(target["reaction_N"]) / normalizer_force,
        "stored_potential_number": float(target["potential_energy_Nmm"]) / normalizer_work,
        "external_work_number": work / normalizer_work,
        "stress_pnorm_utilization": float(target["stress_pnorm_MPa"])
        / protocol.yield_stress_MPa,
        "stress_q99_utilization": float(target["stress_q99_MPa"])
        / protocol.yield_stress_MPa,
        "stress_max_utilization": float(target["maximum_stress_MPa"])
        / protocol.yield_stress_MPa,
        "stress_localization_number": float(target["stress_q99_MPa"])
        * protocol.width_mm
        * thickness
        / max(float(target["reaction_N"]), 1.0e-12),
        "forming_yield_index": protocol.elastic_modulus_MPa
        * thickness
        / (2.0 * protocol.yield_stress_MPa * geometry["radius_min_mm"]),
        "work_per_material_number": work
        / normalizer_work
        / geometry["board_material_fraction"],
        "work_per_stress_number": work
        / normalizer_work
        / max(float(target["stress_pnorm_MPa"]) / protocol.yield_stress_MPa, 1.0e-12),
        "combined_work_material_stress_number": work
        / normalizer_work
        / (
            geometry["board_material_fraction"]
            * max(float(target["stress_pnorm_MPa"]) / protocol.yield_stress_MPa, 1.0e-12)
        ),
        "dimensionless_secant_tangent": tangent_number,
        "path_success": bool(all(bool(row["success"]) for row in states)),
        "maximum_normalized_projected_gradient": float(
            max(float(row["normalized_projected_gradient"]) for row in states)
        ),
        "maximum_reaction_imbalance_fraction": float(
            max(
                abs(float(row["reaction_N"]) - float(row["lower_reaction_N"]))
                / max(abs(float(row["reaction_N"])), abs(float(row["lower_reaction_N"])), 1.0e-12)
                for row in states[1:]
            )
            if len(states) > 1
            else 0.0
        ),
        "maximum_absolute_reaction_imbalance_N": float(
            max(
                abs(float(row["reaction_N"]) - float(row["lower_reaction_N"]))
                for row in states[1:]
            )
            if len(states) > 1
            else 0.0
        ),
        "maximum_reaction_imbalance_force_fraction": float(
            max(
                abs(float(row["reaction_N"]) - float(row["lower_reaction_N"]))
                / force_scale
                for row in states[1:]
            )
            if len(states) > 1
            else 0.0
        ),
        "all_raw_solver_success": bool(all(bool(row["raw_solver_success"]) for row in states)),
        "maximum_subdivision_count": int(
            max(int(row["subdivision_count"]) for row in states)
        ),
        "fallback_solver_state_count": int(
            sum(bool(row["fallback_used"]) for row in states)
        ),
        "independent_kkt_state_count": int(
            sum(bool(row["independent_kkt_confirmed"]) for row in states)
        ),
        "solver_diagnostics_json": json.dumps(
            [
                {
                    "strain": float(row["strain"]),
                    "accepted": bool(row["solver_accepted"]),
                    "selected_subdivision": int(row["subdivision_count"]),
                    "attempts": row["solver_diagnostics"],
                }
                for row in states
            ],
            separators=(",", ":"),
        ),
        "runtime_s": elapsed,
    }
    return states, metrics
