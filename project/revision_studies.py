#!/usr/bin/env python3
"""Additional numerical and plotting studies requested for the second revision.

The script never invents experimental observations.  Paper A studies the
corrected optimizer under a fixed evaluation budget; Paper B perturbs the
declared event definitions using the specimen-level records; and Paper C
executes the published beam/contact kernel for each topology/material state and
benchmarks conservative and unconstrained neural surrogates.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from plot_style import BLACK, GRAY, GRAY_DARK, GRAY_LIGHT, GRAY_PALE, apply_latex_style  # noqa: E402


PROFILE_ORDER = ["C1", "C2", "C3", "Sine"]
SURFACE_ORDER = ["Coated", "Uncoated"]
PROFILE_FILES = {
    "C1": "local_curve_1.txt",
    "C2": "local_curve_2.txt",
    "C3": "local_curve_3.txt",
    "Sine": "local_curve_sine.txt",
}
MATERIALS = {
    "Coated": {"E_MPa": 2794.0, "thickness_mm": 0.517},
    "Uncoated": {"E_MPa": 2899.0, "thickness_mm": 0.525},
}
STRAINS = np.asarray([0.00, 0.05, 0.10, 0.15, 0.20])
COLLAPSE_STRAINS = np.linspace(0.0, 0.80, 17)
COLLAPSE_SNAPSHOTS = np.asarray([0.00, 0.20, 0.40, 0.60, 0.80])


def save_figure(fig: plt.Figure, path: Path, dpi: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def paper_a_hyperparameters() -> None:
    from cbopt.optimizers import run_mo_etpso
    from run_paper_a import LOWER, REFERENCE, UPPER, objective

    """Fixed-budget MO-ETPSO-R population/inertia sensitivity."""

    result_dir = ROOT / "results" / "paper_a"
    figure_dir = ROOT / "figures" / "paper_a"
    result_dir.mkdir(parents=True, exist_ok=True)
    budget = 1200
    rows: list[dict[str, float | int]] = []

    def evaluator(x: np.ndarray):
        return objective(x, sample_size=140)

    for population in (20, 40, 80):
        generations = budget // population
        for inertia in (0.40, 0.70, 0.90):
            for seed in range(5):
                started = time.perf_counter()
                result = run_mo_etpso(
                    evaluator,
                    LOWER,
                    UPPER,
                    population_size=population,
                    generations=generations,
                    inertia=inertia,
                    seed=seed,
                    reference=REFERENCE,
                )
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "population": population,
                        "generations": generations,
                        "inertia": inertia,
                        "seed": seed,
                        "evaluation_budget": population * generations,
                        "final_hypervolume": float(result.history_hypervolume[-1]),
                        "feasible_fraction": float(np.mean(result.feasible)),
                        "pareto_members": int(result.pareto_mask.sum()),
                        "wall_time_s": elapsed,
                    }
                )
    data = pd.DataFrame(rows)
    data.to_csv(result_dir / "mo_etpso_hyperparameter_study.csv", index=False)
    summary = (
        data.groupby(["population", "inertia"], as_index=False)
        .agg(
            hypervolume_mean=("final_hypervolume", "mean"),
            hypervolume_sd=("final_hypervolume", "std"),
            feasible_mean=("feasible_fraction", "mean"),
            feasible_sd=("feasible_fraction", "std"),
            pareto_mean=("pareto_members", "mean"),
            pareto_sd=("pareto_members", "std"),
            time_mean_s=("wall_time_s", "mean"),
            time_sd_s=("wall_time_s", "std"),
        )
    )
    summary.to_csv(result_dir / "mo_etpso_hyperparameter_summary.csv", index=False)

    apply_latex_style(7.8)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
    markers = {20: "o", 40: "s", 80: "^"}
    styles = {20: "-", 40: "--", 80: ":"}
    for population, group in summary.groupby("population"):
        axes[0].errorbar(
            group.inertia,
            group.hypervolume_mean,
            yerr=group.hypervolume_sd,
            color=BLACK,
            linestyle=styles[population],
            marker=markers[population],
            markerfacecolor="white",
            capsize=2,
            label=rf"$N_p={population}$",
        )
        axes[1].errorbar(
            group.inertia,
            group.pareto_mean,
            yerr=group.pareto_sd,
            color=BLACK,
            linestyle=styles[population],
            marker=markers[population],
            markerfacecolor="white",
            capsize=2,
        )
        axes[2].errorbar(
            group.inertia,
            group.time_mean_s,
            yerr=group.time_sd_s,
            color=BLACK,
            linestyle=styles[population],
            marker=markers[population],
            markerfacecolor="white",
            capsize=2,
        )
    axes[0].set_ylabel("Final normalized hypervolume")
    axes[1].set_ylabel("Final Pareto members")
    axes[2].set_ylabel("Wall time per run (s)")
    for ax in axes:
        ax.set_xlabel("Inertia coefficient, $w$")
        ax.set_xticks([0.4, 0.7, 0.9])
        ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.1)
    save_figure(fig, figure_dir / "mo_etpso_hyperparameter_study")


def paper_b_sensitivity() -> None:
    # Imported lazily so the Paper C gallery/surrogate workflows remain
    # independently executable in the standalone FEM/PINN repository.
    from analyze_paper_b import fit_weibull
    """Engagement-threshold and censoring-definition sensitivity."""

    metrics = pd.read_csv(ROOT / "results" / "paper_b" / "specimen_metrics.csv")
    processed = pd.read_csv(ROOT / "results" / "paper_b" / "processed_curves.csv")
    result_dir = ROOT / "results" / "paper_b"
    figure_dir = ROOT / "figures" / "paper_b"

    engagement_rows: list[dict[str, float | str | int]] = []
    for (profile, coating), group in metrics.groupby(["profile", "coating"]):
        for fraction in (0.05, 0.10, 0.20):
            values = group[f"engagement_strain_{int(100 * fraction):02d}"].to_numpy()
            fit = fit_weibull(values)
            engagement_rows.append(
                {
                    "profile": profile,
                    "coating": coating,
                    "engagement_fraction": fraction,
                    "n": int(np.isfinite(values).sum()),
                    "beta": fit.beta,
                    "eta_strain": fit.eta,
                    "fit_success": fit.success,
                }
            )
    engagement = pd.DataFrame(engagement_rows)
    engagement.to_csv(result_dir / "engagement_threshold_sensitivity.csv", index=False)

    terminal_rows: list[dict[str, float | str | int]] = []
    definitions = [(position, drop) for position in (0.96, 0.98, 0.995) for drop in (0.03, 0.05, 0.10)]
    for (profile, coating), group in metrics.groupby(["profile", "coating"]):
        for definition, (position, drop) in enumerate(definitions):
            observed = (group.peak_position_fraction < position) & (group.post_peak_drop_fraction >= drop)
            fit = fit_weibull(group.peak_strain.to_numpy(), observed.to_numpy())
            terminal_rows.append(
                {
                    "profile": profile,
                    "coating": coating,
                    "definition": definition,
                    "peak_position_limit": position,
                    "post_peak_drop": drop,
                    "events": int(observed.sum()),
                    "event_fraction": float(observed.mean()),
                    "beta": fit.beta,
                    "eta_strain": fit.eta,
                    "fit_success": fit.success,
                }
            )
    terminal = pd.DataFrame(terminal_rows)
    terminal.to_csv(result_dir / "terminal_definition_sensitivity.csv", index=False)

    apply_latex_style(7.5)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0))
    profile_styles = {"C1": "-", "C2": "--", "C3": ":", "Sine": "-."}
    surface_markers = {"Coated": "o", "Uncoated": "s"}
    for (profile, coating), group in engagement.groupby(["profile", "coating"]):
        label = f"{profile}, {coating.lower()}"
        axes[0, 0].plot(
            100 * group.engagement_fraction,
            group.eta_strain,
            color=BLACK,
            linestyle=profile_styles[profile],
            marker=surface_markers[coating],
            markerfacecolor="white",
            markersize=3.5,
            label=label,
        )
        axes[0, 1].plot(
            100 * group.engagement_fraction,
            group.beta,
            color=BLACK,
            linestyle=profile_styles[profile],
            marker=surface_markers[coating],
            markerfacecolor="white",
            markersize=3.5,
        )
    for (profile, coating), group in terminal.groupby(["profile", "coating"]):
        axes[1, 0].plot(
            group.definition + 1,
            group.event_fraction,
            color=BLACK,
            linestyle=profile_styles[profile],
            marker=surface_markers[coating],
            markerfacecolor="white",
            markersize=3.0,
        )
        axes[1, 1].plot(
            group.definition + 1,
            group.eta_strain,
            color=BLACK,
            linestyle=profile_styles[profile],
            marker=surface_markers[coating],
            markerfacecolor="white",
            markersize=3.0,
        )
    axes[0, 0].set_ylabel(r"Engagement scale, $\eta_e$")
    axes[0, 1].set_ylabel(r"Engagement shape, $\beta_e$")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_ylabel("Observed terminal-event fraction")
    axes[1, 1].set_ylabel(r"Terminal scale, $\eta_t$")
    axes[0, 0].set_xlabel("Engagement threshold (% early load)")
    axes[0, 1].set_xlabel("Engagement threshold (% early load)")
    axes[1, 0].set_xlabel("Terminal-event definition index")
    axes[1, 1].set_xlabel("Terminal-event definition index")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=6.2)
    for ax in axes.flat:
        ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    fig.tight_layout(h_pad=1.1, w_pad=1.1)
    save_figure(fig, figure_dir / "event_definition_sensitivity")

    # All specimen paths are retained; only medians were visible previously.
    common = np.linspace(0.0, 0.75, 350)
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.25), sharex=True)
    for ax, profile, coating in zip(
        axes.flat,
        PROFILE_ORDER * 2,
        ["Coated"] * 4 + ["Uncoated"] * 4,
    ):
        group = processed[(processed.profile == profile) & (processed.coating == coating)]
        paths = []
        for _, specimen in group.groupby("specimen_id"):
            values = np.interp(common, specimen.strain, specimen.stress_MPa, left=np.nan, right=np.nan)
            paths.append(values)
            ax.plot(common, values, color=GRAY_LIGHT, linewidth=0.45)
        array = np.asarray(paths)
        median = np.nanmedian(array, axis=0)
        valid = np.sum(np.isfinite(array), axis=0) >= 3
        ax.plot(common[valid], median[valid], color=BLACK, linewidth=1.25)
        ax.set_title(f"{profile} -- {coating.lower()} ($n={len(paths)}$)")
        ax.grid(color=GRAY_PALE, linewidth=0.35)
    for ax in axes[-1]:
        ax.set_xlabel("Compression strain")
    for ax in axes[:, 0]:
        ax.set_ylabel("Nominal stress (MPa)")
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save_figure(fig, figure_dir / "individual_curve_variability")

    for profile in PROFILE_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.65), sharex=True)
        for ax, coating in zip(axes, SURFACE_ORDER):
            group = processed[(processed.profile == profile) & (processed.coating == coating)]
            paths = []
            for _, specimen in group.groupby("specimen_id"):
                values = np.interp(common, specimen.strain, specimen.stress_MPa, left=np.nan, right=np.nan)
                paths.append(values)
                ax.plot(common, values, color=GRAY_LIGHT, linewidth=0.55)
            array = np.asarray(paths)
            median = np.nanmedian(array, axis=0)
            valid = np.sum(np.isfinite(array), axis=0) >= 3
            ax.plot(common[valid], median[valid], color=BLACK, linewidth=1.35, label="group median")
            ax.set_title(f"{profile}: {coating.lower()} ($n={len(paths)}$)")
            ax.set_xlabel("Compression strain")
            ax.grid(color=GRAY_PALE, linewidth=0.4)
        axes[0].set_ylabel("Nominal stress (MPa)")
        axes[0].legend(frameon=False)
        fig.tight_layout(w_pad=1.0)
        save_figure(fig, figure_dir / f"individual_curves_{profile.lower()}")

    # Specimen-level relationships distinguish resistance, absorbed energy,
    # initial structural stiffness and the terminal observation window.
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
    profile_markers = {"C1": "o", "C2": "s", "C3": "^", "Sine": "D"}
    for (profile, coating), group in metrics.groupby(["profile", "coating"]):
        face = "white" if coating == "Coated" else BLACK
        common_style = {
            "marker": profile_markers[profile],
            "facecolor": face,
            "edgecolor": BLACK,
            "linewidth": 0.55,
            "s": 17,
            "alpha": 0.85,
        }
        axes[0].scatter(group.mean_crush_stress_MPa, group.areal_energy_kJ_m2, **common_style)
        axes[1].scatter(group.early_tangent_stiffness_N_mm, group.areal_energy_kJ_m2, **common_style)
        axes[2].scatter(group.peak_strain, group.crush_efficiency, **common_style)
    axes[0].set_xlabel("Mean crush stress (MPa)")
    axes[0].set_ylabel(r"Areal energy (kJ m$^{-2}$)")
    axes[1].set_xlabel(r"Early tangent stiffness (N mm$^{-1}$)")
    axes[1].set_ylabel(r"Areal energy (kJ m$^{-2}$)")
    axes[2].set_xlabel("Terminal recorded strain")
    axes[2].set_ylabel("Crush efficiency")
    for profile, marker in profile_markers.items():
        axes[2].scatter([], [], marker=marker, facecolor="white", edgecolor=BLACK, label=profile)
    axes[2].scatter([], [], marker="o", facecolor="white", edgecolor=BLACK, label="coated")
    axes[2].scatter([], [], marker="o", facecolor=BLACK, edgecolor=BLACK, label="uncoated")
    axes[2].legend(frameon=False, ncol=2, fontsize=6.3)
    for ax in axes:
        ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, figure_dir / "specimen_level_coefficient_relationships")


def _solve_profile_path(
    profile: str,
    surface: str,
    *,
    elements: int = 20,
    elastic_scale: float = 1.0,
    yield_stress_MPa: float = 60.0,
) -> list[dict[str, object]]:
    from cbfem import BeamMaterial, BeamSection, CorotationalBeamModel, load_profile, repeat_profile

    properties = MATERIALS[surface]
    nodes = repeat_profile(
        load_profile(ROOT / "data" / "geometries" / PROFILE_FILES[profile]),
        repetitions=1,
        elements_per_wavelength=elements,
    )
    elastic_modulus = properties["E_MPa"] * elastic_scale
    material = BeamMaterial(
        elastic_modulus_MPa=elastic_modulus,
        shear_modulus_MPa=elastic_modulus / 55.0,
        yield_stress_MPa=yield_stress_MPa,
        hardening_ratio=0.02,
    )
    model = CorotationalBeamModel(
        nodes,
        material,
        BeamSection(width_mm=25.0, thickness_mm=properties["thickness_mm"]),
    )
    lower = float(nodes[:, 1].min())
    height = float(np.ptp(nodes[:, 1]))
    last = len(nodes) - 1
    ties = [(3 * last + 1, 1), (3 * last + 2, 2)]
    previous = None
    states: list[dict[str, object]] = []
    for strain in STRAINS:
        result = model.solve(
            fixed_dofs={0: 0.0},
            lower_y_mm=lower,
            upper_y_mm=lower + height * (1.0 - strain),
            contact_mode="bounds",
            tied_dofs=ties,
            initial_displacement=previous,
            tolerance=1e-9,
            max_iterations=5000,
        )
        previous = result.displacement
        current = nodes + result.displacement.reshape((-1, 3))[:, :2]
        stress_arrays = [
            values
            for name, values in result.element_results.items()
            if name.startswith("stress_")
        ]
        stress = np.max(np.abs(np.vstack(stress_arrays)), axis=0)
        states.append(
            {
                "profile": profile,
                "surface": surface,
                "strain": float(strain),
                "nodes": nodes,
                "coordinates": current,
                "stress": stress,
                "reaction_N": float(result.top_reaction_N),
                "upper_y": lower + height * (1.0 - strain),
                "lower_y": lower,
                "success": bool(result.success),
                "iterations": int(result.iterations),
                "elastic_scale": elastic_scale,
                "yield_stress_MPa": yield_stress_MPa,
            }
        )
    return states


def _stress_axis(ax: plt.Axes, state: dict[str, object], norm: Normalize) -> LineCollection:
    coordinates = np.asarray(state["coordinates"])
    undeformed = np.asarray(state["nodes"])
    element_stress = np.asarray(state["stress"], dtype=float)
    # Recover a continuous nodal field by length-weighted averaging and then
    # interpolate each element into subsegments. This is display
    # post-processing only: reaction and element resultants remain unchanged.
    lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    nodal_stress = np.zeros(len(coordinates))
    weights = np.zeros(len(coordinates))
    nodal_stress[:-1] += element_stress * lengths
    nodal_stress[1:] += element_stress * lengths
    weights[:-1] += lengths
    weights[1:] += lengths
    nodal_stress /= np.maximum(weights, 1.0e-14)
    smooth_segments = []
    smooth_values = []
    subdivisions = 10
    for index, (first, second) in enumerate(zip(coordinates[:-1], coordinates[1:])):
        parameter = np.linspace(0.0, 1.0, subdivisions + 1)
        points = first[None, :] * (1.0 - parameter[:, None]) + second[None, :] * parameter[:, None]
        values = nodal_stress[index] * (1.0 - parameter) + nodal_stress[index + 1] * parameter
        smooth_segments.extend(np.stack((points[:-1], points[1:]), axis=1))
        smooth_values.extend(0.5 * (values[:-1] + values[1:]))
    collection = LineCollection(smooth_segments, cmap="turbo", norm=norm, linewidth=2.15)
    collection.set_array(np.asarray(smooth_values))
    ax.add_collection(collection)
    ax.plot(undeformed[:, 0], undeformed[:, 1], color=GRAY_LIGHT, linestyle="--", linewidth=0.65)
    ax.axhline(float(state["lower_y"]), color=BLACK, linewidth=0.85)
    ax.axhline(float(state["upper_y"]), color=BLACK, linewidth=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(undeformed[:, 0].min()) - 0.4, float(undeformed[:, 0].max()) + 0.4)
    ax.set_ylim(float(state["lower_y"]) - 0.55, float(undeformed[:, 1].max()) + 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    return collection


def _nonlocal_midpoint_clearance(coordinates: np.ndarray) -> float:
    midpoints = 0.5 * (coordinates[:-1] + coordinates[1:])
    clearance = np.inf
    for first in range(len(midpoints)):
        for second in range(first + 3, len(midpoints)):
            clearance = min(clearance, float(np.linalg.norm(midpoints[first] - midpoints[second])))
    return float(clearance)


def _solve_large_collapse_path(
    profile: str,
    surface: str,
    *,
    elements: int = 24,
) -> list[dict[str, object]]:
    from cbfem import BeamMaterial, BeamSection, CorotationalBeamModel, load_profile, repeat_profile

    properties = MATERIALS[surface]
    nodes = repeat_profile(
        load_profile(ROOT / "data" / "geometries" / PROFILE_FILES[profile]),
        repetitions=1,
        elements_per_wavelength=elements,
    )
    elastic_modulus = properties["E_MPa"]
    thickness = properties["thickness_mm"]
    model = CorotationalBeamModel(
        nodes,
        BeamMaterial(
            elastic_modulus_MPa=elastic_modulus,
            shear_modulus_MPa=elastic_modulus / 55.0,
            yield_stress_MPa=60.0,
            hardening_ratio=0.02,
        ),
        BeamSection(width_mm=25.0, thickness_mm=thickness),
    )
    lower = float(nodes[:, 1].min())
    height = float(np.ptp(nodes[:, 1]))
    last = len(nodes) - 1
    ties = [(3 * last + 1, 1), (3 * last + 2, 2)]
    previous = None
    states: list[dict[str, object]] = []
    for strain in COLLAPSE_STRAINS:
        result = model.solve(
            fixed_dofs={0: 0.0},
            lower_y_mm=lower,
            upper_y_mm=lower + height * (1.0 - strain),
            contact_mode="bounds",
            tied_dofs=ties,
            self_contact_distance_mm=0.95 * thickness,
            self_contact_penalty_N_mm=7.5e3,
            initial_displacement=previous,
            tolerance=1.0e-6,
            max_iterations=1800,
        )
        previous = result.displacement
        current = nodes + result.displacement.reshape((-1, 3))[:, :2]
        stress_arrays = [
            values for name, values in result.element_results.items() if name.startswith("stress_")
        ]
        stress = np.max(np.abs(np.vstack(stress_arrays)), axis=0)
        states.append(
            {
                "profile": profile,
                "surface": surface,
                "strain": float(strain),
                "nodes": nodes,
                "coordinates": current,
                "stress": stress,
                "reaction_N": float(result.top_reaction_N),
                "energy_Nmm": float(result.potential_energy_Nmm),
                "upper_y": lower + height * (1.0 - strain),
                "lower_y": lower,
                "success": bool(result.success),
                "iterations": int(result.iterations),
                "minimum_nonlocal_clearance_mm": _nonlocal_midpoint_clearance(current),
                "contact_distance_mm": 0.95 * thickness,
            }
        )
    return states


def paper_c_large_collapse() -> None:
    """Contact-regularized continuation from initial loading to 80% collapse."""

    apply_latex_style(7.4)
    figure_dir = ROOT / "figures" / "paper_c" / "collapse"
    result_dir = ROOT / "results" / "paper_c" / "collapse"
    result_dir.mkdir(parents=True, exist_ok=True)
    all_states: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for profile in PROFILE_ORDER:
        profile_states = []
        for surface in SURFACE_ORDER:
            states = _solve_large_collapse_path(profile, surface)
            profile_states.extend(states)
            all_states.extend(states)
            for state in states:
                records.append(
                    {
                        "profile": profile,
                        "surface": surface,
                        "strain": state["strain"],
                        "reaction_N": state["reaction_N"],
                        "energy_Nmm": state["energy_Nmm"],
                        "maximum_extreme_fibre_stress_MPa": float(np.max(state["stress"])),
                        "minimum_nonlocal_clearance_mm": state["minimum_nonlocal_clearance_mm"],
                        "contact_distance_mm": state["contact_distance_mm"],
                        "solver_success": state["success"],
                        "solver_iterations": state["iterations"],
                        "elements_per_wavelength": 24,
                    }
                )
        snapshots = [
            state for state in profile_states
            if np.any(np.isclose(float(state["strain"]), COLLAPSE_SNAPSHOTS))
        ]
        maximum = max(float(np.max(state["stress"])) for state in snapshots)
        norm = Normalize(0.0, maximum)
        fig, axes = plt.subplots(5, 2, figsize=(7.15, 7.25))
        mappable = None
        for row, strain in enumerate(COLLAPSE_SNAPSHOTS):
            for column, surface in enumerate(SURFACE_ORDER):
                state = next(
                    item for item in snapshots
                    if item["surface"] == surface and np.isclose(float(item["strain"]), strain)
                )
                mappable = _stress_axis(axes[row, column], state, norm)
                axes[row, column].text(
                    0.01, 0.93,
                    rf"$\varepsilon={strain:.2f}$; $R={float(state['reaction_N']):.2f}$ N",
                    transform=axes[row, column].transAxes, ha="left", va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.30, "alpha": 0.88},
                )
        if mappable is not None:
            colorbar = fig.colorbar(mappable, ax=axes, location="right", fraction=0.025, pad=0.025)
            colorbar.set_label(r"Smoothed extreme-fibre $|\sigma|$ (MPa)")
        fig.suptitle(f"{profile}: contact-regularized large collapse", y=0.995)
        fig.text(0.23, 0.963, "Coated", ha="center", va="top")
        fig.text(0.66, 0.963, "Uncoated", ha="center", va="top")
        fig.subplots_adjust(left=0.035, right=0.88, bottom=0.02, top=0.94, hspace=0.16, wspace=0.05)
        save_figure(fig, figure_dir / f"{profile.lower()}_large_collapse_gallery")

    data = pd.DataFrame(records)
    data.to_csv(result_dir / "large_collapse_paths.csv", index=False)
    styles = {"C1": "-", "C2": "--", "C3": ":", "Sine": "-."}
    markers = {"Coated": "o", "Uncoated": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    for (profile, surface), group in data.groupby(["profile", "surface"]):
        group = group.sort_values("strain")
        label = f"{profile}, {surface.lower()}"
        kwargs = dict(color=BLACK, linestyle=styles[profile], marker=markers[surface], markerfacecolor="white", markersize=2.7, markevery=4, linewidth=0.85, label=label)
        axes[0].plot(group.strain, group.reaction_N, **kwargs)
        absorbed = np.concatenate(([0.0], np.cumsum(0.5 * (group.reaction_N.to_numpy()[1:] + group.reaction_N.to_numpy()[:-1]) * np.diff(group.strain.to_numpy()) * 10.0)))
        axes[1].plot(group.strain, absorbed, **kwargs)
        axes[2].plot(group.strain, group.minimum_nonlocal_clearance_mm / group.contact_distance_mm, **kwargs)
    axes[0].set_ylabel("Top-platen reaction (N)")
    axes[1].set_ylabel("Accumulated work (N mm)")
    axes[2].set_ylabel("Minimum clearance / contact distance")
    for ax in axes:
        ax.set_xlabel("Compression strain")
        ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    axes[0].legend(frameon=False, fontsize=5.8, ncol=2)
    fig.tight_layout(w_pad=0.9)
    save_figure(fig, figure_dir / "large_collapse_response_and_contact")


def paper_c_galleries() -> None:
    """Full topology/surface FEM stress gallery plus material sensitivity."""

    apply_latex_style(7.4)
    figure_dir = ROOT / "figures" / "paper_c" / "galleries"
    result_dir = ROOT / "results" / "paper_c" / "galleries"
    result_dir.mkdir(parents=True, exist_ok=True)
    all_states: list[dict[str, object]] = []
    records: list[dict[str, float | str | int | bool]] = []
    for profile in PROFILE_ORDER:
        profile_states: list[dict[str, object]] = []
        for surface in SURFACE_ORDER:
            states = _solve_profile_path(profile, surface)
            profile_states.extend(states)
            all_states.extend(states)
            for state in states:
                records.append(
                    {
                        "profile": profile,
                        "surface": surface,
                        "strain": state["strain"],
                        "reaction_N": state["reaction_N"],
                        "maximum_extreme_fibre_stress_MPa": float(np.max(state["stress"])),
                        "solver_success": state["success"],
                        "solver_iterations": state["iterations"],
                        "elements_per_wavelength": 20,
                    }
                )
        maximum = max(float(np.max(state["stress"])) for state in profile_states)
        norm = Normalize(0.0, maximum)
        fig, axes = plt.subplots(5, 2, figsize=(7.15, 7.15))
        mappable = None
        for row, strain in enumerate(STRAINS):
            for column, surface in enumerate(SURFACE_ORDER):
                state = next(
                    item
                    for item in profile_states
                    if item["surface"] == surface and np.isclose(float(item["strain"]), strain)
                )
                ax = axes[row, column]
                mappable = _stress_axis(ax, state, norm)
                ax.text(
                    0.01,
                    0.92,
                    rf"$\varepsilon={strain:.2f}$; $R={float(state['reaction_N']):.2f}$ N",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.86},
                )
        if mappable is not None:
            colorbar = fig.colorbar(mappable, ax=axes, location="right", fraction=0.025, pad=0.025)
            colorbar.set_label(r"Maximum extreme-fibre $|\sigma|$ (MPa)")
        fig.suptitle(f"{profile}: nonlinear deformation and stress", y=0.995)
        fig.text(0.23, 0.963, "Coated", ha="center", va="top")
        fig.text(0.66, 0.963, "Uncoated", ha="center", va="top")
        fig.subplots_adjust(left=0.035, right=0.88, bottom=0.02, top=0.94, hspace=0.16, wspace=0.05)
        save_figure(fig, figure_dir / f"{profile.lower()}_coated_uncoated_stress_gallery")

    pd.DataFrame(records).to_csv(result_dir / "topology_surface_stress_gallery.csv", index=False)

    # Same strain and a common scale: direct across-topology/material comparison.
    final_states = [state for state in all_states if np.isclose(float(state["strain"]), 0.20)]
    maximum = max(float(np.max(state["stress"])) for state in final_states)
    norm = Normalize(0.0, maximum)
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 2.75))
    mappable = None
    for row, surface in enumerate(SURFACE_ORDER):
        for column, profile in enumerate(PROFILE_ORDER):
            state = next(item for item in final_states if item["surface"] == surface and item["profile"] == profile)
            mappable = _stress_axis(axes[row, column], state, norm)
            axes[row, column].set_title(f"{profile}\n$R={float(state['reaction_N']):.1f}$ N")
            if column == 0:
                axes[row, column].set_ylabel(surface)
    if mappable is not None:
        colorbar = fig.colorbar(mappable, ax=axes, location="right", fraction=0.026, pad=0.025)
        colorbar.set_label(r"Maximum extreme-fibre $|\sigma|$ (MPa)")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.04, top=0.93, hspace=0.18, wspace=0.05)
    save_figure(fig, figure_dir / "topology_surface_stress_overview")

    # Orthogonal material sweep for an uncoated sine cell.
    sensitivity_states: list[dict[str, object]] = []
    sensitivity_records: list[dict[str, float | bool | int]] = []
    for yield_stress in (40.0, 60.0, 90.0):
        for scale in (0.65, 1.00, 1.35):
            state = _solve_profile_path(
                "Sine",
                "Uncoated",
                elastic_scale=scale,
                yield_stress_MPa=yield_stress,
            )[-1]
            sensitivity_states.append(state)
            sensitivity_records.append(
                {
                    "elastic_scale": scale,
                    "elastic_modulus_MPa": MATERIALS["Uncoated"]["E_MPa"] * scale,
                    "yield_stress_MPa": yield_stress,
                    "strain": 0.20,
                    "reaction_N": float(state["reaction_N"]),
                    "maximum_extreme_fibre_stress_MPa": float(np.max(state["stress"])),
                    "solver_success": bool(state["success"]),
                    "solver_iterations": int(state["iterations"]),
                }
            )
    pd.DataFrame(sensitivity_records).to_csv(result_dir / "material_parameter_sensitivity.csv", index=False)
    maximum = max(float(np.max(state["stress"])) for state in sensitivity_states)
    norm = Normalize(0.0, maximum)
    fig, axes = plt.subplots(3, 3, figsize=(7.15, 4.25))
    mappable = None
    for row, yield_stress in enumerate((40.0, 60.0, 90.0)):
        for column, scale in enumerate((0.65, 1.00, 1.35)):
            state = next(
                item
                for item in sensitivity_states
                if np.isclose(float(item["yield_stress_MPa"]), yield_stress)
                and np.isclose(float(item["elastic_scale"]), scale)
            )
            mappable = _stress_axis(axes[row, column], state, norm)
            axes[row, column].set_title(rf"$E/E_0={scale:.2f}$; $R={float(state['reaction_N']):.1f}$ N")
            if column == 0:
                axes[row, column].set_ylabel(rf"$\sigma_y={yield_stress:.0f}$ MPa")
    if mappable is not None:
        colorbar = fig.colorbar(mappable, ax=axes, location="right", fraction=0.025, pad=0.025)
        colorbar.set_label(r"Maximum extreme-fibre $|\sigma|$ (MPa)")
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.03, top=0.96, hspace=0.20, wspace=0.06)
    save_figure(fig, figure_dir / "material_parameter_stress_sensitivity")


def _score(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan
    return rmse, r2


def paper_c_surrogates() -> None:
    """Conservative potential vs direct MLP/ridge, including repeat timings."""

    from cbpinn import fit_energy_network
    from run_paper_c_pinn import feature_matrix
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    dataset = pd.read_csv(ROOT / "results" / "paper_c" / "pinn" / "fem_parametric_paths.csv")
    training = dataset[dataset.split == "train"].copy()
    test = dataset[(dataset.split == "topology_test") & (dataset.strain > 0)].copy().reset_index(drop=True)
    x_train = feature_matrix(training)
    x_test = feature_matrix(test)
    y_train = training[["energy_Nmm", "reaction_N"]].to_numpy()
    y_test = test[["energy_Nmm", "reaction_N"]].to_numpy()
    rows: list[dict[str, float | str | int]] = []
    seeds = (1, 3, 7, 11, 19)

    for seed in seeds:
        for width in (6, 10, 18):
            started = time.perf_counter()
            network, diagnostics = fit_energy_network(
                x_train,
                training.energy_Nmm.to_numpy(),
                training.reaction_N.to_numpy(),
                training.height_mm.to_numpy(),
                training.case_id.to_numpy(),
                hidden_size=width,
                seed=seed,
                max_iterations=600,
            )
            elapsed = time.perf_counter() - started
            predicted_energy = network.energy_Nmm(x_test)
            # Reaction is obtained from the same finite-difference derivative used
            # in the reported conservative surrogate.
            predicted_reaction = np.zeros(len(test))
            for _, indices in test.groupby("case_id").groups.items():
                indices = np.asarray(list(indices))
                order = indices[np.argsort(test.loc[indices, "strain"])]
                full_case = dataset[(dataset.case_id == test.loc[order[0], "case_id"])].sort_values("strain")
                full_energy = network.energy_Nmm(feature_matrix(full_case))
                full_reaction = np.zeros(len(full_case))
                full_reaction[1:] = np.diff(full_energy) / (10.0 * np.diff(full_case.strain.to_numpy()))
                predicted_reaction[order] = full_reaction[1:]
            e_rmse, e_r2 = _score(y_test[:, 0], predicted_energy)
            r_rmse, r_r2 = _score(y_test[:, 1], predicted_reaction)
            rows.extend(
                [
                    {"model": "Conservative PINN", "width": width, "seed": seed, "quantity": "Energy", "RMSE": e_rmse, "R2": e_r2, "wall_time_s": elapsed, "iterations": diagnostics["iterations"]},
                    {"model": "Conservative PINN", "width": width, "seed": seed, "quantity": "Reaction", "RMSE": r_rmse, "R2": r_r2, "wall_time_s": elapsed, "iterations": diagnostics["iterations"]},
                ]
            )

        for model_name, estimator in (
            (
                "Direct MLP",
                make_pipeline(
                    StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(18, 18), max_iter=1200, random_state=seed, early_stopping=False),
                ),
            ),
            (
                "Polynomial ridge",
                make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0)),
            ),
        ):
            started = time.perf_counter()
            estimator.fit(x_train, y_train)
            prediction = estimator.predict(x_test)
            elapsed = time.perf_counter() - started
            for column, quantity in enumerate(("Energy", "Reaction")):
                rmse, r2 = _score(y_test[:, column], prediction[:, column])
                rows.append(
                    {"model": model_name, "width": 18 if model_name == "Direct MLP" else 0, "seed": seed, "quantity": quantity, "RMSE": rmse, "R2": r2, "wall_time_s": elapsed, "iterations": np.nan}
                )

    results = pd.DataFrame(rows)
    output = ROOT / "results" / "paper_c" / "pinn"
    results.to_csv(output / "surrogate_comparison_repeats.csv", index=False)
    summary = (
        results.groupby(["model", "width", "quantity"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            time_mean_s=("wall_time_s", "mean"),
            time_sd_s=("wall_time_s", "std"),
        )
    )
    summary.to_csv(output / "surrogate_comparison_summary.csv", index=False)

    apply_latex_style(7.7)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    model_styles = {"Conservative PINN": ("o", "-"), "Direct MLP": ("s", "--"), "Polynomial ridge": ("^", ":")}
    positions = {"Polynomial ridge": 0, "Direct MLP": 1, "Conservative PINN": 2}
    for quantity, offset in (("Energy", -0.07), ("Reaction", 0.07)):
        for model, group in summary[summary.quantity == quantity].groupby("model"):
            if model == "Conservative PINN":
                group = group[group.width == 10]
            x = positions[model] + offset
            axes[0].errorbar(x, group.R2_mean.iloc[0], yerr=group.R2_sd.iloc[0], color=BLACK, marker="o" if quantity == "Energy" else "s", markerfacecolor="white", capsize=2, linestyle="none", label=quantity if model == "Polynomial ridge" else None)
            axes[1].errorbar(x, group.RMSE_mean.iloc[0], yerr=group.RMSE_sd.iloc[0], color=BLACK, marker="o" if quantity == "Energy" else "s", markerfacecolor="white", capsize=2, linestyle="none")
        conservative = summary[(summary.model == "Conservative PINN") & (summary.quantity == quantity)]
        axes[2].errorbar(conservative.width + offset, conservative.R2_mean, yerr=conservative.R2_sd, color=BLACK, linestyle="-" if quantity == "Energy" else "--", marker="o" if quantity == "Energy" else "s", markerfacecolor="white", capsize=2, label=quantity)
    labels = ["Ridge", "Direct\nMLP", "Conservative\nPINN"]
    axes[0].set_xticks([0, 1, 2], labels)
    axes[1].set_xticks([0, 1, 2], labels)
    axes[0].set_ylabel(r"Topology-test $R^2$")
    axes[1].set_ylabel("Topology-test RMSE")
    axes[2].set_xlabel("Conservative hidden width")
    axes[2].set_xticks([6, 10, 18])
    axes[2].set_ylabel(r"Topology-test $R^2$")
    for ax in axes:
        ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.tight_layout(w_pad=1.1)
    save_figure(fig, ROOT / "figures" / "paper_c" / "pinn" / "surrogate_hyperparameter_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=("paper_a", "paper_b", "paper_c_galleries", "paper_c_surrogates", "all"))
    args = parser.parse_args()
    if args.study in ("paper_a", "all"):
        paper_a_hyperparameters()
    if args.study in ("paper_b", "all"):
        paper_b_sensitivity()
    if args.study in ("paper_c_galleries", "all"):
        paper_c_galleries()
    if args.study in ("paper_c_surrogates", "all"):
        paper_c_surrogates()


if __name__ == "__main__":
    main()
