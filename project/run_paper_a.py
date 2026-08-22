#!/usr/bin/env python3
"""Run the paired geometric optimization benchmark for Paper A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cbopt.evaluator import evaluate_design, nurbs_profile  # noqa: E402
from cbopt.optimizers import run_mo_etpso, run_nsga2  # noqa: E402
from plot_style import (  # noqa: E402
    BLACK, GRAY_DARK, GRAY_LIGHT, GRAY_PALE, apply_latex_style,
)


LOWER = np.array([0.1] * 5 + [0.0, 0.0])
UPPER = np.array([10.0] * 5 + [1.0, 1.0])
WAVELENGTH_MM = 5.65
AMPLITUDE_MM = 2.65
THICKNESS_MM = 0.5
REFERENCE = np.array([1.0, 15.0])


def objective(design: np.ndarray, sample_size: int = 300):
    result = evaluate_design(
        design,
        sample_size=sample_size,
        wavelength_mm=WAVELENGTH_MM,
        amplitude_mm=AMPLITUDE_MM,
        thickness_mm=THICKNESS_MM,
        radius_limit_mm=0.9,
    )
    values = np.array([result.inverse_inertia_scaled / 1000.0, result.section_area_m2 * 1e6])
    violation = max(0.0, 0.9 - result.minimum_radius_1_mm) + max(
        0.0, 0.9 - result.minimum_radius_2_mm
    )
    return values, result.feasible, violation


def run_benchmark(seeds: int, population: int, generations: int, sample_size: int):
    final_records = []
    histories = []
    results = []

    def evaluator(x):
        return objective(x, sample_size)

    for seed in range(seeds):
        for function in (run_nsga2, run_mo_etpso):
            result = function(
                evaluator,
                LOWER,
                UPPER,
                population_size=population,
                generations=generations,
                seed=seed,
                reference=REFERENCE,
            )
            results.append(result)
            for generation, hv in enumerate(result.history_hypervolume):
                histories.append(
                    {
                        "algorithm": result.algorithm,
                        "seed": seed,
                        "generation": generation + 1,
                        "evaluations": (generation + 1) * population,
                        "hypervolume": hv,
                    }
                )
            for i, (position, values) in enumerate(zip(result.positions, result.objectives)):
                detail = evaluate_design(
                    position,
                    sample_size=800,
                    wavelength_mm=WAVELENGTH_MM,
                    amplitude_mm=AMPLITUDE_MM,
                    thickness_mm=THICKNESS_MM,
                    radius_limit_mm=0.9,
                )
                final_records.append(
                    {
                        "algorithm": result.algorithm,
                        "seed": seed,
                        "member": i,
                        **{f"x{j + 1}": position[j] for j in range(7)},
                        "inverse_inertia_scaled": detail.inverse_inertia_scaled,
                        "section_area_m2": detail.section_area_m2,
                        "area_per_wavelength_mm": detail.area_per_wavelength_mm,
                        "inertia_per_wavelength_mm3": detail.inertia_per_wavelength_mm3,
                        "effective_transverse_modulus_mpa": detail.effective_transverse_modulus_mpa,
                        "radius_1_mm": detail.minimum_radius_1_mm,
                        "radius_2_mm": detail.minimum_radius_2_mm,
                        "feasible": detail.feasible,
                        "constraint_violation": result.constraint_violation[i],
                        "pareto_member": result.pareto_mask[i],
                        "optimization_objective_1": values[0],
                        "optimization_objective_2": values[1],
                    }
                )
    return pd.DataFrame(final_records), pd.DataFrame(histories), results


def summarize(final: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    final_hv = history.loc[history.groupby(["algorithm", "seed"])["generation"].idxmax()]
    summary = (
        final_hv.groupby("algorithm")["hypervolume"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    algorithms = sorted(final_hv["algorithm"].unique())
    if len(algorithms) == 2:
        a = final_hv[final_hv.algorithm == algorithms[0]].sort_values("seed")["hypervolume"].to_numpy()
        b = final_hv[final_hv.algorithm == algorithms[1]].sort_values("seed")["hypervolume"].to_numpy()
        statistic, p_value = wilcoxon(a, b, zero_method="zsplit")
        summary["paired_wilcoxon_W"] = statistic
        summary["paired_wilcoxon_p"] = p_value
    feasible_counts = final.groupby(["algorithm", "seed"])["feasible"].mean().groupby("algorithm").mean()
    summary["mean_final_feasible_fraction"] = summary.algorithm.map(feasible_counts)
    return summary


def make_figures(final: pd.DataFrame, history: pd.DataFrame, figure_dir: Path) -> None:
    apply_latex_style(8.0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"MO-ETPSO-R": BLACK, "NSGA-II": GRAY_DARK}
    markers = {"MO-ETPSO-R": "o", "NSGA-II": "s"}
    line_styles = {"MO-ETPSO-R": "-", "NSGA-II": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95))
    ax = axes[0]
    for algorithm, group in final[(final.feasible) & (final.pareto_member)].groupby("algorithm"):
        ax.scatter(
            group.area_per_wavelength_mm,
            group.inertia_per_wavelength_mm3,
            s=16,
            alpha=0.65,
            color=colors[algorithm],
            marker=markers[algorithm],
            facecolors="none",
            label=algorithm,
            linewidths=0.65,
        )
    ax.set_xlabel(r"Normalized material area, $A/\lambda$ (mm)")
    ax.set_ylabel(r"Geometric inertia, $I/\lambda$ (mm$^3$)")
    ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for algorithm, group in history.groupby("algorithm"):
        pivot = group.pivot(index="generation", columns="seed", values="hypervolume")
        x = pivot.index.to_numpy() * (group.evaluations.iloc[0] // group.generation.iloc[0])
        median = pivot.median(axis=1).to_numpy()
        q1 = pivot.quantile(0.25, axis=1).to_numpy()
        q3 = pivot.quantile(0.75, axis=1).to_numpy()
        ax.plot(x, median, color=colors[algorithm], linestyle=line_styles[algorithm], label=algorithm)
        band = GRAY_LIGHT if algorithm == "MO-ETPSO-R" else GRAY_PALE
        ax.fill_between(x, q1, q3, color=band, alpha=0.9, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Function evaluations (log scale)")
    ax.set_ylabel("Scaled-objective hypervolume")
    ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"pareto_and_convergence.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pareto = final[(final.feasible) & (final.pareto_member)].copy()
    pareto["selection_score"] = (
        (pareto.area_per_wavelength_mm - pareto.area_per_wavelength_mm.min())
        / pareto.area_per_wavelength_mm.std()
        - (pareto.inertia_per_wavelength_mm3 - pareto.inertia_per_wavelength_mm3.min())
        / pareto.inertia_per_wavelength_mm3.std()
    ).abs()
    selected = pareto.sort_values("selection_score").groupby("algorithm").head(1)
    fig, ax = plt.subplots(figsize=(7.15, 2.25))
    for offset, (_, row) in enumerate(selected.iterrows()):
        design = row[[f"x{i}" for i in range(1, 8)]].to_numpy(float)
        x, y, (a, _, b) = nurbs_profile(
            design,
            sample_size=1000,
            wavelength_mm=WAVELENGTH_MM,
            amplitude_mm=AMPLITUDE_MM,
        )
        x, y = x[a:b], y[a:b]
        x -= x.min()
        ax.plot(
            x, y + offset * 4.0, lw=1.6, color=colors[row.algorithm],
            linestyle=line_styles[row.algorithm], label=row.algorithm,
        )
        ax.fill_between(
            x, y + offset * 4.0 - 0.1, y + offset * 4.0 + 0.1,
            color=GRAY_LIGHT, alpha=0.7,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, WAVELENGTH_MM + 0.2)
    ax.set_ylim(-0.35, 4.0 + AMPLITUDE_MM + 0.35)
    ax.set_xlabel("Machine direction (mm)")
    ax.set_ylabel("Profile ordinate (offset, mm)")
    ax.grid(color=GRAY_LIGHT, linewidth=0.4)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"representative_optimized_profiles.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--sample-size", type=int, default=300)
    args = parser.parse_args()
    results_dir = ROOT / "results" / "paper_a"
    figures_dir = ROOT / "figures" / "paper_a"
    results_dir.mkdir(parents=True, exist_ok=True)
    final, history, _ = run_benchmark(args.seeds, args.population, args.generations, args.sample_size)
    final.to_csv(results_dir / "benchmark_final_populations.csv", index=False)
    history.to_csv(results_dir / "benchmark_hypervolume_history.csv", index=False)
    summary = summarize(final, history)
    summary.to_csv(results_dir / "benchmark_summary.csv", index=False)
    make_figures(final, history, figures_dir)
    metadata = {
        "seeds": args.seeds,
        "population": args.population,
        "generations": args.generations,
        "evaluations_per_run": args.population * args.generations,
        "sample_size_during_search": args.sample_size,
        "final_re_evaluation_sample_size": 800,
        "radius_limit_mm": 0.9,
        "wavelength_mm": WAVELENGTH_MM,
        "amplitude_mm": AMPLITUDE_MM,
        "paper_thickness_mm": THICKNESS_MM,
        "hypervolume_reference_scaled_objectives": REFERENCE.tolist(),
        "benchmark_configuration": "fixed seven-variable geometric design with continuous radius violation",
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
