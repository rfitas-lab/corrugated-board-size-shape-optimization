#!/usr/bin/env python3
"""Reproduce, audit, and benchmark the Paper A optimization campaign."""

from __future__ import annotations

import argparse
import ast
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
    BLACK, GRAY_DARK, GRAY_LIGHT, GRAY_PALE, HIGHLIGHT_RED, apply_latex_style,
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


def audit_archived(repo_root: Path, output_dir: Path) -> pd.DataFrame:
    base = repo_root / "Scaled_parts" / "prob_CB_prod_fix_v5_WC" / "EPSO0"
    records = []
    for run_dir in sorted(base.glob("EPSO_*")):
        final = run_dir / "Iteration_99.csv"
        if not final.exists():
            continue
        frame = pd.read_csv(final)
        for row_number, row in frame.iterrows():
            serialized = str(row["X_all_values"]).replace("np.float64(", "").replace(")", "")
            design = np.asarray(ast.literal_eval(serialized), dtype=float)
            result = evaluate_design(
                design,
                sample_size=800,
                wavelength_mm=WAVELENGTH_MM,
                amplitude_mm=AMPLITUDE_MM,
                thickness_mm=THICKNESS_MM,
                radius_limit_mm=0.9,
            )
            records.append(
                {
                    "archive_run": run_dir.name,
                    "row": row_number,
                    **{f"x{i + 1}": design[i] for i in range(7)},
                    "archived_inverse_inertia": row["obj1"],
                    "recomputed_inverse_inertia": result.inverse_inertia_scaled,
                    "archived_section_area_m2": row["obj2"],
                    "recomputed_section_area_m2": result.section_area_m2,
                    "archived_effective_Ez_Pa": row["obj3"],
                    "recomputed_effective_Ez_Pa": result.effective_transverse_modulus_mpa * 1e6,
                    "radius_1_mm": result.minimum_radius_1_mm,
                    "radius_2_mm": result.minimum_radius_2_mm,
                    "physically_feasible_Rmin_0p9": result.feasible,
                }
            )
    audit = pd.DataFrame(records)
    if len(audit):
        audit["inverse_inertia_relative_error"] = (
            audit["recomputed_inverse_inertia"] - audit["archived_inverse_inertia"]
        ).abs() / audit["archived_inverse_inertia"].abs()
        audit["section_area_relative_error"] = (
            audit["recomputed_section_area_m2"] - audit["archived_section_area_m2"]
        ).abs() / audit["archived_section_area_m2"].abs()
        audit["Ez_relative_error"] = (
            audit["recomputed_effective_Ez_Pa"] - audit["archived_effective_Ez_Pa"]
        ).abs() / audit["archived_effective_Ez_Pa"].abs()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "archived_final_front_audit.csv", index=False)
    return audit


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
    ax.set_ylabel("Normalized 2D hypervolume")
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


def make_audit_figure(audit: pd.DataFrame, figure_dir: Path) -> None:
    """Show the engineering effect of the disabled archived radius constraint."""
    apply_latex_style(8.0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    area = audit.recomputed_section_area_m2.to_numpy() * 1e6 / WAVELENGTH_MM
    inverse = audit.recomputed_inverse_inertia.to_numpy()
    inertia = 1e3 / inverse
    feasible = audit.physically_feasible_Rmin_0p9.to_numpy(bool)
    radius = audit[["radius_1_mm", "radius_2_mm"]].min(axis=1).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7))
    axes[0].scatter(area[~feasible], inertia[~feasible], s=12, marker="x", color=GRAY_DARK, alpha=0.65, label="violates 0.9 mm")
    axes[0].scatter(area[feasible], inertia[feasible], s=16, marker="o", facecolor="none", edgecolor=BLACK, alpha=0.8, label="physically feasible")
    axes[0].set_xlabel(r"Reconstructed $A/\lambda$ (mm)")
    axes[0].set_ylabel(r"Reconstructed $I/\lambda$ (mm$^3$)")
    axes[0].grid(color=GRAY_LIGHT, linewidth=0.4)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].hist(radius, bins=np.linspace(0, 1.4, 29), color=GRAY_DARK, alpha=0.85)
    axes[1].axvline(0.9, color=HIGHLIGHT_RED, linestyle="--", linewidth=1.2, label="stated limit")
    axes[1].set_xlabel("Minimum reconstructed radius (mm)")
    axes[1].set_ylabel("Archived rows")
    axes[1].grid(color=GRAY_LIGHT, linewidth=0.4, axis="y")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"archived_constraint_audit.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def make_reproducibility_figure(
    audit: pd.DataFrame, history: pd.DataFrame, figure_dir: Path
) -> None:
    """Summarize reconstruction accuracy and paired-seed benchmark outcomes."""
    apply_latex_style(8.0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    error_columns = [
        ("inverse_inertia_relative_error", "inverse inertia", BLACK),
        ("section_area_relative_error", "section area", GRAY_DARK),
        ("Ez_relative_error", r"effective $E_z$", GRAY_LIGHT),
    ]
    positions = np.arange(len(error_columns))
    data = [np.maximum(audit[column].to_numpy(), 1e-8) for column, _, _ in error_columns]
    violin = axes[0].violinplot(data, positions=positions, widths=0.75, showmedians=True, showextrema=False)
    for body, (_, _, color) in zip(violin["bodies"], error_columns):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.75)
    violin["cmedians"].set_color("black")
    violin["cmedians"].set_linewidth(0.8)
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([label for _, label, _ in error_columns], rotation=15, ha="right")
    axes[0].set_ylabel("Archived-row relative error")
    axes[0].set_title("Recovered evaluator configuration")
    axes[0].grid(color=GRAY_LIGHT, linewidth=0.4, axis="y")

    final_hv = history.groupby(["algorithm", "seed"]).tail(1).pivot(index="seed", columns="algorithm", values="hypervolume")
    for seed, row in final_hv.iterrows():
        axes[1].plot(
            [row["NSGA-II"], row["MO-ETPSO-R"]],
            [seed, seed],
            color=GRAY_LIGHT,
            linewidth=0.8,
            zorder=1,
        )
    axes[1].scatter(final_hv["NSGA-II"], final_hv.index, marker="s", facecolor="none", edgecolor=GRAY_DARK, s=22, label="NSGA-II", zorder=2)
    axes[1].scatter(final_hv["MO-ETPSO-R"], final_hv.index, marker="o", facecolor="none", edgecolor=BLACK, s=22, label="MO-ETPSO-R", zorder=2)
    axes[1].set_xlabel("Final normalized hypervolume")
    axes[1].set_ylabel("Paired seed")
    axes[1].set_yticks(final_hv.index)
    axes[1].set_title("Paired-run outcomes")
    axes[1].grid(color=GRAY_LIGHT, linewidth=0.4, axis="x")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"reconstruction_and_paired_seeds.{suffix}", dpi=350, bbox_inches="tight")
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
    audit = audit_archived(ROOT.parent / "tmp" / "Opt_CB", results_dir)
    final, history, _ = run_benchmark(args.seeds, args.population, args.generations, args.sample_size)
    final.to_csv(results_dir / "benchmark_final_populations.csv", index=False)
    history.to_csv(results_dir / "benchmark_hypervolume_history.csv", index=False)
    summary = summarize(final, history)
    summary.to_csv(results_dir / "benchmark_summary.csv", index=False)
    make_figures(final, history, figures_dir)
    make_audit_figure(audit, figures_dir)
    make_reproducibility_figure(audit, history, figures_dir)
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
        "archive_parameter_source": "inferred exactly from saved objectives; repository source later changed",
        "hypervolume_reference_normalized": REFERENCE.tolist(),
        "archive_front_rows": int(len(audit)),
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
