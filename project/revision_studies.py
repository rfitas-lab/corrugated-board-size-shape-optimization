#!/usr/bin/env python3
"""Run the fixed-budget Paper A optimizer hyperparameter study."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plot_style import BLACK, GRAY_LIGHT, apply_latex_style  # noqa: E402


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


def main() -> None:
    paper_a_hyperparameters()


if __name__ == "__main__":
    main()
