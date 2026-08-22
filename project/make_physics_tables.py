#!/usr/bin/env python3
"""Create Paper A manuscript tables from the direct FEM result ledgers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from plot_style import BLACK, GRAY_DARK, GRAY_LIGHT, HIGHLIGHT_RED, apply_latex_style


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results" / "paper_a" / "physics_optimization"
TABLES = ROOT / "manuscripts" / "paper_a" / "generated_tables"
FIGURES = ROOT / "figures" / "paper_a"
E_MPA = 2899.0
SIGMA_Y_MPA = 60.0
VARIABLES = [
    "d1", "d2", "d3", "d4", "d5", "w1", "w2",
    "pitch_mm", "height_mm", "thickness_mm",
]


def add_study_numbers(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    required = {
        "radius_min_mm", "thickness_mm", "external_work_number",
        "board_material_fraction", "stress_pnorm_utilization",
    }
    if required.issubset(data.columns):
        data["forming_yield_index"] = (
            E_MPA * data.thickness_mm / (2.0 * SIGMA_Y_MPA * data.radius_min_mm)
        )
        data["work_per_material_number"] = (
            data.external_work_number / data.board_material_fraction
        )
        data["work_per_stress_number"] = (
            data.external_work_number / data.stress_pnorm_utilization
        )
        data["combined_work_material_stress_number"] = (
            data.external_work_number
            / (data.board_material_fraction * data.stress_pnorm_utilization)
        )
    return data


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def validation_table() -> None:
    summary = json.loads((RESULTS / "surrogate_validation.json").read_text())
    stress = pd.read_csv(RESULTS / "stress_test_predictions.csv")
    rows = [
        (
            r"Dimensional potential $U$", "structured energy surrogate",
            "complete path", summary["neural_energy_test"]["r2"],
            summary["neural_energy_test"]["relative_mae_nonzero"],
        ),
        (
            r"Potential number $\Psi_U$", "structured energy surrogate",
            "complete path", summary["neural_potential_number_test"]["r2"],
            summary["neural_potential_number_test"]["relative_mae_nonzero"],
        ),
        (
            r"Compression reaction $F$", "energy derivative",
            "complete path", summary["neural_reaction_test"]["r2"],
            summary["neural_reaction_test"]["relative_mae_nonzero"],
        ),
        (
            r"Stress utilization $\Omega_8$", "tree ensemble", r"20\% strain",
            r2_score(stress.stress_pnorm_utilization, stress.predicted_stress_utilization),
            np.mean(
                np.abs(stress.predicted_stress_utilization - stress.stress_pnorm_utilization)
                / stress.stress_pnorm_utilization
            ),
        ),
    ]
    body = "\n".join(
        f"{label} & {model} & {scope} & {score:.3f} & {100.0 * relative:.1f}\\% \\\\"
        for label, model, scope, score, relative in rows
    )
    write(
        TABLES / "physics_validation.tex",
        rf"""
\begin{{table*}}[t]
\centering\small
\caption{{Held-out design-level validation. Relative error is the mean absolute relative error; near-zero states are excluded from the path-level relative-error statistic.}}
\label{{tab:physics_validation}}
\begin{{tabular}}{{lllrr}}
\toprule
Response & Model & Scope & $R^2$ & Rel. error \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
    )

    calibration = summary["stress_upper_bound"]
    write(
        TABLES / "stress_calibration.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Descriptive additive stress calibration on the 20 held-out designs. The constant is reported as an error diagnostic; because it preserves every within-case rank, the optimizer uses the unshifted ensemble mean.}}
\label{{tab:stress_calibration}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
Target empirical coverage & {100.0 * calibration['target_empirical_coverage']:.0f}\% \\
Diagnostic margin $\Delta\Omega$ & {calibration['additive_calibration_margin']:.5f} \\
Observed empirical coverage & {100.0 * calibration['empirical_coverage']:.0f}\% \\
Largest residual underprediction after shift & {calibration['maximum_underprediction_after_calibration']:.5f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )

    training_lines = []
    for record in summary["training"]:
        succeeded = str(record["optimizer_success"]).strip().lower() == "true"
        status = "converged" if succeeded else "gradient iteration cap"
        training_lines.append(
            f"{int(record['seed'])} & {int(record['iterations'])} & "
            f"{record['loss']:.3e} & {status} \\\\"
        )
    write(
        TABLES / "energy_training_diagnostics.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Fixed-budget energy-surrogate fits. All seeds use the same 900-iteration cap; scientific acceptance is based on geometry-held-out response error rather than the optimizer termination flag.}}
\label{{tab:energy_training}}
\begin{{tabular}}{{rrrl}}
\toprule
Seed & Iterations & Final loss & Termination \\
\midrule
{chr(10).join(training_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def verification_pool_table() -> None:
    data = pd.read_csv(RESULTS / "direct_verification_pool.csv")
    designs = pd.read_csv(RESULTS / "fem_designs.csv")
    optimizer = pd.read_csv(RESULTS / "surrogate_terminal_members.csv")

    def keys(frame: pd.DataFrame) -> set[tuple[float, ...]]:
        return {
            tuple(np.round(row, 10))
            for row in frame[VARIABLES].to_numpy(dtype=float)
        }

    design_keys = keys(designs)
    lines = []
    for case, group in data.groupby("case"):
        group_keys = keys(group)
        optimizer_keys = keys(optimizer[optimizer.case == case])
        lines.append(
            f"{case} & {len(group_keys & optimizer_keys)} & "
            f"{len(group_keys & design_keys)} & "
            f"{len(group_keys & optimizer_keys & design_keys)} & {len(group)} \\\\"
        )
    unique = len(data.drop_duplicates(subset=VARIABLES))
    write(
        TABLES / "verification_pool.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Direct-verification pool after exact case--vector deduplication. Source columns may overlap. The case rows total {len(data)} classifications; vectors shared across cases reduce these to {unique} unique mechanical geometries per broad verification mesh.}}
\label{{tab:verification_pool}}
\begin{{tabular}}{{lrrrr}}
\toprule
Case & Terminal vectors & FEM-design vectors & Source overlap & Union \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def periodicity_audit_table() -> None:
    from cbopt.evaluator import nurbs_profile

    pool = pd.read_csv(RESULTS / "feasible_design_pool.csv")
    maximum_ordinate_gap = 0.0
    maximum_tangent_gap = 0.0
    for position in pool.to_numpy(dtype=float):
        x, y, (left, _, right) = nurbs_profile(
            position[:7],
            sample_size=801,
            wavelength_mm=position[7],
            amplitude_mm=position[8],
        )
        maximum_ordinate_gap = max(
            maximum_ordinate_gap, abs(float(y[left] - y[right]))
        )
        tangent_left = np.array(
            [x[left + 1] - x[left - 1], y[left + 1] - y[left - 1]]
        )
        tangent_right = np.array(
            [x[right + 1] - x[right - 1], y[right + 1] - y[right - 1]]
        )
        tangent_left /= np.linalg.norm(tangent_left)
        tangent_right /= np.linalg.norm(tangent_right)
        maximum_tangent_gap = max(
            maximum_tangent_gap,
            float(np.linalg.norm(tangent_left - tangent_right)),
        )
    write(
        TABLES / "periodicity_audit.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Numerical periodicity audit over the complete feasible design cloud. Unit tangents use centered differences on a grid containing both cell endpoints exactly.}}
\label{{tab:periodicity_audit}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
Profiles checked & {len(pool)} \\
Largest endpoint-ordinate gap [mm] & {maximum_ordinate_gap:.3e} \\
Largest unit-tangent mismatch & {maximum_tangent_gap:.3e} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def front_table() -> None:
    front = pd.read_csv(RESULTS / "fem_rebuilt_pareto_fronts.csv")
    descriptions = {
        "C037": r"$\min(\phi_b,1/\Psi_U)$",
        "C038": r"$\min(\phi_b,\Omega_8)$; $\Psi_U\ge\Psi_{\min}$",
        "C039": r"$\min(1/\Psi_U,\Omega_8)$; $\phi_b\le\phi_{\max}$",
    }
    qualification = {
        "C037": r"exact $24\cap32$",
        "C038": r"tolerance $40\cap48$",
        "C039": r"exact $24\cap32$",
    }
    lines = []
    for case, group in front.groupby("case"):
        lines.append(
            f"{case} & {descriptions[case]} & {qualification[case]} & {len(group)} & "
            f"{group.board_material_fraction.min():.3f}--{group.board_material_fraction.max():.3f} & "
            f"{1e3 * group.stored_potential_number.min():.3f}--{1e3 * group.stored_potential_number.max():.3f} & "
            f"{group.stress_pnorm_utilization.min():.3f}--{group.stress_pnorm_utilization.max():.3f} \\\\"
        )
    write(
        TABLES / "physics_front_summary.tex",
        rf"""
\begin{{table*}}[t]
\centering\small
\caption{{Resolution-qualified direct-FEM fronts. C037 and C039 require exact nondominated membership at 24 and 32 elements; C038 requires uncertainty-aware membership at 40 and 48 elements. The potential column reports $10^3\Psi_U$.}}
\label{{tab:physics_front_summary}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lllrrrr}}
\toprule
Case & Minimization problem & Qualification & Members & $\phi_b$ & $10^3\Psi_U$ & $\Omega_8$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table*}}
""",
    )


def decision_fidelity_table() -> None:
    data = pd.read_csv(RESULTS / "fem_verified_terminal_final.csv")
    lines = []
    for case, group in data.groupby("case"):
        group = group[
            group.path_success.astype(bool)
            & np.isclose(group.fem_constraint_violation, 0.0)
        ].copy()
        rho_1 = group[["surrogate_objective_1", "fem_objective_1"]].corr(
            method="spearman"
        ).iloc[0, 1]
        rho_2 = group[["surrogate_objective_2", "fem_objective_2"]].corr(
            method="spearman"
        ).iloc[0, 1]
        rel_1 = np.median(
            np.abs(group.surrogate_objective_1 - group.fem_objective_1)
            / np.maximum(np.abs(group.fem_objective_1), 1.0e-12)
        )
        rel_2 = np.median(
            np.abs(group.surrogate_objective_2 - group.fem_objective_2)
            / np.maximum(np.abs(group.fem_objective_2), 1.0e-12)
        )
        if case == "C038" and "tolerance_pareto" in group:
            raw_count = int(group.tolerance_pareto.fillna(False).astype(bool).sum())
        else:
            raw_count = int(group.fem_pareto.astype(bool).sum())
        lines.append(
            f"{case} & {len(group)} & {raw_count} & "
            f"{int(group.mesh_stable_pareto.astype(bool).sum())} & "
            f"{rho_1:.3f} & {rho_2:.3f} & {100 * rel_1:.1f}\\% & {100 * rel_2:.1f}\\% \\\\"
        )
    write(
        TABLES / "decision_fidelity.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Surrogate decision fidelity over successful, case-feasible classifications at the publication resolution: 32 elements for C037/C039 and 48 for C038. $\rho_s$ is Spearman rank correlation and MARE is median absolute relative error.}}
\label{{tab:decision_fidelity}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Case & Direct & Raw front & Qualified & $\rho_{{s,1}}$ & $\rho_{{s,2}}$ & MARE$_1$ & MARE$_2$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""",
    )


def selected_table() -> None:
    selection = pd.read_csv(RESULTS / "selected_optimized_geometries.csv")
    refined = add_study_numbers(pd.read_csv(RESULTS / "optimized_geometry_mesh_study.csv"))
    refined = refined[refined.mesh == refined.mesh.max()].copy()
    data = selection[["selection_id", "case"]].merge(
        refined, on="selection_id", suffixes=("_selected", "")
    )
    lines = []
    for _, row in data.iterrows():
        lines.append(
            f"{row.selection_id} & {row.source_case} & {row.pitch_mm:.3f} & {row.height_mm:.3f} & "
            f"{row.thickness_mm:.3f} & {row.radius_min_mm:.3f} & {row.board_material_fraction:.3f} & "
            f"{1e3 * row.stored_potential_number:.3f} & {row.stress_pnorm_utilization:.3f} & "
            f"{row.stress_max_utilization:.3f} & {1e3 * row.combined_work_material_stress_number:.2f} \\\\"
        )
    write(
        TABLES / "selected_physics_designs.tex",
        rf"""
\begin{{table*}}[t]
\centering\scriptsize
\caption{{Representative optimized geometries re-simulated at 48 elements per wavelength. $\mathcal I_{{WMS}}=\Psi_W/(\phi_b\Omega_8)$ is a study-defined selector; its column is multiplied by $10^3$.}}
\label{{tab:selected_physics_designs}}
\begin{{tabular}}{{llrrrrrrrrr}}
\toprule
ID & Case & $P$ & $H$ & $t$ & $R_{{\min}}$ & $\phi_b$ & $10^3\Psi_U$ & $\Omega_8$ & $\Omega_{{\max}}$ & $10^3\mathcal I_{{WMS}}$ \\
 & & mm & mm & mm & mm & & & & & \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
    )

    diagnostic_lines = []
    for _, row in data.iterrows():
        diagnostic_lines.append(
            f"{row.selection_id} & {row.forming_yield_index:.3f} & "
            f"{row.crush_load_number:.4f} & "
            f"{1e3 * row.external_work_number:.3f} & "
            f"{row.stress_q99_utilization:.3f} & "
            f"{row.stress_localization_number:.2f} & "
            f"{row.dimensionless_secant_tangent:.4f} & "
            f"{100 * row.target_yielded_length_fraction:.1f}\\% & "
            f"{1e5 * row.maximum_reaction_imbalance_force_fraction:.2f} \\\\"
        )
    write(
        TABLES / "selected_response_diagnostics.tex",
        rf"""
\begin{{table*}}[t]
\centering\scriptsize
\caption{{Dimensionless diagnostics for the representative geometries at 48 elements per wavelength. The final column reports $10^5\mathcal E_R$; $f_y$ is the terminal yielded-length fraction under the smooth deformation-potential law.}}
\label{{tab:selected_response_diagnostics}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
ID & $\mathcal C_y$ & $C_F$ & $10^3\Psi_W$ & $\Omega_{{99}}$ & $\mathcal L_\sigma$ & $\Theta$ & $f_y$ & $10^5\mathcal E_R$ \\
\midrule
{chr(10).join(diagnostic_lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
    )

    vector_lines = []
    for _, row in selection.iterrows():
        values = [row[name] for name in ("d1", "d2", "d3", "d4", "d5", "w1", "w2")]
        vector_lines.append(
            f"{row.selection_id} & {row.case} & "
            + " & ".join(f"{value:.5f}" for value in values)
            + f" & {row.pitch_mm:.5f} & {row.height_mm:.5f} & {row.thickness_mm:.5f} \\\\"
        )
    write(
        TABLES / "selected_design_vectors.tex",
        rf"""
\begin{{table}}[H]
\centering\scriptsize
\caption{{Complete decision vectors for the FEM-refined representative geometries.}}
\label{{tab:selected_design_vectors}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrrrrr}}
\toprule
ID & Case & $x_1$ & $x_2$ & $x_3$ & $x_4$ & $x_5$ & $x_6$ & $x_7$ & $P$ & $H$ & $t$ \\
\midrule
{chr(10).join(vector_lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""",
    )


def mesh_table() -> None:
    data = pd.read_csv(RESULTS / "optimized_geometry_mesh_study.csv")
    fields = [
        ("target_reaction_N", "$F$"),
        ("stored_potential_number", "$\\Psi_U$"),
        ("stress_pnorm_utilization", "$\\Omega_8$"),
        ("stress_q99_utilization", "$\\Omega_{99}$"),
        ("stress_max_utilization", "$\\Omega_{\\max}$"),
    ]
    lines = []
    for field, label in fields:
        changes = []
        for _, group in data.groupby("selection_id"):
            value_40 = float(group.loc[group.mesh == 40, field].iloc[0])
            value_48 = float(group.loc[group.mesh == 48, field].iloc[0])
            changes.append(abs(value_40 - value_48) / max(abs(value_48), 1.0e-12))
        lines.append(
            f"{label} & {100 * np.median(changes):.2f}\\% & {100 * np.max(changes):.2f}\\% \\\\"
        )
    write(
        TABLES / "mesh_convergence_summary.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Relative change from 40 to 48 elements per wavelength across the representative optimized geometries.}}
\label{{tab:mesh_convergence}}
\begin{{tabular}}{{lrr}}
\toprule
Response & Median & Maximum \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def membership_stability_table() -> None:
    summary = json.loads((RESULTS / "front_membership_stability.json").read_text())
    lines = []
    for case in ("C037", "C038", "C039"):
        values = summary[case]
        lines.append(
            f"{case} & {values['mesh24_front_members']} & {values['mesh32_front_members']} & "
            f"{values['retained_members']} & {values['jaccard_membership']:.3f} & "
            f"{100 * values['maximum_target_reaction_N_relative_change']:.2f}\\% & "
            f"{100 * values['maximum_stored_potential_number_relative_change']:.2f}\\% & "
            f"{100 * values['maximum_stress_pnorm_utilization_relative_change']:.2f}\\% \\\\"
        )
    write(
        TABLES / "mesh_membership_stability.tex",
        rf"""
\begin{{table*}}[t]
\centering\small
\caption{{Complete verification-pool comparison between 24 and 32 elements per wavelength. Retained designs are exact Pareto members at both resolutions. Response columns report the largest paired relative change.}}
\label{{tab:mesh_membership_stability}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Case & Front$_{{24}}$ & Front$_{{32}}$ & Retained & Jaccard & $\Delta F_{{\max}}$ & $\Delta\Psi_{{U,\max}}$ & $\Delta\Omega_{{8,\max}}$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""",
    )


def c038_fine_stability_table() -> None:
    values = json.loads((RESULTS / "c038_fine_mesh_stability.json").read_text())
    write(
        TABLES / "c038_fine_mesh_stability.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Fine-mesh uncertainty qualification for the narrow C038 material--stress front. The absolute stress tolerance equals the largest paired 40--48 change over accepted paths.}}
\label{{tab:c038_fine_stability}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
Exact front members, 40 elements & {values['mesh40_exact_front_members']} \\
Exact front members, 48 elements & {values['mesh48_exact_front_members']} \\
Absolute $\Omega_8$ tolerance & {values['stress_absolute_tolerance']:.5f} \\
Tolerance-front members, 40 elements & {values['mesh40_tolerance_front_members']} \\
Tolerance-front members, 48 elements & {values['mesh48_tolerance_front_members']} \\
Retained at both resolutions & {values['retained_tolerance_members']} \\
Tolerance-front Jaccard index & {values['tolerance_front_jaccard']:.3f} \\
Largest paired relative $\Omega_8$ change & {100 * values['maximum_40_to_48_stress_relative_change']:.2f}\% \\
Qualified $\phi_b$ range & {values['stable_material_fraction_min']:.3f}--{values['stable_material_fraction_max']:.3f} \\
Qualified $\Omega_8$ range & {values['stable_stress_utilization_min']:.3f}--{values['stable_stress_utilization_max']:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def c038_fine_plot() -> None:
    comparison = pd.read_csv(RESULTS / "c038_fine_mesh_stability.csv")
    values = json.loads((RESULTS / "c038_fine_mesh_stability.json").read_text())
    tolerance = float(values["stress_absolute_tolerance"])
    paired = comparison[
        comparison.path_success_mesh40.astype(bool)
        & comparison.path_success_mesh48.astype(bool)
    ].copy()
    apply_latex_style(7.5)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    x = paired.stress_pnorm_utilization_mesh40
    y = paired.stress_pnorm_utilization_mesh48
    limits = [min(x.min(), y.min()), max(x.max(), y.max())]
    axes[0].scatter(x, y, s=11, facecolors="none", edgecolors=GRAY_DARK, linewidths=0.5)
    axes[0].plot(limits, limits, color=HIGHLIGHT_RED, linewidth=0.9)
    axes[0].fill_between(
        limits,
        np.asarray(limits) - tolerance,
        np.asarray(limits) + tolerance,
        color=GRAY_LIGHT,
        alpha=0.45,
        linewidth=0,
    )
    axes[0].set(
        xlabel=r"40 elements: $\Omega_8$",
        ylabel=r"48 elements: $\Omega_8$",
    )
    axes[0].text(
        0.04,
        0.95,
        rf"$\delta_\Omega={tolerance:.4f}$",
        transform=axes[0].transAxes,
        va="top",
    )

    axes[1].scatter(
        paired.board_material_fraction_mesh48,
        paired.stress_pnorm_utilization_mesh48,
        s=10,
        facecolors="none",
        edgecolors=GRAY_LIGHT,
        linewidths=0.5,
        label="all direct",
    )
    front40 = paired[paired.tolerance_pareto_mesh40.astype(bool)].sort_values(
        "board_material_fraction_mesh40"
    )
    front48 = paired[paired.tolerance_pareto_mesh48.astype(bool)].sort_values(
        "board_material_fraction_mesh48"
    )
    stable = paired[
        paired.tolerance_pareto_mesh40.astype(bool)
        & paired.tolerance_pareto_mesh48.astype(bool)
    ].sort_values("board_material_fraction_mesh48")
    axes[1].plot(
        front40.board_material_fraction_mesh40,
        front40.stress_pnorm_utilization_mesh40,
        "o--",
        color=GRAY_DARK,
        markersize=2.8,
        linewidth=0.7,
        label="40 tolerance front",
    )
    axes[1].plot(
        front48.board_material_fraction_mesh48,
        front48.stress_pnorm_utilization_mesh48,
        "s-",
        color=BLACK,
        markersize=2.8,
        linewidth=0.8,
        label="48 tolerance front",
    )
    axes[1].scatter(
        stable.board_material_fraction_mesh48,
        stable.stress_pnorm_utilization_mesh48,
        s=22,
        color=HIGHLIGHT_RED,
        zorder=4,
        label="retained intersection",
    )
    axes[1].set(xlabel=r"board fraction $\phi_b$", ylabel=r"stress $\Omega_8$")
    axes[1].legend(frameon=False, fontsize=6)
    for axis in axes:
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    fig.tight_layout(w_pad=1.0)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"c038_fine_mesh_verification.{suffix}", dpi=350)
    plt.close(fig)


def threshold_sensitivity_table() -> None:
    data = pd.read_csv(RESULTS / "threshold_sensitivity.csv")
    lines = []
    for _, row in data.iterrows():
        if row.case == "C038":
            constraint = r"$\Psi_U\geq\Psi_{\min}$"
            value = rf"${1e3 * row.threshold_value:.3f}\times10^{{-3}}$"
        else:
            constraint = r"$\phi_b\leq\phi_{\max}$"
            value = f"{row.threshold_value:.3f}"
        lines.append(
            f"{row.case} & {constraint} & {100 * row.training_quantile:.0f}\\% & "
            f"{value} & {int(row.feasible_candidates)} & {int(row.front_members)} \\\\"
        )
    write(
        TABLES / "threshold_sensitivity.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Sensitivity of constrained-front size to thresholds defined only from the 60 training designs. Reclassification uses the publication-resolution mechanics: 48 elements for C038 and 32 for C039.}}
\label{{tab:threshold_sensitivity}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lllrrr}}
\toprule
Case & Constraint & Training quantile & Threshold & Feasible & Front \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""",
    )


def optimizer_convergence_table() -> None:
    data = pd.read_csv(RESULTS / "optimization_convergence_summary.csv")
    lines = []
    for case, group in data.groupby("case"):
        gain = 100 * group.relative_gain_42_to_final
        lines.append(
            f"{case} & {len(group)} & {group.final_normalized_hypervolume.mean():.4f} & "
            f"{gain.median():.3f}\\% & {gain.max():.3f}\\% \\\\"
        )
    write(
        TABLES / "optimizer_convergence_summary.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Seed-level optimizer convergence using a fixed hypervolume reference for each case. Gains compare generation 42 with generation 56.}}
\label{{tab:optimizer_convergence}}
\begin{{tabular}}{{lrrrr}}
\toprule
Case & Seeds & Mean final HV & Median gain & Maximum gain \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def solver_audit_table() -> None:
    ledgers = [("Training", 16, pd.read_csv(RESULTS / "fem_terminal_mesh16.csv"))]
    ledgers.extend(
        ("Unique verification", mesh, pd.read_csv(RESULTS / f"fem_unique_geometries_mesh{mesh}.csv"))
        for mesh in (24, 32, 40, 48)
    )
    lines = []
    for stage, mesh, data in ledgers:
        accepted = data[data.path_success.astype(bool)]
        work_potential_gap = (
            np.abs(
                accepted.external_work_Nmm
                - accepted.target_potential_energy_Nmm
            )
            / np.maximum(
                np.abs(accepted.target_potential_energy_Nmm), 1.0e-12
            )
        )
        lines.append(
            f"{stage} & {mesh} & {len(data)} & {len(accepted)} & "
            f"{int(accepted.all_raw_solver_success.astype(bool).sum())} & "
            f"{int((accepted.fallback_solver_state_count > 0).sum())} & "
            f"{int((accepted.maximum_subdivision_count > 1).sum())} & "
            f"{accepted.maximum_normalized_projected_gradient.max():.2e} & "
            f"{accepted.maximum_absolute_reaction_imbalance_N.max():.3e} & "
            f"{100 * accepted.maximum_reaction_imbalance_force_fraction.max():.3f}\\% & "
            f"{100 * work_potential_gap.max():.2f}\\% \\\\"
        )
    write(
        TABLES / "solver_audit.tex",
        rf"""
\begin{{table*}}[t]
\centering\scriptsize
\caption{{Nonlinear-solver and conservative-work audit. A state is accepted when the selected L-BFGS-B result has raw success and satisfies the normalized projected-gradient threshold, or when a separate SLSQP solve returns raw success and satisfies the same KKT check. Load subdivision is retried only when the attempted route is not accepted. Counts for 24--48 elements are unique geometries, not repeated case classifications. The final column is the largest $|W-U|/|U|$ from five-state trapezoidal work integration.}}
\label{{tab:solver_audit}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrrrr}}
\toprule
Stage & Mesh & Attempted & Accepted & All selected raw & Fallback paths & Subdivided paths & Max. projected gradient & Max. imbalance [N] & Imbalance/yield load & Max. work--potential gap \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table*}}
""",
    )


def runtime_table() -> None:
    frames = [
        ("Training paths", 16, pd.read_csv(RESULTS / "fem_terminal_mesh16.csv")),
        *[
            ("Unique verification", mesh, pd.read_csv(RESULTS / f"fem_unique_geometries_mesh{mesh}.csv"))
            for mesh in (24, 32, 40, 48)
        ],
    ]
    rows = [
        (stage, mesh, len(data), data.runtime_s.median(), data.runtime_s.sum())
        for stage, mesh, data in frames
    ]
    selected = pd.read_csv(RESULTS / "optimized_geometry_mesh_study.csv")
    rows.append((
        "Selected mesh study", "8--48", len(selected),
        selected.runtime_s.median(), selected.runtime_s.sum(),
    ))
    body = "\n".join(
        f"{stage} & {resolution} & {count} & {median:.2f} & {total:.1f} \\\\"
        for stage, resolution, count, median, total in rows
    )
    write(
        TABLES / "runtime_summary.tex",
        rf"""
\begin{{table}}[t]
\centering\small
\caption{{Measured complete-path runtimes on the execution host. Verification counts are unique mechanical geometries; values characterize this implementation and are not hardware-independent speed claims.}}
\label{{tab:runtime_summary}}
\begin{{tabular}}{{lrrrr}}
\toprule
Stage & Elements & Paths & Median [s] & Total [s] \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    validation_table()
    periodicity_audit_table()
    verification_pool_table()
    front_table()
    decision_fidelity_table()
    selected_table()
    mesh_table()
    membership_stability_table()
    c038_fine_stability_table()
    c038_fine_plot()
    threshold_sensitivity_table()
    optimizer_convergence_table()
    solver_audit_table()
    runtime_table()
    print("Physics tables generated.")


if __name__ == "__main__":
    main()
