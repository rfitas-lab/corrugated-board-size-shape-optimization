#!/usr/bin/env python3
"""Short, resolution-aware verification pipeline for Paper A.

The complete 24-element direct-FEM ledger is reused.  Only three representative
members per Pareto case are solved at 32 elements, and the compromise member of
each case is refined at 40 elements.  This is the default publication workflow; the former
all-candidate 24/32 plus C038 40/48 sweep is intentionally not repeated.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

import run_paper_a_physics_optimization as full  # noqa: E402
from cbopt.mechanical_evaluator import (  # noqa: E402
    CompressionProtocol,
    paper_a_profile_nodes,
    protocol_fingerprint,
    solve_compression_path,
)
from plot_style import (  # noqa: E402
    BLACK,
    GRAY,
    GRAY_LIGHT,
    HIGHLIGHT_GOLD,
    HIGHLIGHT_RED,
    apply_latex_style,
)


RESULTS = ROOT / "results" / "paper_a" / "physics_optimization"
FIGURES = ROOT / "figures" / "paper_a"
TABLES = ROOT / "manuscripts" / "paper_a" / "generated_tables"
COARSE_MESH = 24
REPRESENTATIVE_MESH = 32
REFINED_MESH = 40
METRICS = (
    "target_reaction_N",
    "stored_potential_number",
    "stress_pnorm_utilization",
    "stress_q99_utilization",
    "stress_max_utilization",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_thresholds() -> dict[str, float]:
    terminal = pd.read_csv(RESULTS / "fem_terminal_mesh16.csv")
    success = terminal.path_success.fillna(False).astype(bool)
    training = terminal[success & terminal["split"].eq("train")]
    validation = json.loads((RESULTS / "surrogate_validation.json").read_text())
    return {
        "trust_limit": float(validation["trust_distance_limit"]),
        "minimum_potential": float(
            np.quantile(training.stored_potential_number, 0.40)
        ),
        "minimum_potential_training_quantile": 0.40,
        "maximum_material_fraction": float(
            np.quantile(training.board_material_fraction, 0.65)
        ),
        "maximum_material_training_quantile": 0.65,
    }


def rebuild_threshold_sensitivity(verified: pd.DataFrame) -> pd.DataFrame:
    """Recompute nearby training-quantile choices on the current mesh-24 ledger."""

    terminal = pd.read_csv(RESULTS / "fem_terminal_mesh16.csv")
    training = terminal[
        terminal.path_success.fillna(False).astype(bool)
        & terminal["split"].eq("train")
    ]
    valid = verified[
        verified.path_success.fillna(False).astype(bool)
        & (verified.radius_min_mm >= 0.9)
    ].copy()
    records: list[dict[str, object]] = []
    for quantile in (0.30, 0.40, 0.50):
        threshold = float(
            np.quantile(training.stored_potential_number, quantile)
        )
        group = valid[
            valid.case.eq("C038")
            & (valid.stored_potential_number >= threshold)
        ].copy()
        values = group[
            ["board_material_fraction", "stress_pnorm_utilization"]
        ].to_numpy(float)
        front = group[full.nondominated(values)] if len(group) else group
        records.append(
            {
                "case": "C038",
                "threshold_kind": "minimum stored-potential number",
                "training_quantile": quantile,
                "threshold_value": threshold,
                "feasible_classifications": len(group),
                "front_members": len(front),
                "dominance_rule": "strict mesh-24 dominance",
            }
        )
    for quantile in (0.55, 0.65, 0.75):
        threshold = float(
            np.quantile(training.board_material_fraction, quantile)
        )
        group = valid[
            valid.case.eq("C039")
            & (valid.board_material_fraction <= threshold)
        ].copy()
        values = np.column_stack(
            (
                1.0
                / np.maximum(
                    group.stored_potential_number.to_numpy(float), 1.0e-12
                ),
                group.stress_pnorm_utilization.to_numpy(float),
            )
        )
        front = group[full.nondominated(values)] if len(group) else group
        records.append(
            {
                "case": "C039",
                "threshold_kind": "maximum board material fraction",
                "training_quantile": quantile,
                "threshold_value": threshold,
                "feasible_classifications": len(group),
                "front_members": len(front),
                "dominance_rule": "strict mesh-24 dominance",
            }
        )
    sensitivity = pd.DataFrame(records)
    sensitivity.to_csv(RESULTS / "threshold_sensitivity_fast.csv", index=False)
    return sensitivity


def run_formulation_verification() -> dict[str, object]:
    """Expose analytical element checks and one small contact-path benchmark."""

    from autograd import grad
    from cbfem import BeamMaterial, BeamSection, CorotationalBeamModel

    length = 1.5
    modulus = 2899.0
    shear_modulus = modulus / 55.0
    section = BeamSection(25.0, 0.15)
    model = CorotationalBeamModel(
        np.array([[0.0, 0.0], [length, 0.0]]),
        BeamMaterial(modulus, shear_modulus),
        section,
    )
    theta = 1.0e-6
    shear_state = np.array([0.0, 0.0, theta, 0.0, 0.0, theta])
    expected_shear = (
        0.5
        * section.shear_correction
        * shear_modulus
        * section.area_mm2
        * length
        * theta**2
    )
    observed_shear = float(model._element_energy(shear_state))
    bending_state = np.array([0.0, 0.0, -theta, 0.0, 0.0, theta])
    curvature = 2.0 * theta / length
    expected_bending = (
        0.5 * modulus * section.inertia_mm4 * length * curvature**2
    )
    observed_bending = float(model._element_energy(bending_state))
    trial = np.array([0.0, 0.0, -2.0e-4, 2.0e-4, -1.0e-4, 3.0e-4])
    analytical_gradient = np.asarray(grad(model._element_energy)(trial))
    step = 1.0e-7
    basis = np.eye(6)
    numerical_gradient = np.array(
        [
            (
                float(model._element_energy(trial + step * basis[index]))
                - float(model._element_energy(trial - step * basis[index]))
            )
            / (2.0 * step)
            for index in range(6)
        ]
    )
    gradient_relative_error = float(
        np.max(np.abs(analytical_gradient - numerical_gradient))
        / max(np.max(np.abs(analytical_gradient)), 1.0e-12)
    )
    verification_position = np.array(
        [
            1.6192895867,
            10.0,
            9.3909827279,
            10.0,
            7.6931263431,
            0.0,
            0.0,
            7.9,
            4.0,
            0.15,
        ]
    )
    _, contact = solve_compression_path(
        verification_position,
        elements_per_wavelength=8,
        protocol=CompressionProtocol(strains=(0.0, 0.05)),
    )
    metrics: dict[str, object] = {
        "constant_shear_energy_relative_error": abs(
            observed_shear - expected_shear
        )
        / max(abs(expected_shear), 1.0e-30),
        "pure_bending_energy_relative_error": abs(
            observed_bending - expected_bending
        )
        / max(abs(expected_bending), 1.0e-30),
        "energy_gradient_max_relative_error": gradient_relative_error,
        "contact_benchmark_mesh": 8,
        "contact_benchmark_strains": [0.0, 0.05],
        "contact_path_success": bool(contact["path_success"]),
        "contact_initial_gap_mm": max(
            abs(float(contact["initial_upper_contact_gap_mm"])),
            abs(float(contact["initial_lower_contact_gap_mm"])),
        ),
        "contact_reaction_imbalance_force_fraction": float(
            contact["maximum_reaction_imbalance_force_fraction"]
        ),
        "contact_normalized_projected_gradient": float(
            contact["maximum_normalized_projected_gradient"]
        ),
    }
    (RESULTS / "formulation_verification.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def rebuild_coarse_front(thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.read_csv(RESULTS / "direct_verification_pool.csv")
    expected_geometries = pool.drop_duplicates(full.VARIABLES)
    mechanics = pd.read_csv(RESULTS / f"fem_unique_geometries_mesh{COARSE_MESH}.csv")
    if len(expected_geometries) != 572 or len(mechanics) != 572:
        raise RuntimeError("The complete 572-geometry mesh-24 ledger is required")
    if set(mechanics.protocol_fingerprint.astype(str)) != {
        protocol_fingerprint(COARSE_MESH)
    }:
        raise RuntimeError("The mesh-24 protocol fingerprint is stale")
    verified = full.verify_candidates(pool, mesh=COARSE_MESH, thresholds=thresholds)
    if len(verified) != len(pool) or not verified.path_success.fillna(False).all():
        raise RuntimeError("Not every coarse classification has a successful FEM record")
    finite = np.isfinite(verified.fem_objective_1) & np.isfinite(
        verified.fem_objective_2
    )
    front = verified[verified.fem_pareto & finite].copy()
    if set(front.case) != {"C037", "C038", "C039"}:
        raise RuntimeError("All three optimization cases must have a coarse Pareto front")
    verified.to_csv(RESULTS / "fem_verified_terminal_fast.csv", index=False)
    front.to_csv(RESULTS / "fem_rebuilt_pareto_fronts_fast.csv", index=False)
    front.to_csv(RESULTS / "fem_rebuilt_pareto_fronts.csv", index=False)
    return verified, front


def select_representatives(front: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    roles = ("objective_1_extreme", "compromise", "objective_2_extreme")
    for case, group in front.groupby("case", sort=True):
        group = group.sort_values("fem_objective_1").reset_index(drop=True)
        objectives = group[["fem_objective_1", "fem_objective_2"]].to_numpy(float)
        span = np.maximum(np.ptp(objectives, axis=0), 1.0e-12)
        normalized = (objectives - objectives.min(axis=0)) / span
        compromise = int(np.argmin(np.linalg.norm(normalized, axis=1)))
        indices = (0, compromise, len(group) - 1)
        for index, role in zip(indices, roles):
            row = group.iloc[index].to_dict()
            row["selection_role"] = role
            records.append(row)
    selected = pd.DataFrame(records).drop_duplicates(
        subset=["case", *full.VARIABLES]
    )
    if len(selected) != 9 or len(selected.drop_duplicates(full.VARIABLES)) != 9:
        raise RuntimeError("Representative selection must contain nine unique geometries")
    selected.insert(0, "selection_id", [f"S{i + 1}" for i in range(len(selected))])
    selected.to_csv(RESULTS / "selected_optimized_geometries.csv", index=False)
    return selected


def refinement_subset(selected: pd.DataFrame) -> pd.DataFrame:
    keep = selected.selection_role.eq("compromise")
    refined = selected[keep].copy()
    if len(refined) != 3 or set(refined.case) != {"C037", "C038", "C039"}:
        raise RuntimeError("The refined subset must contain one compromise per case")
    refined.to_csv(RESULTS / "selected_refined_geometries.csv", index=False)
    return refined


def _solve_selected_worker(
    row: dict[str, object], mesh: int
) -> tuple[
    dict[str, object],
    str,
    np.ndarray,
    list[dict[str, object]],
    dict[str, float],
]:
    """Worker-safe representative solve; file writes remain in the parent."""

    position = np.asarray([row[name] for name in full.VARIABLES], dtype=float)
    states, metrics = solve_compression_path(
        position,
        elements_per_wavelength=mesh,
        protocol=CompressionProtocol(strains=full.STRAINS),
    )
    nodes, _ = paper_a_profile_nodes(position, mesh)
    return (
        {**full._position_row(position), **metrics},
        str(row["selection_id"]),
        nodes,
        states,
        metrics,
    )


def evaluate_selected(
    selected: pd.DataFrame,
    mesh: int,
    *,
    require_states: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, list[dict[str, object]], dict[str, float]]]]:
    output = RESULTS / f"fem_representative_geometries_mesh{mesh}.csv"
    state_output = RESULTS / f"fem_representative_states_mesh{mesh}.pkl.gz"
    expected_protocol = protocol_fingerprint(mesh)
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame()
    if not existing.empty and (
        "protocol_fingerprint" not in existing
        or set(existing.protocol_fingerprint.astype(str)) != {expected_protocol}
    ):
        existing = pd.DataFrame()
    broad_cache = RESULTS / f"fem_unique_geometries_mesh{mesh}.csv"
    if broad_cache.exists():
        broad = pd.read_csv(broad_cache)
        if (
            "protocol_fingerprint" in broad
            and set(broad.protocol_fingerprint.astype(str)) == {expected_protocol}
        ):
            existing = pd.concat([existing, broad], ignore_index=True, sort=False)
    existing = full.add_geometry_key(existing) if not existing.empty else existing
    requested = full.add_geometry_key(selected)
    if not existing.empty:
        existing = existing[existing.geometry_key.isin(requested.geometry_key)]
        existing = existing.drop_duplicates("geometry_key", keep="first")
    completed = set(existing.geometry_key) if not existing.empty else set()
    records = existing.drop(columns="geometry_key", errors="ignore").to_dict("records")
    refined_states: dict[
        str, tuple[np.ndarray, list[dict[str, object]], dict[str, float]]
    ] = {}

    if require_states and state_output.exists():
        with gzip.open(state_output, "rb") as stream:
            loaded_states = pickle.load(stream)
        if not isinstance(loaded_states, dict):
            raise RuntimeError(f"Invalid representative-state cache: {state_output}")
        refined_states = {
            str(selection_id): bundle
            for selection_id, bundle in loaded_states.items()
            if str(selection_id) in set(requested.selection_id.astype(str))
        }

    missing_mechanics = ~requested.geometry_key.isin(completed)
    missing_states = (
        ~requested.selection_id.astype(str).isin(refined_states)
        if require_states
        else np.zeros(len(requested), dtype=bool)
    )
    pending = requested[missing_mechanics | missing_states]

    def save_records() -> None:
        frame = full.add_geometry_key(pd.DataFrame(records))
        frame = frame.drop_duplicates("geometry_key", keep="last")
        frame = frame.drop(columns="geometry_key")
        temporary = output.with_name(f"{output.name}.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(output)

    def save_states() -> None:
        temporary = state_output.with_name(f"{state_output.name}.tmp")
        with gzip.open(temporary, "wb") as stream:
            pickle.dump(refined_states, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(state_output)

    if len(pending):
        workers = min(3, len(pending))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_solve_selected_worker, row.to_dict(), mesh): str(
                    row.selection_id
                )
                for _, row in pending.iterrows()
            }
            for ordinal, future in enumerate(as_completed(futures), start=1):
                record, selection_id, nodes, states, metrics = future.result()
                records.append(record)
                refined_states[selection_id] = (nodes, states, metrics)
                save_records()
                if require_states:
                    save_states()
                print(
                    f"Representative FEM mesh {mesh}: {ordinal}/{len(pending)}",
                    flush=True,
                )
    elif records and not output.exists():
        save_records()

    mechanics = pd.read_csv(output)
    keyed = full.add_geometry_key(mechanics)
    available = set(keyed.geometry_key)
    missing = set(requested.geometry_key) - available
    if missing:
        raise RuntimeError(f"Missing {len(missing)} representative mesh-{mesh} records")
    used = keyed[keyed.geometry_key.isin(requested.geometry_key)]
    if not used.path_success.fillna(False).astype(bool).all():
        failures = int((~used.path_success.fillna(False).astype(bool)).sum())
        raise RuntimeError(
            f"{failures} representative mesh-{mesh} paths failed strict acceptance"
        )
    if require_states:
        requested_ids = set(requested.selection_id.astype(str))
        missing_state_ids = requested_ids - set(refined_states)
        if missing_state_ids:
            raise RuntimeError(
                f"Missing representative mesh-{mesh} states: "
                + ", ".join(sorted(missing_state_ids))
            )
    return mechanics, refined_states


def long_mesh_table(
    selected: pd.DataFrame,
    coarse: pd.DataFrame,
    mesh32: pd.DataFrame,
    refined: pd.DataFrame,
    refined_mesh: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for selection, mechanics, mesh in (
        (selected, coarse, COARSE_MESH),
        (selected, mesh32, REPRESENTATIVE_MESH),
        (refined, refined_mesh, REFINED_MESH),
    ):
        left = full.add_geometry_key(selection)[
            ["selection_id", "case", "selection_role", "geometry_key", *full.VARIABLES]
        ]
        right = full.add_geometry_key(mechanics).drop(columns=full.VARIABLES)
        merged = left.merge(right, on="geometry_key", validate="many_to_one")
        merged.insert(3, "mesh", mesh)
        frames.append(merged)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data.to_csv(RESULTS / "representative_mesh_validation.csv", index=False)
    return data


def convergence_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for selection_id, group in data.groupby("selection_id", sort=True):
        group = group.set_index("mesh")
        row: dict[str, object] = {
            "selection_id": selection_id,
            "case": group.case.iloc[0],
            "selection_role": group.selection_role.iloc[0],
        }
        for metric in METRICS:
            row[f"{metric}_relative_change_24_to_32"] = float(
                abs(group.loc[32, metric] - group.loc[24, metric])
                / max(abs(group.loc[32, metric]), 1.0e-12)
            )
            row[f"{metric}_relative_change_32_to_40"] = (
                float(
                    abs(group.loc[REFINED_MESH, metric] - group.loc[32, metric])
                    / max(abs(group.loc[REFINED_MESH, metric]), 1.0e-12)
                )
                if REFINED_MESH in group.index
                else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "representative_convergence_summary.csv", index=False)
    return summary


def plot_fronts(verified: pd.DataFrame, front: pd.DataFrame, selected: pd.DataFrame) -> None:
    apply_latex_style(7.2)
    labels = {
        "C037": (r"board material fraction $\phi_b$", r"inverse stored potential $1/\Psi_U$"),
        "C038": (r"board material fraction $\phi_b$", r"stress utilization $\Omega_8$"),
        "C039": (r"inverse stored potential $1/\Psi_U$", r"stress utilization $\Omega_8$"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for axis, case in zip(axes, ("C037", "C038", "C039")):
        feasible = verified[
            verified.case.eq(case)
            & verified.path_success
            & np.isclose(verified.fem_constraint_violation, 0.0)
        ]
        case_front = front[front.case.eq(case)].sort_values("fem_objective_1")
        case_selected = selected[selected.case.eq(case)]
        axis.scatter(
            feasible.fem_objective_1,
            feasible.fem_objective_2,
            s=8,
            facecolors="none",
            edgecolors=GRAY,
            linewidths=0.45,
        )
        axis.plot(
            case_front.fem_objective_1,
            case_front.fem_objective_2,
            color=HIGHLIGHT_RED,
            linewidth=1.0,
        )
        axis.scatter(
            case_selected.fem_objective_1,
            case_selected.fem_objective_2,
            s=24,
            color=HIGHLIGHT_GOLD,
            edgecolors=BLACK,
            linewidths=0.5,
            zorder=3,
        )
        axis.set(xlabel=labels[case][0], ylabel=labels[case][1], title=case)
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    fig.tight_layout(w_pad=0.8)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"physics_pareto_fronts.{suffix}", dpi=350)
    plt.close(fig)


def plot_convergence(summary: pd.DataFrame) -> None:
    apply_latex_style(7.2)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), sharey=True)
    fields = (
        ("stored_potential_number", r"$\Psi_U$"),
        ("stress_pnorm_utilization", r"$\Omega_8$"),
    )
    x = np.arange(len(summary))
    for axis, (field, label) in zip(axes, fields):
        axis.plot(
            x,
            100 * summary[f"{field}_relative_change_24_to_32"],
            marker="o",
            color=BLACK,
            linewidth=0.9,
            label="24--32",
        )
        refined = summary[f"{field}_relative_change_32_to_40"].notna()
        axis.plot(
            x[refined],
            100 * summary.loc[refined, f"{field}_relative_change_32_to_40"],
            marker="s",
            color=HIGHLIGHT_RED,
            linewidth=0.9,
            label="32--40",
        )
        axis.set_xticks(x, summary.selection_id, rotation=0)
        axis.set(
            xlabel="representative geometry",
            ylabel=r"relative change [\%]",
            title=label,
        )
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    axes[0].legend(frameon=False)
    fig.tight_layout(w_pad=1.0)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"representative_mesh_convergence.{suffix}", dpi=350)
    plt.close(fig)


def write_tables(
    verified: pd.DataFrame,
    front: pd.DataFrame,
    selected: pd.DataFrame,
    mesh_data: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    formulation: dict[str, object],
) -> None:
    linebreak = r"\\"
    objective_labels = {
        "C037": (r"$\phi_b$", r"$1/\Psi_U$"),
        "C038": (r"$\phi_b$", r"$\Omega_8$"),
        "C039": (r"$1/\Psi_U$", r"$\Omega_8$"),
    }
    rows = []
    for case, group in front.groupby("case", sort=True):
        coordinate_1, coordinate_2 = objective_labels[str(case)]
        rows.append(
            f"{case} & {len(group)} & "
            f"{coordinate_1} & {group.fem_objective_1.min():.4g}--{group.fem_objective_1.max():.4g} & "
            f"{coordinate_2} & {group.fem_objective_2.min():.4g}--{group.fem_objective_2.max():.4g} \\\\"
        )
    (TABLES / "physics_front_summary_fast.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Direct-FEM nondominated sets over the pooled candidate universe, reconstructed from the complete 24-element ledger.}\n"
        "\\label{tab:physics_front_summary_fast}\n\\begin{tabular}{lrlrlr}\n\\toprule\n"
        f"Case & Members & Coordinate 1 & Range & Coordinate 2 & Range {linebreak}\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    sensitivity_rows = []
    for row in sensitivity.itertuples(index=False):
        sensitivity_rows.append(
            f"{row.case} & {100 * row.training_quantile:.0f} & "
            f"{row.threshold_value:.6g} & {row.feasible_classifications} & "
            f"{row.front_members} {linebreak}"
        )
    (TABLES / "threshold_sensitivity_fast.tex").write_text(
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\caption{Dependence of the strict mesh-24 nondominated sets on nearby training-defined constraint quantiles.}\n"
        "\\label{tab:threshold_sensitivity_fast}\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        f"Case & Training quantile [\\%] & Threshold & Feasible & Members {linebreak}\n\\midrule\n"
        + "\n".join(sensitivity_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    rows = []
    for row in summary.itertuples(index=False):
        p24 = 100 * row.stored_potential_number_relative_change_24_to_32
        s24 = 100 * row.stress_pnorm_utilization_relative_change_24_to_32
        p40 = row.stored_potential_number_relative_change_32_to_40
        s40 = row.stress_pnorm_utilization_relative_change_32_to_40
        rows.append(
            f"{row.selection_id} & {row.case} & {p24:.2f} & {s24:.2f} & "
            f"{('--' if pd.isna(p40) else f'{100*p40:.2f}')} & "
            f"{('--' if pd.isna(s40) else f'{100*s40:.2f}')} \\\\"
        )
    (TABLES / "representative_mesh_convergence.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Relative changes (percent) for representative optimized geometries, using the finer-mesh response as denominator. All nine are checked at 24--32 elements; the three case compromises are also checked at 32--40.}\n"
        "\\label{tab:representative_mesh_convergence}\n\\begin{tabular}{llrrrr}\n\\toprule\n"
        f"ID & Case & $\\Psi_U^{{24\\to32}}$ & $\\Omega_8^{{24\\to32}}$ & $\\Psi_U^{{32\\to40}}$ & $\\Omega_8^{{32\\to40}}$ {linebreak}\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    (TABLES / "verification_pool_fast.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Mechanical evidence budget. Case classifications share mechanics when their ten-variable vectors coincide.}\n"
        "\\label{tab:verification_pool_fast}\n\\begin{tabular}{lr}\n\\toprule\n"
        f"Evidence set & Count {linebreak}\n\\midrule\n"
        f"Direct case classifications at mesh 24 & {len(verified)} {linebreak}\n"
        f"Unique mesh-24 mechanical geometries & 572 {linebreak}\n"
        f"Representative mesh-32 geometries & 9 {linebreak}\n"
        f"Compromise mesh-40 geometries & 3 {linebreak}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    audit_rows = []
    ledgers = (
        ("Training", 16, pd.read_csv(RESULTS / "fem_terminal_mesh16.csv")),
        ("Coarse reconstruction", 24, pd.read_csv(RESULTS / "fem_unique_geometries_mesh24.csv")),
        ("Representatives", 32, pd.read_csv(RESULTS / "fem_representative_geometries_mesh32.csv")),
        ("Case compromises", 40, pd.read_csv(RESULTS / "fem_representative_geometries_mesh40.csv")),
    )
    for label, mesh, data in ledgers:
        accepted = data.path_success.fillna(False).astype(bool)
        audit_rows.append(
            f"{label} & {mesh} & {len(data)} & {int(accepted.sum())} & "
            f"{int(data.fallback_solver_state_count.sum())} & {data.runtime_s.median():.2f} \\\\"
        )
    (TABLES / "solver_audit_fast.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Strict solver audit for the evidence used in the paper. Fallback counts are states accepted by the alternate SLSQP solve of the same energy and the common KKT check; runtime is the median complete-path time.}\n"
        "\\label{tab:solver_audit_fast}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        f"Ledger & Mesh & Paths & Accepted & Fallback states & Runtime [s] {linebreak}\n\\midrule\n"
        + "\n".join(audit_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    formulation_rows = [
        (
            "Constant-shear element energy",
            "relative error",
            float(formulation["constant_shear_energy_relative_error"]),
            1.0e-9,
        ),
        (
            "Pure-bending element energy",
            "relative error",
            float(formulation["pure_bending_energy_relative_error"]),
            1.0e-9,
        ),
        (
            "Element energy gradient",
            "maximum relative error",
            float(formulation["energy_gradient_max_relative_error"]),
            2.0e-5,
        ),
        (
            "Two-state contact path",
            "reaction imbalance",
            float(formulation["contact_reaction_imbalance_force_fraction"]),
            5.0e-4,
        ),
        (
            "Two-state contact path",
            "normalized projected gradient",
            float(formulation["contact_normalized_projected_gradient"]),
            5.0e-4,
        ),
    ]
    (TABLES / "formulation_verification.tex").write_text(
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\caption{Independent analytical and numerical checks of the element and contact implementation.}\n"
        "\\label{tab:formulation_verification}\n\\begin{tabular}{llrr}\n\\toprule\n"
        f"Check & Metric & Observed & Acceptance limit {linebreak}\n\\midrule\n"
        + "\n".join(
            f"{name} & {metric} & {observed:.3e} & {limit:.1e} {linebreak}"
            for name, metric, observed, limit in formulation_rows
        )
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    mesh32 = mesh_data[mesh_data.mesh.eq(32)].sort_values("selection_id")
    design_rows = []
    diagnostic_rows = []
    vector_rows = []
    role_names = {
        "objective_1_extreme": "objective-1 extreme",
        "compromise": "compromise",
        "objective_2_extreme": "objective-2 extreme",
    }
    for row in mesh32.itertuples(index=False):
        role = role_names[str(row.selection_role)]
        design_rows.append(
            f"{row.selection_id} & {row.case} & {role} & {row.radius_min_mm:.3f} & "
            f"{row.board_material_fraction:.3f} & {1000*row.stored_potential_number:.3f} & "
            f"{row.stress_pnorm_utilization:.3f} \\\\"
        )
        diagnostic_rows.append(
            f"{row.selection_id} & {row.crush_load_number:.4f} & {1000*row.external_work_number:.3f} & "
            f"{row.stress_q99_utilization:.3f} & {row.stress_max_utilization:.3f} & "
            f"{row.stress_localization_number:.2f} & {row.combined_work_material_stress_number:.4f} & "
            f"{row.forming_yield_index:.3f} & {row.dimensionless_secant_tangent:.5f} & "
            f"{row.target_yielded_length_fraction:.3f} \\\\"
        )
    for row in selected.sort_values("selection_id").itertuples(index=False):
        values = " & ".join(f"{float(getattr(row, name)):.5f}" for name in full.VARIABLES)
        vector_rows.append(f"{row.selection_id} & {row.case} & {values} {linebreak}")
    (TABLES / "selected_physics_designs_fast.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\small\n"
        "\\caption{Representative direct-FEM designs at 32 elements per period. $10^3\\Psi_U$ is shown for readability.}\n"
        "\\label{tab:selected_physics_designs_fast}\n\\begin{tabular}{lllrrrr}\n\\toprule\n"
        f"ID & Case & Role & $R_{{\\min}}$ [mm] & $\\phi_b$ & $10^3\\Psi_U$ & $\\Omega_8$ {linebreak}\n\\midrule\n"
        + "\n".join(design_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )
    (TABLES / "selected_response_diagnostics_fast.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\scriptsize\n"
        "\\caption{Dimensionless response diagnostics for the nine mesh-32 representatives.}\n"
        "\\label{tab:selected_response_diagnostics_fast}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
        f"ID & $C_F$ & $10^3\\Psi_W$ & $\\Omega_{{99}}$ & $\\Omega_{{\\max}}$ & $\\mathcal L_\\sigma$ & $\\mathcal I_{{WMS}}$ & $\\mathcal C_y$ & $\\Theta$ & $f_y$ {linebreak}\n\\midrule\n"
        + "\n".join(diagnostic_rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n",
        encoding="utf-8",
    )
    (TABLES / "selected_design_vectors.tex").write_text(
        "\\begin{table}[H]\n\\centering\\scriptsize\n"
        "\\caption{Complete ten-variable vectors for the representative designs.}\n"
        "\\label{tab:selected_design_vectors}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llrrrrrrrrrr}\n"
        f"\\toprule\nID & Case & $x_1$ & $x_2$ & $x_3$ & $x_4$ & $x_5$ & $x_6$ & $x_7$ & $P$ & $H$ & $t$ {linebreak}\n\\midrule\n"
        + "\n".join(vector_rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n",
        encoding="utf-8",
    )

    fidelity_rows = []
    for case, group in verified.groupby("case", sort=True):
        group = group[
            group.path_success.fillna(False).astype(bool)
            & np.isclose(group.fem_constraint_violation, 0.0)
        ]
        rho1 = group.surrogate_objective_1.corr(group.fem_objective_1, method="spearman")
        rho2 = group.surrogate_objective_2.corr(group.fem_objective_2, method="spearman")
        mare1 = np.median(
            np.abs(group.surrogate_objective_1 - group.fem_objective_1)
            / np.maximum(np.abs(group.fem_objective_1), 1.0e-12)
        )
        mare2 = np.median(
            np.abs(group.surrogate_objective_2 - group.fem_objective_2)
            / np.maximum(np.abs(group.fem_objective_2), 1.0e-12)
        )
        fidelity_rows.append(
            f"{case} & {len(group)} & {rho1:.3f} & {100*mare1:.2f} & {rho2:.3f} & {100*mare2:.2f} \\\\"
        )
    (TABLES / "decision_fidelity_fast.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Surrogate-to-direct decision fidelity over feasible mesh-24 classifications. $\\rho_s$ is Spearman correlation and MARE is median absolute relative error.}\n"
        "\\label{tab:decision_fidelity_fast}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        f"Case & $n$ & $\\rho_{{s,1}}$ & MARE$_1$ [\\%] & $\\rho_{{s,2}}$ & MARE$_2$ [\\%] {linebreak}\n\\midrule\n"
        + "\n".join(fidelity_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    runtime_rows = []
    for label, mesh, data in ledgers:
        runtime_rows.append(
            f"{label} & {mesh} & {len(data)} & {data.runtime_s.sum():.1f} & "
            f"{data.runtime_s.median():.2f} & {data.runtime_s.max():.2f} \\\\"
        )
    rejected = pd.read_csv(
        RESULTS / "fem_representative_geometries_mesh48_rejected_solver_audit.csv"
    )
    runtime_rows.append(
        f"Rejected audit (excluded) & 48 & {len(rejected)} & "
        f"{rejected.runtime_s.sum():.1f} & {rejected.runtime_s.median():.2f} & "
        f"{rejected.runtime_s.max():.2f} \\\\"
    )
    (TABLES / "runtime_summary_fast.tex").write_text(
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Measured complete-path FEM cost. Times are CPU-process seconds in the x86--64 container on an Intel Xeon Platinum 8573C; representative batches use at most three processes. Accepted evidence rows are followed by an excluded solver-audit row.}\n"
        "\\label{tab:runtime_summary_fast}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        f"Ledger & Mesh & Paths & Sum [s] & Median [s] & Max [s] {linebreak}\n\\midrule\n"
        + "\n".join(runtime_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    verified, front = rebuild_coarse_front(thresholds)
    sensitivity = rebuild_threshold_sensitivity(verified)
    formulation = run_formulation_verification()
    selected = select_representatives(front)
    refined = refinement_subset(selected)
    coarse = pd.read_csv(RESULTS / f"fem_unique_geometries_mesh{COARSE_MESH}.csv")
    mesh32, _ = evaluate_selected(selected, REPRESENTATIVE_MESH)
    refined_mechanics, refined_states = evaluate_selected(
        refined, REFINED_MESH, require_states=True
    )
    mesh_data = long_mesh_table(
        selected, coarse, mesh32, refined, refined_mechanics
    )
    convergence = convergence_summary(mesh_data)
    plot_fronts(verified, front, selected)
    plot_convergence(convergence)
    if len(refined_states) == len(refined):
        full.plot_optimized_gallery(
            refined,
            refined_states,
            filename="optimized_fem_stress_gallery",
            maximum_rows=5,
        )
    full.plot_dimensionless_map(
        pd.read_csv(RESULTS / "fem_terminal_mesh16.csv"), front
    )
    full.plot_validation(
        json.loads((RESULTS / "surrogate_validation.json").read_text())
    )
    write_tables(
        verified,
        front,
        selected,
        mesh_data,
        convergence,
        sensitivity,
        formulation,
    )
    summary = {
        "strategy": "complete mesh-24 front plus representative refinement",
        "coarse_classifications": int(len(verified)),
        "coarse_unique_geometries": 572,
        "coarse_front_members": {k: int(v) for k, v in front.groupby("case").size().items()},
        "mesh32_representatives": 9,
        "mesh40_refinements": 3,
        "formulation_verification_paths": 1,
        "avoided_new_all_candidate_solves": 1072,
        "rejected_mesh48_cpu_process_seconds": float(
            pd.read_csv(
                RESULTS
                / "fem_representative_geometries_mesh48_rejected_solver_audit.csv"
            ).runtime_s.sum()
        ),
        "thresholds": thresholds,
        "maximum_relative_change_24_to_32": {
            metric: float(convergence[f"{metric}_relative_change_24_to_32"].max())
            for metric in METRICS
        },
        "maximum_relative_change_32_to_40": {
            metric: float(convergence[f"{metric}_relative_change_32_to_40"].max())
            for metric in METRICS
        },
    }
    (RESULTS / "fast_verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    source_inputs = sorted(
        {
            *ROOT.glob("*.py"),
            *(ROOT / "src").rglob("*.py"),
            *(ROOT / "vendor").rglob("*.py"),
            *(ROOT.parent / "tests").glob("*.py"),
            ROOT / "requirements-paper-a.txt",
            ROOT / "requirements.txt",
            ROOT / "manuscripts" / "paper_a" / "main.tex",
            ROOT / "manuscripts" / "references.bib",
            ROOT.parent / "README.md",
            ROOT.parent / "CITATION.cff",
        }
    )
    model_inputs = sorted((RESULTS / "models").glob("*"))
    numerical_inputs = [
        ROOT / "results" / "paper_a" / "benchmark_final_populations.csv",
        ROOT / "results" / "paper_a" / "benchmark_hypervolume_history.csv",
        ROOT / "results" / "paper_a" / "benchmark_summary.csv",
        ROOT / "results" / "paper_a" / "mo_etpso_hyperparameter_study.csv",
        ROOT / "results" / "paper_a" / "mo_etpso_hyperparameter_summary.csv",
        ROOT / "results" / "paper_a" / "run_metadata.json",
        *(ROOT / "results" / "paper_a" / "coupled_size_shape").glob("*.csv"),
        RESULTS / "direct_verification_pool.csv",
        RESULTS / "compressed_evidence_manifest.json",
        RESULTS / "energy_surrogate_test_predictions.csv",
        RESULTS / "feasible_design_pool.csv",
        RESULTS / "fem_designs.csv",
        RESULTS / "fem_paths_mesh16.csv",
        RESULTS / "fem_terminal_mesh16.csv",
        RESULTS / "fem_unique_geometries_mesh24.csv",
        RESULTS / "fem_unique_geometries_mesh24.csv.gz",
        RESULTS / "fem_verified_terminal_fast.csv.gz",
        RESULTS / "optimization_convergence.csv",
        RESULTS / "optimization_convergence_summary.csv",
        RESULTS / "periodicity_audit.json",
        RESULTS / "stress_test_predictions.csv",
        RESULTS / "surrogate_pareto_candidates.csv",
        RESULTS / "surrogate_terminal_members.csv",
        RESULTS / "surrogate_validation.json",
        RESULTS / "fem_representative_geometries_mesh48_rejected_solver_audit.csv",
    ]
    numerical_outputs = [
        RESULTS / "fem_verified_terminal_fast.csv",
        RESULTS / "fem_rebuilt_pareto_fronts_fast.csv",
        RESULTS / "fem_rebuilt_pareto_fronts.csv",
        RESULTS / "selected_optimized_geometries.csv",
        RESULTS / "selected_refined_geometries.csv",
        RESULTS / "fem_representative_geometries_mesh32.csv",
        RESULTS / "fem_representative_geometries_mesh40.csv",
        RESULTS / "fem_representative_states_mesh40.pkl.gz",
        RESULTS / "representative_mesh_validation.csv",
        RESULTS / "representative_convergence_summary.csv",
        RESULTS / "threshold_sensitivity_fast.csv",
        RESULTS / "formulation_verification.json",
        RESULTS / "fast_verification_summary.json",
    ]
    publication_outputs = [
        FIGURES / "optimization_parameterizations.pdf",
        FIGURES / "pareto_and_convergence.pdf",
        FIGURES / "constrained_selected_designs.pdf",
        FIGURES / "pareto_atlas_5x3.pdf",
        FIGURES / "fem_energy_surrogate_validation.pdf",
        FIGURES / "physics_optimizer_convergence.pdf",
        FIGURES / "physics_pareto_fronts.pdf",
        FIGURES / "representative_mesh_convergence.pdf",
        FIGURES / "dimensionless_performance_map.pdf",
        FIGURES / "optimized_fem_stress_gallery.pdf",
        TABLES / "periodicity_audit.tex",
        TABLES / "physics_validation.tex",
        TABLES / "energy_training_diagnostics.tex",
        TABLES / "stress_calibration.tex",
        TABLES / "solver_audit_fast.tex",
        TABLES / "formulation_verification.tex",
        TABLES / "optimizer_convergence_summary.tex",
        TABLES / "physics_front_summary_fast.tex",
        TABLES / "threshold_sensitivity_fast.tex",
        TABLES / "decision_fidelity_fast.tex",
        TABLES / "representative_mesh_convergence.tex",
        TABLES / "selected_physics_designs_fast.tex",
        TABLES / "selected_response_diagnostics_fast.tex",
        TABLES / "runtime_summary_fast.tex",
        TABLES / "selected_design_vectors.tex",
        TABLES / "verification_pool_fast.tex",
    ]
    inputs = sorted(set(source_inputs + model_inputs + numerical_inputs))
    outputs = sorted(set(numerical_outputs + publication_outputs))

    def relative(path: Path) -> str:
        if path.is_relative_to(ROOT):
            return str(path.relative_to(ROOT))
        return str(Path("..") / path.relative_to(ROOT.parent))

    missing = [path for path in inputs + outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Manifest coverage contains missing files: "
            + ", ".join(str(path) for path in missing)
        )
    manifest = {
        "command": "python project/run_paper_a_fast_verification.py",
        "budget": {"mesh32": 9, "mesh40": 3, "mesh8_formulation_check": 1},
        "coverage": {
            "source_files": [relative(path) for path in source_inputs],
            "trained_model_files": [relative(path) for path in model_inputs],
            "numerical_input_files": [relative(path) for path in numerical_inputs],
            "numerical_output_files": [relative(path) for path in numerical_outputs],
            "manuscript_dependency_files": [
                relative(path) for path in publication_outputs
            ],
        },
        "inputs": {
            relative(path): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        },
        "outputs": {
            relative(path): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        },
    }
    (RESULTS / "fast_execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Completed short verification: {len(front)} coarse nondominated members, "
        "9 mesh-32 representatives, 3 mesh-40 refinements.",
        flush=True,
    )


if __name__ == "__main__":
    main()
