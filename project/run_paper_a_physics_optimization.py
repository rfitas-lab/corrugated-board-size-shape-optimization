#!/usr/bin/env python3
"""Run the dimensionless FEM/neural-potential optimization study for Paper A.

The neural potential is trained only on FEM paths generated inside the Paper A
size--shape domain.  It screens the multi-objective search; every reported
front member is re-solved with the nonlinear FEM and Pareto membership is then
recomputed from the direct mechanical response.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from cbopt.mechanical_evaluator import (  # noqa: E402
    CompressionProtocol,
    dimensionless_geometry,
    paper_a_profile_nodes,
    protocol_fingerprint,
    solve_compression_path,
    surrogate_features,
)
from cbopt.evaluator import nurbs_profile  # noqa: E402
from cbopt.optimizers import run_mo_etpso  # noqa: E402
from cbenergy import fit_energy_network  # noqa: E402
from plot_style import (  # noqa: E402
    BLACK,
    GRAY,
    GRAY_DARK,
    GRAY_LIGHT,
    HIGHLIGHT_GOLD,
    HIGHLIGHT_RED,
    apply_latex_style,
)


LOWER = np.asarray([0.1] * 5 + [0.0, 0.0, 4.8, 2.2, 0.15])
UPPER = np.asarray([10.0] * 5 + [1.0, 1.0, 7.9, 4.0, 0.25])
VARIABLES = ["d1", "d2", "d3", "d4", "d5", "w1", "w2", "pitch_mm", "height_mm", "thickness_mm"]
STRAINS = (0.0, 0.05, 0.10, 0.15, 0.20)
SEEDS = (3, 11, 29, 47, 83)
TRAINING_MESH = 16
FINAL_MESHES = (24, 32)
C038_FINE_MESHES = (40, 48)
FEATURE_COUNT = 8
RESULTS = ROOT / "results" / "paper_a" / "physics_optimization"
FIGURES = ROOT / "figures" / "paper_a"
MODELS = RESULTS / "models"


def nondominated(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        dominated = np.all(values <= values[index], axis=1) & np.any(
            values < values[index], axis=1
        )
        dominated[index] = False
        if dominated.any():
            keep[index] = False
    return keep


def tolerance_nondominated(values: np.ndarray, absolute_tolerance: np.ndarray) -> np.ndarray:
    """Return points not robustly dominated beyond stated objective uncertainty."""

    values = np.asarray(values, dtype=float)
    tolerance = np.asarray(absolute_tolerance, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        robustly_dominates = np.all(
            values <= values[index] + tolerance, axis=1
        ) & np.any(values < values[index] - tolerance, axis=1)
        robustly_dominates[index] = False
        if robustly_dominates.any():
            keep[index] = False
    return keep


def _position_row(position: np.ndarray) -> dict[str, float]:
    return dict(zip(VARIABLES, map(float, position)))


def _row_position(row: pd.Series) -> np.ndarray:
    return row[VARIABLES].to_numpy(dtype=float)


def table_fingerprint(data: pd.DataFrame, columns: list[str]) -> str:
    """Hash an ordered numerical ledger for cache invalidation."""

    ordered = data[columns].copy()
    text_rows = []
    for row in ordered.itertuples(index=False, name=None):
        text_rows.append(
            ",".join(
                str(value) if isinstance(value, str) else f"{float(value):.12g}"
                for value in row
            )
        )
    return hashlib.sha256("\n".join(text_rows).encode("utf-8")).hexdigest()


def add_geometry_key(data: pd.DataFrame, *, decimals: int = 10) -> pd.DataFrame:
    """Attach a stable key for CSV round trips without merging on raw floats."""

    keyed = data.copy()
    values = keyed[VARIABLES].to_numpy(dtype=float)
    keyed["geometry_key"] = [
        ":".join(f"{value:.{decimals}f}" for value in row) for row in values
    ]
    return keyed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_fingerprint(
    data: pd.DataFrame,
    columns: list[str],
    *,
    mesh: int,
    thresholds: dict[str, float],
) -> str:
    """Hash the candidate ledger and every setting used in classification."""

    payload = {
        "candidate_ledger": table_fingerprint(data, columns),
        "mesh": int(mesh),
        "protocol": protocol_fingerprint(mesh),
        "thresholds": {key: float(value) for key, value in sorted(thresholds.items())},
        "radius_limit_mm": 0.9,
        "objective_cases": [37, 38, 39],
        "nondomination": "strict pairwise minimization",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_seed_designs() -> np.ndarray:
    rows = []
    source = ROOT / "results" / "paper_a" / "coupled_size_shape"
    for case in (34, 35, 36):
        data = pd.read_csv(source / f"C{case:03d}_all_members.csv")
        rows.extend(ast.literal_eval(value) for value in data["X_all_values"])
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for row in rows:
        position = np.asarray(row, dtype=float)
        unique[tuple(np.round(position, 8))] = position
    return np.vstack(list(unique.values()))


def geometric_pool(target_size: int = 1200, seed: int = 20260821) -> np.ndarray:
    """Build a feasible local design cloud around the prior coupled campaigns."""

    rng = np.random.default_rng(seed)
    bases = load_seed_designs()
    accepted: dict[tuple[float, ...], np.ndarray] = {}

    def add(position: np.ndarray) -> None:
        position = np.clip(np.asarray(position, dtype=float), LOWER, UPPER)
        key = tuple(np.round(position, 7))
        if key in accepted:
            return
        try:
            geometry = dimensionless_geometry(position)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return
        if geometry["radius_min_mm"] >= 0.905:
            accepted[key] = position

    for base in bases:
        add(base)
    attempts = 0
    while len(accepted) < target_size and attempts < 40 * target_size:
        base = bases[rng.integers(len(bases))].copy()
        base[:5] *= np.exp(rng.normal(0.0, 0.055, 5))
        base[5:7] += rng.normal(0.0, 0.025, 2)
        base[7] += rng.normal(0.0, 0.10)
        base[8] += rng.normal(0.0, 0.065)
        base[9] += rng.normal(0.0, 0.006)
        add(base)
        attempts += 1
    if len(accepted) < target_size // 2:
        raise RuntimeError("Could not construct a sufficiently large feasible pool")
    return np.vstack(list(accepted.values()))


def write_periodicity_audit(pool: np.ndarray) -> dict[str, object]:
    """Audit crown-to-crown ordinate and tangent closure over the full pool."""

    maximum_ordinate_gap = 0.0
    maximum_tangent_gap = 0.0
    for position in np.asarray(pool, dtype=float):
        x, y, (left, _, right) = nurbs_profile(
            position[:7],
            sample_size=801,
            wavelength_mm=position[7],
            amplitude_mm=position[8],
        )
        maximum_ordinate_gap = max(
            maximum_ordinate_gap, abs(float(y[left] - y[right]))
        )
        tangent_left = np.asarray(
            [x[left + 1] - x[left - 1], y[left + 1] - y[left - 1]],
            dtype=float,
        )
        tangent_right = np.asarray(
            [x[right + 1] - x[right - 1], y[right + 1] - y[right - 1]],
            dtype=float,
        )
        tangent_left /= np.linalg.norm(tangent_left)
        tangent_right /= np.linalg.norm(tangent_right)
        maximum_tangent_gap = max(
            maximum_tangent_gap,
            float(np.linalg.norm(tangent_left - tangent_right)),
        )
    audit: dict[str, object] = {
        "profiles_checked": int(len(pool)),
        "maximum_endpoint_ordinate_gap_mm": maximum_ordinate_gap,
        "maximum_unit_tangent_mismatch": maximum_tangent_gap,
        "sampling_intervals": 800,
        "period_parameter_interval": [0.25, 0.75],
    }
    (RESULTS / "periodicity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def farthest_point_sample(pool: np.ndarray, size: int) -> np.ndarray:
    descriptors = np.vstack([surrogate_features(row) for row in pool])
    scaled = StandardScaler().fit_transform(descriptors)
    selected = [int(np.argmax(np.linalg.norm(scaled - scaled.mean(axis=0), axis=1)))]
    distance = np.linalg.norm(scaled - scaled[selected[0]], axis=1)
    while len(selected) < min(size, len(pool)):
        nxt = int(np.argmax(distance))
        selected.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(scaled - scaled[nxt], axis=1))
    return pool[np.asarray(selected)]


def save_selected_designs(positions: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, position in enumerate(positions):
        geometry = dimensionless_geometry(position)
        rows.append(
            {
                "design_id": f"D{index:03d}",
                "split": "test" if index % 4 == 0 else "train",
                **_position_row(position),
                **geometry,
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(RESULTS / "fem_designs.csv", index=False)
    return data


def generate_fem_dataset(designs: pd.DataFrame, mesh: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_file = RESULTS / f"fem_paths_mesh{mesh}.csv"
    terminal_file = RESULTS / f"fem_terminal_mesh{mesh}.csv"
    expected_fingerprint = protocol_fingerprint(mesh)
    design_fingerprint = table_fingerprint(designs, ["design_id", "split", *VARIABLES])
    existing_path = pd.read_csv(path_file) if path_file.exists() else pd.DataFrame()
    existing_terminal = pd.read_csv(terminal_file) if terminal_file.exists() else pd.DataFrame()
    required_path_columns = {
        "raw_solver_success",
        "solver_accepted",
        "solver_normalized_projected_gradient",
        "solver_iterations",
        "solver_message",
        "lower_reaction_N",
        "lower_platen_surface_mm",
        "upper_platen_surface_mm",
        "solver_optimizer_method",
        "solver_primary_success",
        "solver_fallback_used",
        "solver_independent_kkt_confirmed",
        "solver_subdivision_count",
        "solver_diagnostics_json",
    }
    required_terminal_columns = {
        "path_success",
        "all_raw_solver_success",
        "maximum_normalized_projected_gradient",
        "maximum_reaction_imbalance_fraction",
        "maximum_absolute_reaction_imbalance_N",
        "maximum_reaction_imbalance_force_fraction",
        "initial_upper_contact_gap_mm",
        "initial_lower_contact_gap_mm",
        "maximum_subdivision_count",
        "fallback_solver_state_count",
        "independent_kkt_state_count",
        "solver_diagnostics_json",
    }
    cache_valid = (
        not existing_terminal.empty
        and required_terminal_columns.issubset(existing_terminal.columns)
        and required_path_columns.issubset(existing_path.columns)
        and "protocol_fingerprint" in existing_terminal
        and set(existing_terminal.protocol_fingerprint.astype(str)) == {expected_fingerprint}
        and "design_ledger_fingerprint" in existing_terminal
        and set(existing_terminal.design_ledger_fingerprint.astype(str)) == {design_fingerprint}
        and "protocol_fingerprint" in existing_path
        and set(existing_path.protocol_fingerprint.astype(str)) == {expected_fingerprint}
        and "design_ledger_fingerprint" in existing_path
        and set(existing_path.design_ledger_fingerprint.astype(str)) == {design_fingerprint}
    )
    if not cache_valid:
        existing_path = pd.DataFrame()
        existing_terminal = pd.DataFrame()
    completed = set(existing_terminal.get("design_id", pd.Series(dtype=str)).astype(str))
    path_records = existing_path.to_dict("records")
    terminal_records = existing_terminal.to_dict("records")
    protocol = CompressionProtocol(strains=STRAINS)
    for ordinal, row in designs.iterrows():
        design_id = str(row.design_id)
        if design_id in completed:
            continue
        position = _row_position(row)
        print(f"FEM training {ordinal + 1:03d}/{len(designs):03d}: {design_id}", flush=True)
        try:
            states, metrics = solve_compression_path(
                position, elements_per_wavelength=mesh, protocol=protocol
            )
        except Exception as error:  # preserve a reproducible failure record
            terminal_records.append(
                {
                    "design_id": design_id,
                    "split": row.split,
                    **_position_row(position),
                    "path_success": False,
                    "failure": f"{type(error).__name__}: {error}",
                    "protocol_fingerprint": expected_fingerprint,
                    "design_ledger_fingerprint": design_fingerprint,
                }
            )
        else:
            descriptor = surrogate_features(position)
            for state in states:
                path_records.append(
                    {
                        "design_id": design_id,
                        "split": row.split,
                        **_position_row(position),
                        **{f"feature_{i}": value for i, value in enumerate(descriptor)},
                        "strain": state["strain"],
                        "normalized_strain": float(state["strain"]) / 0.25,
                        "reaction_N": state["reaction_N"],
                        "lower_reaction_N": state["lower_reaction_N"],
                        "potential_energy_Nmm": state["potential_energy_Nmm"],
                        "stress_pnorm_MPa": state["stress_pnorm_MPa"],
                        "stress_q99_MPa": state["stress_q99_MPa"],
                        "maximum_stress_MPa": state["maximum_stress_MPa"],
                        "lower_platen_surface_mm": state["lower_platen_surface_mm"],
                        "upper_platen_surface_mm": state["upper_platen_surface_mm"],
                        "raw_solver_success": state["raw_solver_success"],
                        "solver_accepted": state["solver_accepted"],
                        "solver_gradient_norm": state["gradient_norm"],
                        "solver_normalized_projected_gradient": state[
                            "normalized_projected_gradient"
                        ],
                        "solver_iterations": state["iterations"],
                        "solver_message": state["message"],
                        "solver_optimizer_method": state["optimizer_method"],
                        "solver_primary_success": state["primary_solver_success"],
                        "solver_fallback_used": state["fallback_used"],
                        "solver_independent_kkt_confirmed": state[
                            "independent_kkt_confirmed"
                        ],
                        "solver_subdivision_count": state["subdivision_count"],
                        "solver_diagnostics_json": state["solver_diagnostics_json"],
                        "protocol_fingerprint": expected_fingerprint,
                        "design_ledger_fingerprint": design_fingerprint,
                    }
                )
            terminal_records.append(
                {
                    "design_id": design_id,
                    "split": row.split,
                    **_position_row(position),
                    **{f"feature_{i}": value for i, value in enumerate(descriptor)},
                    "design_ledger_fingerprint": design_fingerprint,
                    **metrics,
                }
            )
        pd.DataFrame(path_records).to_csv(path_file, index=False)
        pd.DataFrame(terminal_records).to_csv(terminal_file, index=False)
    return pd.read_csv(path_file), pd.read_csv(terminal_file)


def _error_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    nonzero = np.abs(truth) > max(1.0e-10, 0.01 * np.max(np.abs(truth)))
    return {
        "r2": float(r2_score(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "relative_mae_nonzero": float(
            np.mean(np.abs(prediction[nonzero] - truth[nonzero]) / np.abs(truth[nonzero]))
        ),
    }


def fit_surrogates(paths: pd.DataFrame, terminal: pd.DataFrame):
    feature_columns = [f"feature_{i}" for i in range(FEATURE_COUNT)]
    successful_ids = set(
        terminal.loc[terminal.path_success.astype(bool), "design_id"].astype(str)
    )
    paths = paths[paths.design_id.astype(str).isin(successful_ids)].copy()
    train = paths[paths.split == "train"]
    test = paths[paths.split == "test"]
    arc_by_design = terminal.set_index("design_id").arc_length_mm.to_dict()

    def work_scale(frame: pd.DataFrame) -> np.ndarray:
        arc = frame.design_id.map(arc_by_design).to_numpy(dtype=float)
        return 60.0 * 25.0 * frame.thickness_mm.to_numpy(dtype=float) * arc

    train_scale = work_scale(train)
    test_scale = work_scale(test)
    train_dimensionless_energy = train.potential_energy_Nmm.to_numpy(dtype=float) / train_scale
    train_dimensionless_force = (
        train.reaction_N.to_numpy(dtype=float)
        * train.height_mm.to_numpy(dtype=float)
        / train_scale
    )
    x_train = np.column_stack(
        (train[feature_columns].to_numpy(dtype=float), train.normalized_strain)
    )
    x_test = np.column_stack(
        (test[feature_columns].to_numpy(dtype=float), test.normalized_strain)
    )
    networks = []
    diagnostics = []
    for seed in SEEDS:
        print(f"Training energy-consistent neural potential, seed {seed}", flush=True)
        network, diagnostic = fit_energy_network(
            x_train,
            train_dimensionless_energy,
            train_dimensionless_force,
            np.ones(len(train), dtype=float),
            train.design_id.to_numpy(),
            hidden_size=14,
            seed=seed,
            max_iterations=900,
        )
        network.save(MODELS / f"energy_surrogate_seed{seed}.npz")
        networks.append(network)
        diagnostics.append(
            {
                "seed": seed,
                "optimizer_success": diagnostic["success"],
                "optimizer_message": diagnostic["message"],
                "iterations": diagnostic["iterations"],
                "loss": diagnostic["final_loss"],
            }
        )
    potential_number_predictions = np.vstack(
        [network.energy_Nmm(x_test) for network in networks]
    )
    dimensionless_force_predictions = np.vstack(
        [
            network.reaction_N(x_test, np.ones(len(test), dtype=float))
            for network in networks
        ]
    )
    energy_predictions = potential_number_predictions * test_scale[None, :]
    reaction_predictions = (
        dimensionless_force_predictions
        * test_scale[None, :]
        / test.height_mm.to_numpy(dtype=float)[None, :]
    )
    test_predictions = test[
        ["design_id", "strain", "potential_energy_Nmm", "reaction_N"]
    ].copy()
    test_predictions["neural_energy_Nmm"] = energy_predictions.mean(axis=0)
    test_predictions["neural_energy_std_Nmm"] = energy_predictions.std(axis=0)
    test_predictions["neural_reaction_N"] = reaction_predictions.mean(axis=0)
    test_predictions["neural_reaction_std_N"] = reaction_predictions.std(axis=0)
    test_predictions["fem_potential_number"] = (
        test.potential_energy_Nmm.to_numpy(dtype=float) / test_scale
    )
    test_predictions["neural_potential_number"] = potential_number_predictions.mean(axis=0)
    test_predictions["neural_potential_number_std"] = potential_number_predictions.std(axis=0)
    test_predictions.to_csv(RESULTS / "energy_surrogate_test_predictions.csv", index=False)

    terminal_success = terminal[terminal.path_success.astype(bool)].copy()
    stress_train = terminal_success[terminal_success.split == "train"]
    stress_test = terminal_success[terminal_success.split == "test"]
    stress_model = ExtraTreesRegressor(
        n_estimators=160,
        min_samples_leaf=2,
        max_features=0.85,
        random_state=20260821,
        # Single-row optimizer queries are faster without parallel dispatch.
        n_jobs=1,
    )
    stress_x_train = stress_train[feature_columns].to_numpy(dtype=float)
    stress_x_test = stress_test[feature_columns].to_numpy(dtype=float)
    stress_model.fit(stress_x_train, stress_train.stress_pnorm_utilization)
    stress_tree_prediction = np.vstack(
        [tree.predict(stress_x_test) for tree in stress_model.estimators_]
    )
    stress_prediction = stress_tree_prediction.mean(axis=0)
    stress_std = stress_tree_prediction.std(axis=0)
    stress_calibration_margin = float(
        max(
            0.0,
            np.quantile(
                stress_test.stress_pnorm_utilization.to_numpy(dtype=float)
                - stress_prediction,
                0.90,
                method="higher",
            ),
        )
    )
    stress_upper = stress_prediction + stress_calibration_margin
    stress_validation = stress_test[["design_id", "stress_pnorm_utilization"]].copy()
    stress_validation["predicted_stress_utilization"] = stress_prediction
    stress_validation["predicted_stress_std"] = stress_std
    stress_validation["calibrated_upper_stress_utilization"] = stress_upper
    stress_validation["upper_bound_covers_truth"] = (
        stress_upper >= stress_test.stress_pnorm_utilization.to_numpy(dtype=float)
    )
    stress_validation.to_csv(RESULTS / "stress_test_predictions.csv", index=False)

    scaler = StandardScaler().fit(stress_x_train)
    scaled_train = scaler.transform(stress_x_train)
    neighbors = NearestNeighbors(n_neighbors=2).fit(scaled_train)
    leave_one_out_distance = neighbors.kneighbors(scaled_train)[0][:, 1]
    trust_limit = float(1.5 * np.quantile(leave_one_out_distance, 0.95))
    dump(stress_model, MODELS / "stress_ensemble.joblib")
    dump(scaler, MODELS / "feature_scaler.joblib")
    dump(neighbors, MODELS / "trust_neighbors.joblib")

    metrics = {
        "dataset": {
            "successful_designs": int(len(terminal_success)),
            "training_designs": int(len(stress_train)),
            "test_designs": int(len(stress_test)),
            "path_rows": int(len(paths)),
        },
        "neural_energy_test": _error_metrics(
            test.potential_energy_Nmm, energy_predictions.mean(axis=0)
        ),
        "neural_reaction_test": _error_metrics(
            test.reaction_N, reaction_predictions.mean(axis=0)
        ),
        "neural_potential_number_test": _error_metrics(
            test.potential_energy_Nmm.to_numpy(dtype=float) / test_scale,
            potential_number_predictions.mean(axis=0),
        ),
        "stress_model_test": _error_metrics(
            stress_test.stress_pnorm_utilization, stress_prediction
        ),
        "stress_upper_bound": {
            "calibration_split": "held-out design-level test split",
            "target_empirical_coverage": 0.90,
            "additive_calibration_margin": stress_calibration_margin,
            "empirical_coverage": float(np.mean(stress_upper >= stress_test.stress_pnorm_utilization)),
            "maximum_underprediction_after_calibration": float(
                np.max(stress_test.stress_pnorm_utilization.to_numpy(dtype=float) - stress_upper)
            ),
        },
        "trust_distance_limit": trust_limit,
        "training": diagnostics,
    }
    (RESULTS / "surrogate_validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return (
        networks,
        stress_model,
        scaler,
        neighbors,
        trust_limit,
        stress_calibration_margin,
        metrics,
    )


def build_predictor(networks, stress_model, scaler, neighbors, stress_calibration_margin):
    cache: dict[tuple[float, ...], dict[str, float]] = {}

    def predict(position: np.ndarray) -> dict[str, float]:
        position = np.asarray(position, dtype=float)
        key = tuple(np.round(position, 9))
        if key in cache:
            return cache[key]
        geometry = dimensionless_geometry(position)
        descriptor = surrogate_features(position)
        scaled = scaler.transform(descriptor.reshape(1, -1))
        trust_distance = float(neighbors.kneighbors(scaled, n_neighbors=1)[0][0, 0])
        x = np.r_[descriptor, STRAINS[-1] / 0.25].reshape(1, -1)
        energy = np.asarray([network.energy_Nmm(x)[0] for network in networks])
        descriptor_2d = descriptor.reshape(1, -1)
        stress_mean = float(stress_model.predict(descriptor_2d)[0])
        result = {
            **geometry,
            "trust_distance": trust_distance,
            "predicted_potential_number": float(energy.mean()),
            "predicted_potential_std": float(energy.std()),
            "predicted_stress_utilization": stress_mean,
            "predicted_stress_upper": float(stress_mean + stress_calibration_margin),
        }
        cache[key] = result
        return result

    return predict


def case_objectives(case: int, prediction: dict[str, float], thresholds: dict[str, float]):
    inverse_work = 1.0 / max(prediction["predicted_potential_number"], 1.0e-12)
    # The constant held-out residual quantile is reported as a descriptive
    # calibration diagnostic.  It cannot change dominance or PSO motion, so
    # the actual mean stress prediction is the screening objective.
    screening_stress = prediction["predicted_stress_utilization"]
    violation = max(0.0, 0.9 - prediction["radius_min_mm"]) / 0.9
    violation += max(0.0, prediction["trust_distance"] - thresholds["trust_limit"]) / max(
        thresholds["trust_limit"], 1.0e-12
    )
    if case == 37:
        values = np.asarray([prediction["board_material_fraction"], inverse_work])
    elif case == 38:
        values = np.asarray([prediction["board_material_fraction"], screening_stress])
        violation += max(
            0.0,
            thresholds["minimum_potential"] - prediction["predicted_potential_number"],
        ) / thresholds["minimum_potential"]
    elif case == 39:
        values = np.asarray([inverse_work, screening_stress])
        violation += max(
            0.0,
            prediction["board_material_fraction"] - thresholds["maximum_material_fraction"],
        ) / thresholds["maximum_material_fraction"]
    else:
        raise ValueError(case)
    return values, bool(violation <= 1.0e-12), float(violation)


def initial_population(case: int, pool: np.ndarray, predict, thresholds, size: int, seed: int):
    rng = np.random.default_rng(seed)
    records = []
    for position in pool:
        values, feasible, violation = case_objectives(case, predict(position), thresholds)
        if feasible:
            records.append((position, values))
    if len(records) < size:
        raise RuntimeError(f"Only {len(records)} feasible seeds for C{case:03d}")
    values = np.vstack([row[1] for row in records])
    front = np.flatnonzero(nondominated(values))
    chosen = list(front)
    if len(chosen) > size:
        order = np.argsort(values[chosen, 0])
        chosen = list(np.asarray(chosen)[order[np.linspace(0, len(order) - 1, size).astype(int)]])
    remaining = np.setdiff1d(np.arange(len(records)), np.asarray(chosen), assume_unique=False)
    rng.shuffle(remaining)
    chosen.extend(remaining[: size - len(chosen)].tolist())
    return np.vstack([records[index][0] for index in chosen])


def optimize_surrogates(pool, predict, thresholds, population_size: int, generations: int):
    all_records = []
    history_records = []
    for case in (37, 38, 39):
        pool_values = []
        for position in pool:
            values, feasible, _ = case_objectives(case, predict(position), thresholds)
            if feasible:
                pool_values.append(values)
        if len(pool_values) < population_size:
            raise RuntimeError(f"Insufficient feasible reference pool for C{case:03d}")
        reference = 1.10 * np.max(np.vstack(pool_values), axis=0)
        for seed in SEEDS:
            print(f"Surrogate multi-objective search C{case:03d}, seed {seed}", flush=True)
            initial = initial_population(
                case, pool, predict, thresholds, population_size, seed
            )
            result = run_mo_etpso(
                lambda x, c=case: case_objectives(c, predict(x), thresholds),
                LOWER,
                UPPER,
                population_size=population_size,
                generations=generations,
                inertia=0.7,
                seed=seed,
                reference=reference,
                initial_positions=initial,
                initial_velocity_scale=0.025,
            )
            for generation, hypervolume in enumerate(result.history_hypervolume, start=1):
                history_records.append(
                    {
                        "case": f"C{case:03d}",
                        "seed": seed,
                        "generation": generation,
                        "evaluations": generation * population_size,
                        "hypervolume": float(hypervolume),
                        "normalized_hypervolume": float(
                            hypervolume / max(float(np.prod(reference)), 1.0e-12)
                        ),
                        "reference_objective_1": float(reference[0]),
                        "reference_objective_2": float(reference[1]),
                    }
                )
            for member, (position, objectives, feasible, violation) in enumerate(
                zip(result.positions, result.objectives, result.feasible, result.constraint_violation)
            ):
                prediction = predict(position)
                all_records.append(
                    {
                        "case": f"C{case:03d}",
                        "seed": seed,
                        "member": member,
                        **_position_row(position),
                        "surrogate_objective_1": objectives[0],
                        "surrogate_objective_2": objectives[1],
                        "surrogate_feasible": feasible,
                        "surrogate_violation": violation,
                        **prediction,
                    }
                )
    all_members = pd.DataFrame(all_records)
    all_members.to_csv(RESULTS / "surrogate_terminal_members.csv", index=False)
    history = pd.DataFrame(history_records)
    history.to_csv(RESULTS / "optimization_convergence.csv", index=False)
    front_records = []
    for case, group in all_members[all_members.surrogate_feasible].groupby("case"):
        group = group.drop_duplicates(subset=VARIABLES).copy()
        mask = nondominated(group[["surrogate_objective_1", "surrogate_objective_2"]].to_numpy())
        front = group[mask].sort_values("surrogate_objective_1")
        front_records.extend(front.to_dict("records"))
    front = pd.DataFrame(front_records)
    front.to_csv(RESULTS / "surrogate_pareto_candidates.csv", index=False)
    return all_members, front, history


def build_verification_pool(
    all_members: pd.DataFrame,
    training_designs: pd.DataFrame,
    predict,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Merge every terminal particle with all direct-FEM training geometries.

    Each of the 80 training geometries is admitted to every problem for which
    its direct response can be classified.  This prevents the neural search
    trajectory from defining the universe over which final nondomination is
    assessed.
    """

    optimizer = all_members.copy()
    optimizer["candidate_source"] = "optimizer_terminal"
    optimizer["source_design_id"] = ""
    additions = []
    for _, row in training_designs.iterrows():
        position = _row_position(row)
        prediction = predict(position)
        for case in (37, 38, 39):
            objectives, feasible, violation = case_objectives(
                case, prediction, thresholds
            )
            additions.append(
                {
                    "case": f"C{case:03d}",
                    "seed": -1,
                    "member": str(row.design_id),
                    **_position_row(position),
                    "surrogate_objective_1": float(objectives[0]),
                    "surrogate_objective_2": float(objectives[1]),
                    "surrogate_feasible": bool(feasible),
                    "surrogate_violation": float(violation),
                    **prediction,
                    "candidate_source": "fem_training_design",
                    "source_design_id": str(row.design_id),
                }
            )
    combined = pd.concat([optimizer, pd.DataFrame(additions)], ignore_index=True)
    combined["_source_priority"] = (
        combined.candidate_source == "optimizer_terminal"
    ).astype(int)
    combined = (
        combined.sort_values("_source_priority", ascending=False)
        .drop_duplicates(subset=["case", *VARIABLES])
        .drop(columns="_source_priority")
        .reset_index(drop=True)
    )
    combined.to_csv(RESULTS / "direct_verification_pool.csv", index=False)
    return combined


def direct_objectives(case: int, metrics: dict[str, float], thresholds: dict[str, float]):
    inverse_potential = 1.0 / max(metrics["stored_potential_number"], 1.0e-12)
    violation = 0.0 if metrics["path_success"] else 1.0
    violation += max(0.0, 0.9 - metrics["radius_min_mm"]) / 0.9
    trust_distance = float(metrics.get("trust_distance", np.nan))
    if not np.isfinite(trust_distance):
        raise ValueError("Direct classification requires a finite trust distance")
    violation += max(0.0, trust_distance - thresholds["trust_limit"]) / max(
        thresholds["trust_limit"], 1.0e-12
    )
    if case == 37:
        values = [metrics["board_material_fraction"], inverse_potential]
    elif case == 38:
        values = [metrics["board_material_fraction"], metrics["stress_pnorm_utilization"]]
        violation += max(0.0, thresholds["minimum_potential"] - metrics["stored_potential_number"]) / thresholds[
            "minimum_potential"
        ]
    elif case == 39:
        values = [inverse_potential, metrics["stress_pnorm_utilization"]]
        violation += max(
            0.0,
            metrics["board_material_fraction"] - thresholds["maximum_material_fraction"],
        ) / thresholds["maximum_material_fraction"]
    else:
        raise ValueError(case)
    return values, violation


def _solve_direct_geometry(
    row: dict[str, object],
    mesh: int,
    geometry_ledger_fingerprint: str,
) -> dict[str, object]:
    """Worker-safe direct FEM evaluation for one unique geometry."""

    position = np.asarray([row[name] for name in VARIABLES], dtype=float)
    base = {
        **_position_row(position),
        "geometry_ledger_fingerprint": geometry_ledger_fingerprint,
    }
    try:
        _, metrics = solve_compression_path(
            position,
            elements_per_wavelength=mesh,
            protocol=CompressionProtocol(strains=STRAINS),
        )
    except Exception as error:
        return {
            **base,
            "path_success": False,
            "failure": f"{type(error).__name__}: {error}",
            "protocol_fingerprint": protocol_fingerprint(mesh),
        }
    return {**base, **metrics}


def verify_candidates(candidates: pd.DataFrame, mesh: int, thresholds: dict[str, float]):
    candidates = candidates.drop_duplicates(subset=["case", *VARIABLES]).reset_index(drop=True)
    fingerprint_columns = ["case", *VARIABLES, "trust_distance"]
    candidate_fingerprint = table_fingerprint(candidates, fingerprint_columns)
    classification_fingerprint = analysis_fingerprint(
        candidates,
        fingerprint_columns,
        mesh=mesh,
        thresholds=thresholds,
    )
    output = RESULTS / f"fem_verified_terminal_mesh{mesh}.csv"
    mechanics_output = RESULTS / f"fem_unique_geometries_mesh{mesh}.csv"
    expected_fingerprint = protocol_fingerprint(mesh)
    geometries = candidates.drop_duplicates(subset=VARIABLES).reset_index(drop=True)
    geometry_fingerprint = table_fingerprint(geometries, VARIABLES)
    existing = pd.read_csv(mechanics_output) if mechanics_output.exists() else pd.DataFrame()
    required_columns = {
        "path_success",
        "all_raw_solver_success",
        "maximum_normalized_projected_gradient",
        "maximum_reaction_imbalance_fraction",
        "maximum_absolute_reaction_imbalance_N",
        "maximum_reaction_imbalance_force_fraction",
        "initial_upper_contact_gap_mm",
        "initial_lower_contact_gap_mm",
        "maximum_subdivision_count",
        "fallback_solver_state_count",
        "independent_kkt_state_count",
        "solver_diagnostics_json",
    }
    cache_valid = (
        not existing.empty
        and required_columns.issubset(existing.columns)
        and "protocol_fingerprint" in existing
        and set(existing.protocol_fingerprint.astype(str)) == {expected_fingerprint}
        and "geometry_ledger_fingerprint" in existing
        and set(existing.geometry_ledger_fingerprint.astype(str)) == {geometry_fingerprint}
    )
    if not cache_valid:
        existing = pd.DataFrame()
    records = existing.to_dict("records")
    completed = {
        tuple(np.round([row[name] for name in VARIABLES], 7))
        for row in records
    }
    pending = []
    for _, row in geometries.iterrows():
        position = _row_position(row)
        key = tuple(np.round(position, 7))
        if key in completed:
            continue
        pending.append(row.to_dict())
    if pending:
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(
                    _solve_direct_geometry,
                    row,
                    mesh,
                    geometry_fingerprint,
                ): row
                for row in pending
            }
            for ordinal, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                print(
                    f"Direct FEM mesh {mesh}: {ordinal:03d}/{len(pending):03d} unique geometries",
                    flush=True,
                )
                if ordinal % 4 == 0 or ordinal == len(pending):
                    pd.DataFrame(records).to_csv(mechanics_output, index=False)
    mechanics = pd.read_csv(mechanics_output)

    metadata_columns = [
        "case",
        "seed",
        "member",
        *VARIABLES,
        "surrogate_objective_1",
        "surrogate_objective_2",
        "surrogate_feasible",
        "surrogate_violation",
        "trust_distance",
        "candidate_source",
        "source_design_id",
    ]
    candidate_metadata = add_geometry_key(candidates[metadata_columns])
    keyed_mechanics = add_geometry_key(mechanics)
    if keyed_mechanics.geometry_key.duplicated().any():
        raise RuntimeError("Rounded geometry keys are not unique in the mechanics ledger")
    verified = candidate_metadata.merge(
        keyed_mechanics.drop(columns=VARIABLES),
        on="geometry_key",
        how="left",
        validate="many_to_one",
    )
    missing_mechanics = verified.protocol_fingerprint.isna()
    if missing_mechanics.any():
        raise RuntimeError(
            f"Mechanical records are missing for {int(missing_mechanics.sum())} classifications"
        )
    verified["path_success"] = verified.path_success.fillna(False).astype(bool)
    verified["candidate_ledger_fingerprint"] = candidate_fingerprint
    verified["analysis_fingerprint"] = classification_fingerprint
    verified["fem_objective_1"] = float("inf")
    verified["fem_objective_2"] = float("inf")
    verified["fem_constraint_violation"] = 1.0
    for index, row in verified.iterrows():
        if not bool(row.get("path_success", False)):
            continue
        objectives, violation = direct_objectives(
            int(str(row.case)[1:]), row.to_dict(), thresholds
        )
        verified.loc[index, "fem_objective_1"] = objectives[0]
        verified.loc[index, "fem_objective_2"] = objectives[1]
        verified.loc[index, "fem_constraint_violation"] = violation
    verified["fem_pareto"] = False
    for case, index in verified.groupby("case").groups.items():
        group = verified.loc[index]
        valid = group.path_success.astype(bool) & np.isclose(group.fem_constraint_violation, 0.0)
        valid_index = group.index[valid]
        if len(valid_index):
            mask = nondominated(
                verified.loc[valid_index, ["fem_objective_1", "fem_objective_2"]].to_numpy()
            )
            verified.loc[valid_index[mask], "fem_pareto"] = True
    verified.to_csv(output, index=False)
    verified[verified.fem_pareto].to_csv(
        RESULTS / f"fem_rebuilt_pareto_fronts_mesh{mesh}.csv", index=False
    )
    return verified


def compare_mesh_fronts(coarse: pd.DataFrame, final: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quantify response and Pareto-membership changes from 24 to 32 elements."""

    def keyed(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["design_key"] = frame.apply(
            lambda row: str(row.case)
            + ":"
            + ":".join(f"{float(row[name]):.9g}" for name in VARIABLES),
            axis=1,
        )
        return frame

    fields = [
        "case",
        "design_key",
        "path_success",
        "fem_constraint_violation",
        "fem_pareto",
        "target_reaction_N",
        "stored_potential_number",
        "stress_pnorm_utilization",
        "stress_q99_utilization",
        "stress_max_utilization",
    ]
    comparison = keyed(coarse)[fields].merge(
        keyed(final)[fields], on=["case", "design_key"], suffixes=("_mesh24", "_mesh32")
    )
    response_fields = [
        "target_reaction_N",
        "stored_potential_number",
        "stress_pnorm_utilization",
        "stress_q99_utilization",
        "stress_max_utilization",
    ]
    for field in response_fields:
        comparison[f"{field}_relative_change"] = np.abs(
            comparison[f"{field}_mesh32"] - comparison[f"{field}_mesh24"]
        ) / np.maximum(np.abs(comparison[f"{field}_mesh32"]), 1.0e-12)
    comparison["membership_stable"] = (
        comparison.fem_pareto_mesh24.astype(bool)
        == comparison.fem_pareto_mesh32.astype(bool)
    )
    comparison.to_csv(RESULTS / "front_membership_stability.csv", index=False)

    summary: dict[str, dict[str, float | int]] = {}
    for case, group in comparison.groupby("case"):
        front24 = set(group.loc[group.fem_pareto_mesh24.astype(bool), "design_key"])
        front32 = set(group.loc[group.fem_pareto_mesh32.astype(bool), "design_key"])
        union = front24 | front32
        summary[str(case)] = {
            "mesh24_front_members": len(front24),
            "mesh32_front_members": len(front32),
            "retained_members": len(front24 & front32),
            "jaccard_membership": float(len(front24 & front32) / max(len(union), 1)),
            "all_candidate_membership_stability_fraction": float(group.membership_stable.mean()),
            **{
                f"maximum_{field}_relative_change": float(group[f"{field}_relative_change"].max())
                for field in response_fields
            },
        }
    (RESULTS / "front_membership_stability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return comparison, summary


def analyze_c038_fine_meshes(
    mesh40: pd.DataFrame,
    mesh48: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Build an uncertainty-aware C038 front from 40/48-element solutions.

    Material fraction is mesh independent.  The absolute stress tolerance is
    set conservatively to the maximum paired 40--48 stress change among paths
    accepted at both meshes.  A design is called robustly dominated only when
    another point improves at least one objective by more than this numerical
    tolerance without being worse beyond it.  Publication membership requires
    tolerance-front membership at both resolutions.
    """

    def keyed(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["design_key"] = frame.apply(
            lambda row: str(row.case)
            + ":"
            + ":".join(f"{float(row[name]):.9g}" for name in VARIABLES),
            axis=1,
        )
        return frame

    coarse = keyed(mesh40)
    fine = keyed(mesh48)
    comparison = coarse[
        [
            "design_key",
            "path_success",
            "fem_constraint_violation",
            "board_material_fraction",
            "stored_potential_number",
            "stress_pnorm_utilization",
            "fem_pareto",
        ]
    ].merge(
        fine[
            [
                "design_key",
                "path_success",
                "fem_constraint_violation",
                "board_material_fraction",
                "stored_potential_number",
                "stress_pnorm_utilization",
                "fem_pareto",
            ]
        ],
        on="design_key",
        suffixes=("_mesh40", "_mesh48"),
    )
    paired = comparison[
        comparison.path_success_mesh40.astype(bool)
        & comparison.path_success_mesh48.astype(bool)
    ].copy()
    stress_tolerance = float(
        np.max(
            np.abs(
                paired.stress_pnorm_utilization_mesh48
                - paired.stress_pnorm_utilization_mesh40
            )
        )
    )
    fine_analysis_payload = {
        "mesh40_classification_fingerprints": sorted(
            set(mesh40.analysis_fingerprint.astype(str))
        ),
        "mesh48_classification_fingerprints": sorted(
            set(mesh48.analysis_fingerprint.astype(str))
        ),
        "stress_absolute_tolerance": stress_tolerance,
        "tolerance_definition": "maximum paired absolute 40-to-48 Omega_8 change",
        "dominance_rule": (
            "all objectives no worse than candidate plus absolute tolerance and "
            "at least one objective better than candidate minus absolute tolerance"
        ),
        "publication_membership": "intersection of mesh40 and mesh48 tolerance fronts",
    }
    fine_analysis_fingerprint = hashlib.sha256(
        json.dumps(
            fine_analysis_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    tolerance = np.asarray([0.0, stress_tolerance])
    tolerance_keys: dict[int, set[str]] = {}
    for mesh, frame in ((40, coarse), (48, fine)):
        valid = frame[
            frame.path_success.astype(bool)
            & np.isclose(frame.fem_constraint_violation, 0.0)
        ].copy()
        values = valid[
            ["board_material_fraction", "stress_pnorm_utilization"]
        ].to_numpy(dtype=float)
        mask = tolerance_nondominated(values, tolerance)
        keys = set(valid.loc[mask, "design_key"])
        tolerance_keys[mesh] = keys
        frame["tolerance_pareto"] = frame.design_key.isin(keys)
        frame["fine_analysis_fingerprint"] = fine_analysis_fingerprint
        frame.to_csv(RESULTS / f"fem_verified_terminal_mesh{mesh}.csv", index=False)
        frame[frame.tolerance_pareto].to_csv(
            RESULTS / f"fem_tolerance_pareto_fronts_mesh{mesh}.csv", index=False
        )
    stable_keys = tolerance_keys[40] & tolerance_keys[48]
    fine["tolerance_pareto"] = fine.design_key.isin(tolerance_keys[48])
    fine["fine_mesh_stable_pareto"] = fine.design_key.isin(stable_keys)
    fine["fine_analysis_fingerprint"] = fine_analysis_fingerprint
    fine.to_csv(RESULTS / "fem_verified_terminal_mesh48.csv", index=False)
    comparison["tolerance_pareto_mesh40"] = comparison.design_key.isin(
        tolerance_keys[40]
    )
    comparison["tolerance_pareto_mesh48"] = comparison.design_key.isin(
        tolerance_keys[48]
    )
    comparison["tolerance_membership_stable"] = (
        comparison.tolerance_pareto_mesh40
        == comparison.tolerance_pareto_mesh48
    )
    comparison["fine_analysis_fingerprint"] = fine_analysis_fingerprint
    comparison["stress_absolute_change"] = np.abs(
        comparison.stress_pnorm_utilization_mesh48
        - comparison.stress_pnorm_utilization_mesh40
    )
    comparison["stress_relative_change"] = comparison.stress_absolute_change / np.maximum(
        np.abs(comparison.stress_pnorm_utilization_mesh48), 1.0e-12
    )
    comparison.to_csv(RESULTS / "c038_fine_mesh_stability.csv", index=False)
    union = tolerance_keys[40] | tolerance_keys[48]
    stable = fine[fine.fine_mesh_stable_pareto]
    summary: dict[str, float | int] = {
        "mesh40_exact_front_members": int(coarse.fem_pareto.astype(bool).sum()),
        "mesh48_exact_front_members": int(fine.fem_pareto.astype(bool).sum()),
        "stress_absolute_tolerance": stress_tolerance,
        "fine_analysis_fingerprint": fine_analysis_fingerprint,
        "tolerance_definition": fine_analysis_payload["tolerance_definition"],
        "mesh40_tolerance_front_members": len(tolerance_keys[40]),
        "mesh48_tolerance_front_members": len(tolerance_keys[48]),
        "retained_tolerance_members": len(stable_keys),
        "tolerance_front_jaccard": float(len(stable_keys) / max(len(union), 1)),
        "all_candidate_tolerance_membership_stability_fraction": float(
            comparison.tolerance_membership_stable.mean()
        ),
        "maximum_40_to_48_stress_relative_change": float(
            paired.stress_relative_change.max()
            if "stress_relative_change" in paired
            else comparison.loc[paired.index, "stress_relative_change"].max()
        ),
        "stable_material_fraction_min": float(stable.board_material_fraction.min()),
        "stable_material_fraction_max": float(stable.board_material_fraction.max()),
        "stable_stress_utilization_min": float(stable.stress_pnorm_utilization.min()),
        "stable_stress_utilization_max": float(stable.stress_pnorm_utilization.max()),
    }
    (RESULTS / "c038_fine_mesh_stability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return fine, comparison, summary


def threshold_sensitivity(
    training_terminal: pd.DataFrame,
    verified: pd.DataFrame,
    c038_stress_tolerance: float,
) -> pd.DataFrame:
    """Rebuild direct fronts under nearby training-defined threshold quantiles."""

    records = []
    valid = verified[
        verified.path_success.astype(bool) & (verified.radius_min_mm >= 0.9)
    ].copy()
    for quantile in (0.30, 0.40, 0.50):
        threshold = float(np.quantile(training_terminal.stored_potential_number, quantile))
        group = valid[(valid.case == "C038") & (valid.stored_potential_number >= threshold)].copy()
        values = group[["board_material_fraction", "stress_pnorm_utilization"]].to_numpy()
        front = (
            group[
                tolerance_nondominated(
                    values, np.asarray([0.0, c038_stress_tolerance])
                )
            ]
            if len(group)
            else group
        )
        records.append(
            {
                "case": "C038",
                "threshold_kind": "minimum stored-potential number",
                "training_quantile": quantile,
                "threshold_value": threshold,
                "feasible_candidates": len(group),
                "front_members": len(front),
                "dominance_rule": "40/48-derived stress-tolerance dominance",
                "objective_1_min": float(front.board_material_fraction.min()) if len(front) else np.nan,
                "objective_1_max": float(front.board_material_fraction.max()) if len(front) else np.nan,
                "objective_2_min": float(front.stress_pnorm_utilization.min()) if len(front) else np.nan,
                "objective_2_max": float(front.stress_pnorm_utilization.max()) if len(front) else np.nan,
            }
        )
    for quantile in (0.55, 0.65, 0.75):
        threshold = float(np.quantile(training_terminal.board_material_fraction, quantile))
        group = valid[(valid.case == "C039") & (valid.board_material_fraction <= threshold)].copy()
        values = np.column_stack(
            (
                1.0 / np.maximum(group.stored_potential_number.to_numpy(), 1.0e-12),
                group.stress_pnorm_utilization.to_numpy(),
            )
        )
        front = group[nondominated(values)] if len(group) else group
        records.append(
            {
                "case": "C039",
                "threshold_kind": "maximum board material fraction",
                "training_quantile": quantile,
                "threshold_value": threshold,
                "feasible_candidates": len(group),
                "front_members": len(front),
                "dominance_rule": "strict dominance",
                "objective_1_min": float((1.0 / front.stored_potential_number).min()) if len(front) else np.nan,
                "objective_1_max": float((1.0 / front.stored_potential_number).max()) if len(front) else np.nan,
                "objective_2_min": float(front.stress_pnorm_utilization.min()) if len(front) else np.nan,
                "objective_2_max": float(front.stress_pnorm_utilization.max()) if len(front) else np.nan,
            }
        )
    data = pd.DataFrame(records)
    data.to_csv(RESULTS / "threshold_sensitivity.csv", index=False)
    return data


def summarize_optimizer_convergence(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, seed), group in history.groupby(["case", "seed"]):
        group = group.sort_values("generation")
        final = float(group.normalized_hypervolume.iloc[-1])
        checkpoint = float(
            group.loc[group.generation <= 42, "normalized_hypervolume"].iloc[-1]
        )
        rows.append(
            {
                "case": case,
                "seed": seed,
                "generation_42_normalized_hypervolume": checkpoint,
                "final_normalized_hypervolume": final,
                "relative_gain_42_to_final": (final - checkpoint) / max(abs(final), 1.0e-12),
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(RESULTS / "optimization_convergence_summary.csv", index=False)
    return data


def select_representatives(front: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for case, group in front.groupby("case"):
        group = group.sort_values("fem_objective_1").reset_index(drop=True)
        if len(group) <= 3:
            chosen = group
        else:
            normalized = group[["fem_objective_1", "fem_objective_2"]].to_numpy(dtype=float)
            span = np.maximum(np.ptp(normalized, axis=0), 1.0e-12)
            normalized = (normalized - normalized.min(axis=0)) / span
            compromise = int(np.argmin(np.linalg.norm(normalized, axis=1)))
            chosen = group.iloc[np.unique([0, compromise, len(group) - 1])]
        selected.extend(chosen.to_dict("records"))
    selected = pd.DataFrame(selected).drop_duplicates(subset=VARIABLES).reset_index(drop=True)
    selected.insert(0, "selection_id", [f"S{i + 1}" for i in range(len(selected))])
    selected.to_csv(RESULTS / "selected_optimized_geometries.csv", index=False)
    return selected


def run_mesh_study(selected: pd.DataFrame):
    records = []
    refined: dict[str, tuple[np.ndarray, list[dict[str, object]], dict[str, float]]] = {}
    protocol = CompressionProtocol(strains=STRAINS)
    for _, row in selected.iterrows():
        position = _row_position(row)
        for mesh in (8, 12, 16, 24, 32, 40, 48):
            print(f"Mesh study {row.selection_id}: {mesh} elements", flush=True)
            states, metrics = solve_compression_path(
                position, elements_per_wavelength=mesh, protocol=protocol
            )
            records.append(
                {
                    "selection_id": row.selection_id,
                    "source_case": row.case,
                    "mesh": mesh,
                    **_position_row(position),
                    **metrics,
                }
            )
            if mesh == 48:
                nodes, _ = paper_a_profile_nodes(position, mesh)
                refined[str(row.selection_id)] = (nodes, states, metrics)
    data = pd.DataFrame(records)
    data.to_csv(RESULTS / "optimized_geometry_mesh_study.csv", index=False)
    return data, refined


def plot_validation(metrics: dict[str, object]):
    apply_latex_style(7.5)
    p = pd.read_csv(RESULTS / "energy_surrogate_test_predictions.csv")
    s = pd.read_csv(RESULTS / "stress_test_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    panels = [
        (p.potential_energy_Nmm, p.neural_energy_Nmm, r"FEM $U$ [N mm]", r"neural $U$ [N mm]"),
        (p.reaction_N, p.neural_reaction_N, r"FEM $F$ [N]", r"neural $F$ [N]"),
        (
            s.stress_pnorm_utilization,
            s.predicted_stress_utilization,
            r"FEM $\Omega_8$",
            r"stress model $\widehat{\Omega}_8$",
        ),
    ]
    for axis, (truth, prediction, xlabel, ylabel) in zip(axes, panels):
        axis.scatter(truth, prediction, s=14, facecolors="none", edgecolors=BLACK, linewidths=0.7)
        limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
        axis.plot(limits, limits, color=HIGHLIGHT_RED, linewidth=0.9)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    axes[0].text(0.04, 0.94, f"$R^2={metrics['neural_energy_test']['r2']:.3f}$", transform=axes[0].transAxes, va="top")
    axes[1].text(0.04, 0.94, f"$R^2={metrics['neural_reaction_test']['r2']:.3f}$", transform=axes[1].transAxes, va="top")
    axes[2].text(0.04, 0.94, f"$R^2={metrics['stress_model_test']['r2']:.3f}$", transform=axes[2].transAxes, va="top")
    fig.tight_layout(w_pad=1.0)
    for suffix in ("pdf", "png"):
        target = FIGURES / f"fem_energy_surrogate_validation.{suffix}"
        temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
        fig.savefig(temporary, format=suffix, dpi=350)
        temporary.replace(target)
    plt.close(fig)


def plot_fronts(candidates: pd.DataFrame, verified: pd.DataFrame):
    apply_latex_style(7.5)
    labels = {
        "C037": (r"board material fraction $\phi_b$", r"inverse potential number $1/\Psi_U$"),
        "C038": (r"board material fraction $\phi_b$", r"stress utilization $\Omega_8$"),
        "C039": (r"inverse potential number $1/\Psi_U$", r"stress utilization $\Omega_8$"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for axis, case in zip(axes, ("C037", "C038", "C039")):
        predicted = candidates[candidates.case == case]
        direct = verified[(verified.case == case) & verified.path_success.astype(bool)]
        pareto_column = "mesh_stable_pareto" if "mesh_stable_pareto" in direct else "fem_pareto"
        final = direct[direct[pareto_column].astype(bool)]
        axis.scatter(
            predicted.surrogate_objective_1,
            predicted.surrogate_objective_2,
            s=14,
            facecolors="none",
            edgecolors=GRAY,
            linewidths=0.65,
            label="neural-screened",
        )
        axis.scatter(
            direct.fem_objective_1,
            direct.fem_objective_2,
            s=12,
            color=GRAY_DARK,
            alpha=0.7,
            label="direct FEM",
        )
        final = final.sort_values("fem_objective_1")
        axis.plot(
            final.fem_objective_1,
            final.fem_objective_2,
            "o-",
            color=HIGHLIGHT_RED,
            markersize=3.0,
            label="mesh-stable FEM front",
        )
        axis.set(xlabel=labels[case][0], ylabel=labels[case][1], title=case)
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=0.9)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"physics_pareto_fronts.{suffix}", dpi=350)
    plt.close(fig)


def plot_dimensionless_map(terminal: pd.DataFrame, front: pd.DataFrame):
    apply_latex_style(7.5)
    valid = terminal[terminal.path_success.astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35))
    scatter = axes[0].scatter(
        valid.board_material_fraction,
        valid.stored_potential_number,
        c=valid.stress_pnorm_utilization,
        cmap="Greys",
        s=20,
        edgecolors=BLACK,
        linewidths=0.25,
    )
    axes[0].set(xlabel=r"$\phi_b$", ylabel=r"$\Psi_U$")
    fig.colorbar(scatter, ax=axes[0], label=r"$\Omega_8$")
    axes[1].scatter(
        valid.aspect_ratio_H_over_P,
        valid.thickness_ratio_t_over_H,
        c=valid.curvature_index_t_over_Rmin,
        cmap="Greys",
        s=20,
        edgecolors=BLACK,
        linewidths=0.25,
    )
    axes[1].scatter(
        front.aspect_ratio_H_over_P,
        front.thickness_ratio_t_over_H,
        marker="x",
        color=HIGHLIGHT_RED,
        s=22,
        label="FEM Pareto",
    )
    axes[1].set(xlabel=r"aspect $H/P$", ylabel=r"slenderness $t/H$")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    fig.tight_layout(w_pad=1.0)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"dimensionless_performance_map.{suffix}", dpi=350)
    plt.close(fig)


def plot_mesh_study(data: pd.DataFrame):
    apply_latex_style(7.5)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    fields = [
        ("target_reaction_N", r"target reaction $F$ [N]"),
        ("stored_potential_number", r"potential number $\Psi_U$"),
        ("stress_pnorm_utilization", r"stress utilization $\Omega_8$"),
    ]
    for axis, (field, label) in zip(axes, fields):
        for selection_id, group in data.groupby("selection_id"):
            axis.plot(group.mesh, group[field], "o-", markersize=2.8, label=selection_id)
        axis.set(xlabel="elements per wavelength", ylabel=label)
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout(w_pad=0.9)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"mesh_convergence.{suffix}", dpi=350)
    plt.close(fig)


def plot_optimizer_convergence(history: pd.DataFrame):
    apply_latex_style(7.5)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    for axis, case in zip(axes, ("C037", "C038", "C039")):
        pivot = history[history.case == case].pivot(
            index="generation", columns="seed", values="normalized_hypervolume"
        )
        axis.plot(pivot.index, pivot.median(axis=1), color=BLACK, linewidth=1.1)
        axis.fill_between(
            pivot.index,
            pivot.quantile(0.25, axis=1),
            pivot.quantile(0.75, axis=1),
            color=GRAY_LIGHT,
            linewidth=0,
        )
        axis.axvline(42, color=HIGHLIGHT_RED, linestyle="--", linewidth=0.8)
        axis.set(
            title=case,
            xlabel="generation",
            ylabel="normalized hypervolume",
        )
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    fig.tight_layout(w_pad=0.9)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"physics_optimizer_convergence.{suffix}", dpi=350)
    plt.close(fig)


def plot_candidate_mesh_verification(comparison: pd.DataFrame):
    apply_latex_style(7.5)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    fields = [
        ("target_reaction_N", r"reaction $F$ [N]"),
        ("stored_potential_number", r"potential number $\Psi_U$"),
        ("stress_pnorm_utilization", r"stress utilization $\Omega_8$"),
    ]
    for axis, (field, label) in zip(axes, fields):
        x = comparison[f"{field}_mesh24"]
        y = comparison[f"{field}_mesh32"]
        axis.scatter(x, y, s=10, facecolors="none", edgecolors=GRAY_DARK, linewidths=0.5)
        bounds = [min(x.min(), y.min()), max(x.max(), y.max())]
        axis.plot(bounds, bounds, color=HIGHLIGHT_RED, linewidth=0.9)
        max_change = comparison[f"{field}_relative_change"].max()
        axis.text(0.04, 0.94, rf"max $\Delta={100*max_change:.1f}\%$", transform=axis.transAxes, va="top")
        axis.set(xlabel=f"24 elements: {label}", ylabel=f"32 elements: {label}")
        axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    fig.tight_layout(w_pad=0.9)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"candidate_mesh_verification.{suffix}", dpi=350)
    plt.close(fig)


def plot_optimized_gallery(
    selected: pd.DataFrame,
    refined,
    *,
    filename: str = "optimized_fem_stress_gallery",
    maximum_rows: int = 6,
):
    apply_latex_style(7.2)
    rows = min(len(selected), maximum_rows)
    chosen = selected.iloc[:rows]
    fig, axes = plt.subplots(rows, 2, figsize=(6.7, 1.45 * rows), squeeze=False)
    for row_axis, (_, design) in zip(axes, chosen.iterrows()):
        nodes, states, metrics = refined[str(design.selection_id)]
        target = states[-1]
        displacement = np.asarray(target["displacement"]).reshape(-1, 3)
        current = nodes + displacement[:, :2]
        stress = np.asarray(target["element_stress_MPa"])
        segments = np.stack((current[:-1], current[1:]), axis=1)
        collection = LineCollection(segments, cmap="inferno", linewidth=2.2)
        collection.set_array(stress)
        row_axis[0].add_collection(collection)
        row_axis[0].autoscale()
        row_axis[0].set_aspect("equal", adjustable="datalim")
        row_axis[0].axhline(
            float(target["lower_platen_surface_mm"]), color=GRAY_LIGHT, linewidth=0.6
        )
        row_axis[0].axhline(
            float(target["upper_platen_surface_mm"]), color=GRAY_LIGHT, linewidth=0.6
        )
        row_axis[0].set(
            title=f"{design.selection_id} ({design.case}), deformed stress field",
            xlabel="$x$ [mm]",
            ylabel="$y$ [mm]",
        )
        fig.colorbar(collection, ax=row_axis[0], label="MPa", fraction=0.05, pad=0.03)
        strain = [state["strain"] for state in states]
        reaction = [state["reaction_N"] for state in states]
        row_axis[1].plot(strain, reaction, "o-", color=BLACK, markersize=3)
        row_axis[1].set(
            title=(
                rf"$\phi_b={metrics['board_material_fraction']:.3f}$, "
                rf"$\Psi_U={metrics['stored_potential_number']:.4f}$, "
                rf"$\Omega_8={metrics['stress_pnorm_utilization']:.3f}$"
            ),
            xlabel="compressive strain",
            ylabel="reaction [N]",
        )
        for axis in row_axis:
            axis.grid(color=GRAY_LIGHT, linewidth=0.45)
    fig.tight_layout(h_pad=0.9, w_pad=0.7)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{filename}.{suffix}", dpi=350)
    plt.close(fig)


def write_summary(
    terminal,
    all_members,
    candidates,
    verified,
    raw_front,
    front,
    selected,
    mesh_data,
    thresholds,
    metrics,
    optimizer_convergence,
    membership_summary,
    sensitivity,
):
    direct_pool = pd.read_csv(RESULTS / "direct_verification_pool.csv")
    broad_mechanics = pd.read_csv(RESULTS / "fem_unique_geometries_mesh32.csv")
    c038_fine_mechanics = pd.read_csv(RESULTS / "fem_unique_geometries_mesh48.csv")
    convergence = {}
    for selection_id, group in mesh_data.groupby("selection_id"):
        group = group.sort_values("mesh")
        reference = group.iloc[-1]
        coarse = group.iloc[-2]
        convergence[selection_id] = {
            field: float(abs(coarse[field] - reference[field]) / max(abs(reference[field]), 1.0e-12))
            for field in (
                "target_reaction_N",
                "stored_potential_number",
                "stress_pnorm_utilization",
                "stress_q99_utilization",
                "stress_max_utilization",
            )
        }
    summary = {
        "protocol": {
            "strain_path": list(STRAINS),
            "elastic_modulus_MPa": 2899.0,
            "yield_stress_MPa": 60.0,
            "width_mm": 25.0,
            "radius_limit_mm": 0.9,
            "element_formulation": "corotational assumed-strain Timoshenko beam",
            "contact_mesh": (
                "prescribed normal motion at trough and periodic crowns with free "
                "tangential slip, unilateral remaining nodes, signed root reactions, "
                "and t/2 surface offset"
            ),
            "training_mesh": TRAINING_MESH,
            "final_verification_meshes": [*FINAL_MESHES, *C038_FINE_MESHES],
            "refined_meshes": [8, 12, 16, 24, 32, 40, 48],
            "optimizer_seeds": list(SEEDS),
            "protocol_fingerprint_training": protocol_fingerprint(TRAINING_MESH),
            "protocol_fingerprint_final": {
                str(mesh): protocol_fingerprint(mesh)
                for mesh in (*FINAL_MESHES, *C038_FINE_MESHES)
            },
        },
        "ledger_fingerprints": {
            "training_designs": str(terminal.design_ledger_fingerprint.iloc[0]),
            "direct_verification_pool": sorted(
                set(verified.candidate_ledger_fingerprint.astype(str))
            ),
            "classification_analyses": sorted(
                set(verified.analysis_fingerprint.astype(str))
            ),
            "c038_fine_analysis": sorted(
                set(
                    verified.loc[
                        verified.case == "C038", "fine_analysis_fingerprint"
                    ].astype(str)
                )
            ),
        },
        "thresholds": thresholds,
        "counts": {
            "FEM_dataset_designs": int(terminal.path_success.astype(bool).sum()),
            "surrogate_terminal_particles": int(len(all_members)),
            "surrogate_nondominated_candidates": int(len(candidates)),
            "direct_case_classifications": int(len(direct_pool)),
            "unique_broad_verification_geometries": int(len(broad_mechanics)),
            "c038_fine_case_classifications": int(
                (direct_pool.case == "C038").sum()
            ),
            "unique_c038_fine_geometries": int(len(c038_fine_mechanics)),
            "successful_publication_resolution_classifications": int(
                verified.path_success.astype(bool).sum()
            ),
            "publication_resolution_raw_front_members": int(len(raw_front)),
            "resolution_qualified_front_members": int(len(front)),
            "refined_selected_geometries": int(len(selected)),
        },
        "publication_resolution_raw_front_members_by_case": {
            key: int(value) for key, value in raw_front.groupby("case").size().items()
        },
        "resolution_qualified_front_members_by_case": {
            key: int(value) for key, value in front.groupby("case").size().items()
        },
        "surrogate_validation": metrics,
        "selected_mesh_40_to_48_relative_change": convergence,
        "all_candidate_front_membership_stability": membership_summary,
        "optimizer_convergence_by_seed": optimizer_convergence.to_dict("records"),
        "threshold_sensitivity": sensitivity.to_dict("records"),
        "maximum_initial_contact_gap_mm_training": float(
            terminal[["initial_upper_contact_gap_mm", "initial_lower_contact_gap_mm"]]
            .abs()
            .to_numpy()
            .max()
        ),
    }
    (RESULTS / "physics_optimization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def write_execution_manifest(args: argparse.Namespace, thresholds: dict[str, float]) -> None:
    """Write commands, fixed settings, package lock, and numerical file hashes."""

    input_paths = [
        ROOT / "run_paper_a.py",
        ROOT / "make_paper_a_gallery.py",
        ROOT / "run_paper_a_physics_optimization.py",
        ROOT / "run_coupled_size_shape.py",
        ROOT / "make_physics_tables.py",
        ROOT / "src" / "plot_style.py",
        ROOT / "requirements-paper-a.txt",
        ROOT / "manuscripts" / "paper_a" / "main.tex",
        ROOT / "src" / "cbopt" / "__init__.py",
        ROOT / "src" / "cbopt" / "mechanical_evaluator.py",
        ROOT / "src" / "cbopt" / "evaluator.py",
        ROOT / "src" / "cbopt" / "optimizers.py",
        ROOT / "src" / "cbfem" / "__init__.py",
        ROOT / "src" / "cbfem" / "geometry.py",
        ROOT / "src" / "cbfem" / "model.py",
        ROOT / "src" / "cbenergy" / "__init__.py",
        ROOT / "src" / "cbenergy" / "energy_network.py",
        ROOT.parent / "tests" / "test_physics_optimization.py",
        ROOT.parent / "tests" / "test_evidence_integrity.py",
        ROOT.parent / "tests" / "test_smoke.py",
        *[
            ROOT
            / "results"
            / "paper_a"
            / "coupled_size_shape"
            / f"C{case:03d}_all_members.csv"
            for case in (34, 35, 36)
        ],
    ]
    outputs = sorted(
        path
        for path in RESULTS.rglob("*")
        if path.is_file() and path.name != "execution_manifest.json"
    )
    manifest = {
        "commands": [
            "python project/run_paper_a.py",
            "python project/run_coupled_size_shape.py",
            "python project/make_paper_a_gallery.py",
            (
                "python project/run_paper_a_physics_optimization.py "
                "--allow-exhaustive-sweep "
                f"--dataset-size {args.dataset_size} --population {args.population} "
                f"--generations {args.generations}"
            ),
            "python project/make_physics_tables.py",
            "latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex",
            "python -m unittest discover -s tests -v",
        ],
        "fixed_settings": {
            "training_mesh": TRAINING_MESH,
            "terminal_meshes": list(FINAL_MESHES),
            "c038_fine_meshes": list(C038_FINE_MESHES),
            "strain_path": list(STRAINS),
            "optimizer_seeds": list(SEEDS),
            "dataset_size": int(args.dataset_size),
            "population": int(args.population),
            "generations": int(args.generations),
            "feature_count": FEATURE_COUNT,
            "thresholds": thresholds,
        },
        "execution_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "package_lock": (ROOT / "requirements-paper-a.txt").read_text(
            encoding="utf-8"
        ).splitlines(),
        "inputs": {
            str(path.relative_to(ROOT)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
            if path.is_relative_to(ROOT)
        } | {
            str(Path("..") / path.relative_to(ROOT.parent)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
            if not path.is_relative_to(ROOT)
        },
        "outputs": {
            str(path.relative_to(ROOT)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        },
    }
    (RESULTS / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-exhaustive-sweep",
        action="store_true",
        help=(
            "run the superseded all-candidate 24/32 and C038 40/48 sweep; "
            "the reported paper uses run_paper_a_fast_verification.py instead"
        ),
    )
    parser.add_argument("--dataset-size", type=int, default=80)
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--generations", type=int, default=56)
    args = parser.parse_args()
    if not args.allow_exhaustive_sweep:
        parser.error(
            "This entry point is the archived exhaustive protocol and is disabled "
            "by default. Run 'python project/run_paper_a_fast_verification.py' for "
            "the reported 9+3 verification, or pass --allow-exhaustive-sweep only "
            "when intentionally reproducing the superseded long audit."
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    pool = geometric_pool()
    np.savetxt(RESULTS / "feasible_design_pool.csv", pool, delimiter=",", header=",".join(VARIABLES), comments="")
    write_periodicity_audit(pool)
    selected_designs = farthest_point_sample(pool, args.dataset_size)
    designs = save_selected_designs(selected_designs)
    paths, terminal = generate_fem_dataset(designs, mesh=TRAINING_MESH)
    (
        networks,
        stress_model,
        scaler,
        neighbors,
        trust_limit,
        stress_calibration_margin,
        validation,
    ) = fit_surrogates(paths, terminal)
    successful = terminal[terminal.path_success.astype(bool)]
    training_terminal = successful[successful.split == "train"]
    thresholds = {
        "trust_limit": trust_limit,
        "minimum_potential": float(np.quantile(training_terminal.stored_potential_number, 0.40)),
        "minimum_potential_training_quantile": 0.40,
        "maximum_material_fraction": float(np.quantile(training_terminal.board_material_fraction, 0.65)),
        "maximum_material_training_quantile": 0.65,
    }
    predictor = build_predictor(
        networks, stress_model, scaler, neighbors, stress_calibration_margin
    )
    all_members, candidates, history = optimize_surrogates(
        pool, predictor, thresholds, args.population, args.generations
    )
    convergence_summary = summarize_optimizer_convergence(history)
    verification_pool = build_verification_pool(
        all_members, designs, predictor, thresholds
    )
    verified_by_mesh = {
        mesh: verify_candidates(verification_pool, mesh=mesh, thresholds=thresholds)
        for mesh in FINAL_MESHES
    }
    mesh_comparison, membership_summary = compare_mesh_fronts(
        verified_by_mesh[24], verified_by_mesh[32]
    )
    coarse_stable_keys = set(
        mesh_comparison.loc[
            mesh_comparison.fem_pareto_mesh24.astype(bool)
            & mesh_comparison.fem_pareto_mesh32.astype(bool),
            "design_key",
        ]
    )

    c038_pool = verification_pool[verification_pool.case == "C038"].copy()
    c038_by_mesh = {
        mesh: verify_candidates(c038_pool, mesh=mesh, thresholds=thresholds)
        for mesh in C038_FINE_MESHES
    }
    c038_final, c038_comparison, c038_fine_summary = analyze_c038_fine_meshes(
        c038_by_mesh[40], c038_by_mesh[48]
    )
    membership_summary["C038_fine"] = c038_fine_summary

    mesh32 = verified_by_mesh[32].copy()
    mesh32["design_key"] = mesh32.apply(
        lambda row: str(row.case)
        + ":"
        + ":".join(f"{float(row[name]):.9g}" for name in VARIABLES),
        axis=1,
    )
    mesh32_non_c038 = mesh32[mesh32.case != "C038"].copy()
    mesh32_non_c038["publication_mesh"] = 32
    c038_final["publication_mesh"] = 48
    verified = pd.concat([mesh32_non_c038, c038_final], ignore_index=True, sort=False)
    verified["mesh_stable_pareto"] = False
    non_c038 = verified.case != "C038"
    verified.loc[non_c038, "mesh_stable_pareto"] = verified.loc[
        non_c038, "design_key"
    ].isin(coarse_stable_keys)
    verified.loc[~non_c038, "mesh_stable_pareto"] = verified.loc[
        ~non_c038, "fine_mesh_stable_pareto"
    ].astype(bool)
    verified.to_csv(RESULTS / "fem_verified_terminal_final.csv", index=False)
    raw_front = pd.concat(
        [
            mesh32_non_c038[mesh32_non_c038.fem_pareto.astype(bool)],
            c038_final[c038_final.tolerance_pareto.astype(bool)],
        ],
        ignore_index=True,
        sort=False,
    )
    if raw_front.empty:
        raise RuntimeError("No direct-FEM Pareto member survived the declared constraints")
    front = verified[verified.mesh_stable_pareto].copy()
    raw_front.to_csv(RESULTS / "fem_publication_resolution_fronts.csv", index=False)
    front.to_csv(RESULTS / "fem_mesh_stable_pareto_fronts.csv", index=False)
    front.to_csv(RESULTS / "fem_rebuilt_pareto_fronts.csv", index=False)
    sensitivity = threshold_sensitivity(
        training_terminal,
        verified,
        float(c038_fine_summary["stress_absolute_tolerance"]),
    )
    selected = select_representatives(front)
    mesh_data, refined = run_mesh_study(selected)

    plot_validation(validation)
    plot_fronts(candidates, verified)
    plot_dimensionless_map(terminal, front)
    plot_mesh_study(mesh_data)
    plot_optimizer_convergence(history)
    plot_candidate_mesh_verification(mesh_comparison)
    plot_optimized_gallery(selected, refined)
    plot_optimized_gallery(
        selected[selected.case == "C039"],
        refined,
        filename="optimized_fem_stress_gallery_c039",
        maximum_rows=3,
    )
    write_summary(
        terminal,
        all_members,
        candidates,
        verified,
        raw_front,
        front,
        selected,
        mesh_data,
        thresholds,
        validation,
        convergence_summary,
        membership_summary,
        sensitivity,
    )
    write_execution_manifest(args, thresholds)
    print(
        f"Completed: {len(all_members)} terminal particles, {len(candidates)} screened candidates, "
        f"{len(front)} resolution-qualified FEM Pareto members, "
        f"{len(selected)} refined geometries.",
        flush=True,
    )


if __name__ == "__main__":
    main()
