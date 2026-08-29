from pathlib import Path
import gzip
import hashlib
import json
import os
import pickle
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
RESULTS = PROJECT / "results" / "paper_a" / "physics_optimization"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper-a-tests")
sys.path.insert(0, str(PROJECT / "vendor"))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))


class EvidenceIntegrityTest(unittest.TestCase):
    def test_fast_execution_manifest_hashes_all_declared_files(self):
        manifest = json.loads((RESULTS / "fast_execution_manifest.json").read_text())
        self.assertEqual(
            manifest["budget"],
            {"mesh32": 9, "mesh40": 3, "mesh8_formulation_check": 1},
        )
        for section in ("inputs", "outputs"):
            for relative, expected in manifest[section].items():
                path = PROJECT / relative
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, expected["sha256"], msg=str(path))
                self.assertEqual(path.stat().st_size, expected["bytes"], msg=str(path))
        required_inputs = {
            "src/cbfem/model.py",
            "src/cbopt/mechanical_evaluator.py",
            "results/paper_a/physics_optimization/models/stress_ensemble.joblib",
            "manuscripts/paper_a/main.tex",
        }
        required_outputs = {
            "results/paper_a/physics_optimization/formulation_verification.json",
            "results/paper_a/physics_optimization/threshold_sensitivity_fast.csv",
            "manuscripts/paper_a/generated_tables/formulation_verification.tex",
            "manuscripts/paper_a/generated_tables/threshold_sensitivity_fast.tex",
            "figures/paper_a/physics_pareto_fronts.pdf",
            "results/paper_a/physics_optimization/fem_representative_states_mesh40.pkl.gz",
        }
        self.assertTrue(required_inputs.issubset(manifest["inputs"]))
        self.assertTrue(required_outputs.issubset(manifest["outputs"]))

    def test_current_threshold_sensitivity_is_bound_to_mesh24_sets(self):
        sensitivity = pd.read_csv(RESULTS / "threshold_sensitivity_fast.csv")
        self.assertEqual(
            sensitivity.front_members.tolist(), [49, 31, 7, 81, 121, 122]
        )
        self.assertEqual(
            set(sensitivity.dominance_rule), {"strict mesh-24 dominance"}
        )
        self.assertFalse((RESULTS / "threshold_sensitivity.csv").exists())

    def test_formulation_verification_meets_declared_limits(self):
        checks = json.loads((RESULTS / "formulation_verification.json").read_text())
        self.assertLessEqual(checks["constant_shear_energy_relative_error"], 1.0e-9)
        self.assertLessEqual(checks["pure_bending_energy_relative_error"], 1.0e-9)
        self.assertLessEqual(checks["energy_gradient_max_relative_error"], 2.0e-5)
        self.assertTrue(checks["contact_path_success"])
        self.assertEqual(checks["contact_initial_gap_mm"], 0.0)
        self.assertLessEqual(
            checks["contact_reaction_imbalance_force_fraction"], 5.0e-4
        )
        self.assertLessEqual(
            checks["contact_normalized_projected_gradient"], 5.0e-4
        )

    def test_feasible_pool_has_periodic_ordinate_and_tangent(self):
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
        self.assertEqual(maximum_ordinate_gap, 0.0)
        self.assertLess(maximum_tangent_gap, 4.0e-14)
        audit = json.loads((RESULTS / "periodicity_audit.json").read_text())
        self.assertEqual(audit["profiles_checked"], len(pool))
        self.assertEqual(audit["maximum_endpoint_ordinate_gap_mm"], maximum_ordinate_gap)
        self.assertAlmostEqual(
            audit["maximum_unit_tangent_mismatch"], maximum_tangent_gap
        )

    def test_training_ledger_and_path_count(self):
        from run_paper_a_physics_optimization import VARIABLES, table_fingerprint

        designs = pd.read_csv(RESULTS / "fem_designs.csv")
        paths = pd.read_csv(RESULTS / "fem_paths_mesh16.csv")
        terminal = pd.read_csv(RESULTS / "fem_terminal_mesh16.csv")
        fingerprint = table_fingerprint(designs, ["design_id", "split", *VARIABLES])
        self.assertEqual(len(designs), 80)
        self.assertEqual(len(paths), 400)
        self.assertEqual(len(terminal), 80)
        self.assertEqual(set(paths.design_ledger_fingerprint.astype(str)), {fingerprint})
        self.assertEqual(set(terminal.design_ledger_fingerprint.astype(str)), {fingerprint})
        self.assertTrue(
            np.allclose(paths.lower_platen_surface_mm, -0.5 * paths.thickness_mm)
        )
        self.assertTrue(
            np.allclose(
                paths.upper_platen_surface_mm,
                paths.height_mm
                + 0.5 * paths.thickness_mm
                - paths.height_mm * paths.strain,
            )
        )

    def test_complete_coarse_ledger_uses_float_safe_geometry_keys(self):
        from run_paper_a_physics_optimization import (
            VARIABLES,
            add_geometry_key,
            table_fingerprint,
        )

        candidates = pd.read_csv(RESULTS / "direct_verification_pool.csv").drop_duplicates(
            subset=["case", *VARIABLES]
        ).reset_index(drop=True)
        geometries = candidates.drop_duplicates(subset=VARIABLES).reset_index(drop=True)
        mechanics = pd.read_csv(RESULTS / "fem_unique_geometries_mesh24.csv")
        verified = pd.read_csv(RESULTS / "fem_verified_terminal_fast.csv")
        self.assertEqual(len(candidates), 732)
        self.assertEqual(len(geometries), 572)
        self.assertEqual(len(mechanics), 572)
        self.assertEqual(len(verified), 732)
        self.assertTrue(verified.path_success.fillna(False).astype(bool).all())
        self.assertFalse(verified.protocol_fingerprint.isna().any())
        fingerprint = table_fingerprint(
            candidates, ["case", *VARIABLES, "trust_distance"]
        )
        self.assertEqual(
            set(verified.candidate_ledger_fingerprint.astype(str)), {fingerprint}
        )

        perturbed = geometries.copy()
        perturbed.loc[0, VARIABLES[0]] += 5.0e-13
        original_keys = add_geometry_key(geometries).geometry_key
        perturbed_keys = add_geometry_key(perturbed).geometry_key
        self.assertEqual(original_keys.iloc[0], perturbed_keys.iloc[0])
        mechanics_keys = set(add_geometry_key(mechanics).geometry_key)
        self.assertEqual(len(set(original_keys) & mechanics_keys), 572)

    def test_direct_classification_enforces_trust_distance(self):
        from run_paper_a_physics_optimization import direct_objectives

        verified = pd.read_csv(RESULTS / "fem_verified_terminal_fast.csv")
        summary = json.loads((RESULTS / "fast_verification_summary.json").read_text())
        trust_limit = float(summary["thresholds"]["trust_limit"])
        self.assertTrue(np.isfinite(verified.trust_distance).all())
        self.assertLessEqual(float(verified.trust_distance.max()), trust_limit)
        front = verified[verified.fem_pareto.astype(bool)]
        self.assertLessEqual(float(front.trust_distance.max()), trust_limit)

        outside = verified.iloc[0].to_dict()
        outside["trust_distance"] = 1.1 * trust_limit
        _, violation = direct_objectives(
            int(str(outside["case"])[1:]), outside, summary["thresholds"]
        )
        self.assertGreater(violation, 0.0)

    def test_fast_front_counts_and_objectives_are_finite(self):
        front = pd.read_csv(RESULTS / "fem_rebuilt_pareto_fronts_fast.csv")
        summary = json.loads((RESULTS / "fast_verification_summary.json").read_text())
        expected = {"C037": 58, "C038": 31, "C039": 121}
        self.assertEqual(front.groupby("case").size().to_dict(), expected)
        self.assertEqual(summary["coarse_front_members"], expected)
        self.assertEqual(len(front), 210)
        self.assertTrue(np.isfinite(front.fem_objective_1).all())
        self.assertTrue(np.isfinite(front.fem_objective_2).all())
        self.assertTrue(front.fem_pareto.astype(bool).all())

    def test_representative_budget_and_strict_success(self):
        from run_paper_a_physics_optimization import VARIABLES, add_geometry_key

        selected = pd.read_csv(RESULTS / "selected_optimized_geometries.csv")
        refined = pd.read_csv(RESULTS / "selected_refined_geometries.csv")
        mesh32 = pd.read_csv(RESULTS / "fem_representative_geometries_mesh32.csv")
        mesh40 = pd.read_csv(RESULTS / "fem_representative_geometries_mesh40.csv")
        self.assertEqual(len(selected), 9)
        self.assertEqual(len(selected.drop_duplicates(VARIABLES)), 9)
        self.assertEqual(
            selected.groupby("case").size().to_dict(),
            {"C037": 3, "C038": 3, "C039": 3},
        )
        self.assertEqual(len(refined), 3)
        self.assertEqual(set(refined.case), {"C037", "C038", "C039"})
        self.assertTrue(refined.selection_role.eq("compromise").all())
        self.assertEqual(len(mesh32), 9)
        self.assertEqual(len(mesh40), 3)
        self.assertTrue(mesh32.path_success.fillna(False).astype(bool).all())
        self.assertTrue(mesh40.path_success.fillna(False).astype(bool).all())
        self.assertEqual(
            set(add_geometry_key(refined).geometry_key),
            set(add_geometry_key(mesh40).geometry_key),
        )

    def test_refined_state_cache_regenerates_the_stress_gallery(self):
        refined = pd.read_csv(RESULTS / "selected_refined_geometries.csv")
        cache_path = RESULTS / "fem_representative_states_mesh40.pkl.gz"
        with gzip.open(cache_path, "rb") as stream:
            states = pickle.load(stream)
        self.assertEqual(set(states), set(refined.selection_id.astype(str)))
        for nodes, path_states, metrics in states.values():
            self.assertEqual(nodes.ndim, 2)
            self.assertEqual(nodes.shape[1], 2)
            self.assertTrue(np.isfinite(nodes).all())
            self.assertEqual(len(path_states), 5)
            self.assertTrue(bool(metrics["path_success"]))

    def test_representative_convergence_matches_summary(self):
        convergence = pd.read_csv(RESULTS / "representative_convergence_summary.csv")
        summary = json.loads((RESULTS / "fast_verification_summary.json").read_text())
        self.assertEqual(len(convergence), 9)
        self.assertEqual(
            convergence.stored_potential_number_relative_change_32_to_40.notna().sum(),
            3,
        )
        for metric, expected in summary["maximum_relative_change_24_to_32"].items():
            self.assertAlmostEqual(
                expected,
                float(convergence[f"{metric}_relative_change_24_to_32"].max()),
            )
        for metric, expected in summary["maximum_relative_change_32_to_40"].items():
            self.assertAlmostEqual(
                expected,
                float(convergence[f"{metric}_relative_change_32_to_40"].max()),
            )

    def test_accepted_paths_meet_solver_and_contact_rules(self):
        for name in (
            "fem_terminal_mesh16.csv",
            "fem_unique_geometries_mesh24.csv",
            "fem_representative_geometries_mesh32.csv",
            "fem_representative_geometries_mesh40.csv",
        ):
            data = pd.read_csv(RESULTS / name)
            accepted = data[data.path_success.fillna(False).astype(bool)]
            self.assertTrue(accepted.all_raw_solver_success.astype(bool).all())
            self.assertLessEqual(
                float(accepted.maximum_normalized_projected_gradient.max()), 5.0e-4
            )
            self.assertLessEqual(
                float(accepted.maximum_reaction_imbalance_force_fraction.max()),
                5.0e-4,
            )
            self.assertTrue(
                (
                    accepted.fallback_solver_state_count
                    == accepted.independent_kkt_state_count
                ).all()
            )
            work_potential_gap = (
                np.abs(accepted.external_work_Nmm - accepted.target_potential_energy_Nmm)
                / np.maximum(np.abs(accepted.target_potential_energy_Nmm), 1.0e-12)
            )
            self.assertLessEqual(float(work_potential_gap.max()), 0.05)
            self.assertTrue(np.allclose(accepted.initial_upper_contact_gap_mm, 0.0))
            self.assertTrue(np.allclose(accepted.initial_lower_contact_gap_mm, 0.0))

    def test_rejected_mesh48_attempt_is_not_used_as_accepted_evidence(self):
        rejected = pd.read_csv(
            RESULTS / "fem_representative_geometries_mesh48_rejected_solver_audit.csv"
        )
        used = pd.read_csv(RESULTS / "representative_mesh_validation.csv")
        self.assertEqual(len(rejected), 3)
        self.assertFalse(rejected.path_success.fillna(False).astype(bool).any())
        self.assertEqual(set(used.mesh), {24, 32, 40})


if __name__ == "__main__":
    unittest.main()
