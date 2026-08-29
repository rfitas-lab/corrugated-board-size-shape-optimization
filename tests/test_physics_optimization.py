from pathlib import Path
import os
import sys
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper-a-tests")
sys.path.insert(0, str(PROJECT / "vendor"))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))


class PhysicsOptimizationSmokeTest(unittest.TestCase):
    def setUp(self):
        self.position = np.array(
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

    def test_dimensionless_geometry_is_feasible_and_finite(self):
        from cbopt.mechanical_evaluator import dimensionless_geometry

        values = dimensionless_geometry(self.position)
        self.assertGreaterEqual(values["radius_min_mm"], 0.9)
        self.assertEqual(values["geometry_feasible"], 1.0)
        self.assertGreater(values["board_material_fraction"], 0.0)
        self.assertTrue(np.isfinite(list(values.values())).all())

    def test_profile_mesh_preserves_declared_period(self):
        from cbopt.evaluator import nurbs_profile
        from cbopt.mechanical_evaluator import paper_a_profile_nodes

        nodes, arc_length = paper_a_profile_nodes(self.position, 12)
        self.assertEqual(nodes.shape, (13, 2))
        self.assertAlmostEqual(nodes[0, 0], 0.0)
        self.assertAlmostEqual(nodes[-1, 0], self.position[7])
        self.assertAlmostEqual(nodes[0, 1], nodes[-1, 1])
        self.assertAlmostEqual(nodes[:, 1].min(), 0.0)
        self.assertAlmostEqual(nodes[:, 1].max(), self.position[8])
        self.assertGreater(arc_length, self.position[7])

        x, y, (left, _, right) = nurbs_profile(
            self.position[:7],
            sample_size=1601,
            wavelength_mm=self.position[7],
            amplitude_mm=self.position[8],
        )
        self.assertAlmostEqual(y[left], y[right], places=12)
        tangent_left = np.array(
            [x[left + 1] - x[left - 1], y[left + 1] - y[left - 1]]
        )
        tangent_right = np.array(
            [x[right + 1] - x[right - 1], y[right + 1] - y[right - 1]]
        )
        tangent_left /= np.linalg.norm(tangent_left)
        tangent_right /= np.linalg.norm(tangent_right)
        np.testing.assert_allclose(tangent_left, tangent_right, atol=2.0e-8)

    def test_assumed_strain_element_matches_closed_form_energy(self):
        from autograd import grad
        from cbfem import BeamMaterial, BeamSection, CorotationalBeamModel

        length = 1.5
        modulus = 2899.0
        shear = modulus / 55.0
        section = BeamSection(25.0, 0.15)
        model = CorotationalBeamModel(
            np.array([[0.0, 0.0], [length, 0.0]]),
            BeamMaterial(modulus, shear),
            section,
        )
        theta = 1.0e-6
        rigid_shear = np.array([0.0, 0.0, theta, 0.0, 0.0, theta])
        expected_shear = (
            0.5
            * section.shear_correction
            * shear
            * section.area_mm2
            * length
            * theta**2
        )
        self.assertAlmostEqual(
            float(model._element_energy(rigid_shear)), expected_shear, delta=1.0e-18
        )

        pure_bending = np.array([0.0, 0.0, -theta, 0.0, 0.0, theta])
        curvature = 2.0 * theta / length
        expected_bending = 0.5 * modulus * section.inertia_mm4 * length * curvature**2
        self.assertAlmostEqual(
            float(model._element_energy(pure_bending)), expected_bending, delta=1.0e-18
        )

        trial = np.array([0.0, 0.0, -2.0e-4, 2.0e-4, -1.0e-4, 3.0e-4])
        analytical = np.asarray(grad(model._element_energy)(trial))
        step = 1.0e-7
        numerical = np.array(
            [
                (
                    float(model._element_energy(trial + step * np.eye(6)[index]))
                    - float(model._element_energy(trial - step * np.eye(6)[index]))
                )
                / (2.0 * step)
                for index in range(6)
            ]
        )
        np.testing.assert_allclose(analytical, numerical, rtol=2.0e-5, atol=2.0e-8)

    def test_dimensionless_group_identities(self):
        from cbopt.mechanical_evaluator import dimensionless_geometry, surrogate_features

        values = dimensionless_geometry(self.position)
        self.assertAlmostEqual(
            values["thickness_ratio_t_over_P"],
            values["aspect_ratio_H_over_P"] * values["thickness_ratio_t_over_H"],
        )
        forming_yield = (
            2899.0
            * values["curvature_index_t_over_Rmin"]
            / (2.0 * 60.0)
        )
        self.assertGreater(forming_yield, 0.0)
        features = surrogate_features(self.position)
        spacing = self.position[:5] / self.position[:5].sum()
        self.assertEqual(features.shape, (8,))
        np.testing.assert_allclose(features[:4], spacing[:4])
        np.testing.assert_allclose(features[4:6], self.position[5:7])
        self.assertAlmostEqual(features[6], values["aspect_ratio_H_over_P"])
        self.assertAlmostEqual(features[7], values["thickness_ratio_t_over_H"])

    def test_contact_path_equilibrium_and_acceptance(self):
        from cbopt.mechanical_evaluator import CompressionProtocol, solve_compression_path

        _, metrics = solve_compression_path(
            self.position,
            elements_per_wavelength=8,
            protocol=CompressionProtocol(strains=(0.0, 0.05)),
        )
        self.assertTrue(metrics["path_success"])
        self.assertTrue(metrics["all_raw_solver_success"])
        self.assertAlmostEqual(metrics["initial_upper_contact_gap_mm"], 0.0)
        self.assertAlmostEqual(metrics["initial_lower_contact_gap_mm"], 0.0)
        self.assertLess(metrics["maximum_reaction_imbalance_fraction"], 5.0e-3)
        self.assertLess(metrics["maximum_reaction_imbalance_force_fraction"], 5.0e-4)
        self.assertLessEqual(metrics["maximum_normalized_projected_gradient"], 5.0e-4)
        self.assertAlmostEqual(
            metrics["forming_yield_index"],
            2899.0 * metrics["curvature_index_t_over_Rmin"] / (2.0 * 60.0),
        )

    def test_nondominated_mask(self):
        from run_paper_a_physics_optimization import nondominated, tolerance_nondominated

        values = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [2.5, 3.5]])
        np.testing.assert_array_equal(nondominated(values), [True, True, True, False])
        close = np.array([[1.0, 1.00], [1.0, 0.99], [1.1, 0.80]])
        np.testing.assert_array_equal(
            tolerance_nondominated(close, np.array([0.0, 0.02])),
            [True, True, True],
        )

    def test_ledger_fingerprint_tracks_order_and_values(self):
        import pandas as pd
        from run_paper_a_physics_optimization import table_fingerprint

        ledger = pd.DataFrame({"case": ["C037", "C038"], "x": [0.1, 0.2]})
        baseline = table_fingerprint(ledger, ["case", "x"])
        self.assertEqual(baseline, table_fingerprint(ledger.copy(), ["case", "x"]))
        self.assertNotEqual(
            baseline,
            table_fingerprint(ledger.iloc[::-1].reset_index(drop=True), ["case", "x"]),
        )
        modified = ledger.copy()
        modified.loc[1, "x"] = 0.2001
        self.assertNotEqual(baseline, table_fingerprint(modified, ["case", "x"]))


if __name__ == "__main__":
    unittest.main()
