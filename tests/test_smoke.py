from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT))


class EvaluatorSmokeTest(unittest.TestCase):
    def test_nurbs_evaluator_returns_finite_engineering_quantities(self):
        from src.cbopt import evaluate_design

        result = evaluate_design(
            np.array([1.0, 1.2, 0.9, 1.1, 1.0, 0.35, 0.55]),
            sample_size=240,
        )
        self.assertTrue(np.isfinite(result.objectives).all())
        self.assertGreater(result.area_per_wavelength_mm, 0.0)
        self.assertGreater(result.inertia_per_wavelength_mm3, 0.0)


if __name__ == "__main__":
    unittest.main()
