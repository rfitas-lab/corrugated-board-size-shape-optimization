"""Geometry evaluation and multi-objective optimization for Paper A."""

from .evaluator import BoardEvaluation, evaluate_design
from .optimizers import OptimizationResult, run_mo_etpso, run_nsga2

__all__ = [
    "BoardEvaluation",
    "OptimizationResult",
    "evaluate_design",
    "run_mo_etpso",
    "run_nsga2",
]
