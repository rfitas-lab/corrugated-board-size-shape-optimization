"""Equal-budget MO-ETPSO and NSGA-II implementations for Paper A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class OptimizationResult:
    algorithm: str
    seed: int
    positions: Array
    objectives: Array
    feasible: Array
    constraint_violation: Array
    evaluations: int
    history_hypervolume: Array

    @property
    def pareto_mask(self) -> Array:
        idx = np.where(self.feasible)[0]
        mask = np.zeros(len(self.positions), dtype=bool)
        if idx.size:
            mask[idx[_nondominated_mask(self.objectives[idx])]] = True
        return mask


def _nondominated_mask(values: Array) -> Array:
    n = len(values)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominated_by[i] = False
        if dominated_by.any():
            keep[i] = False
    return keep


def _fast_fronts(values: Array, feasible: Array, violation: Array | None = None) -> list[Array]:
    n = len(values)
    domination_count = np.zeros(n, dtype=int)
    dominated: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]
    penalized = values.copy()
    if (~feasible).any():
        worst = np.nanmax(values[feasible], axis=0) if feasible.any() else np.ones(values.shape[1])
        extra = np.ones((np.sum(~feasible), 1)) if violation is None else violation[~feasible, None]
        penalized[~feasible] = worst * 10.0 + 1.0 + extra
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(penalized[p] <= penalized[q]) and np.any(penalized[p] < penalized[q]):
                dominated[p].append(q)
            elif np.all(penalized[q] <= penalized[p]) and np.any(penalized[q] < penalized[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [np.asarray(front, dtype=int) for front in fronts[:-1]]


def _crowding(values: Array, front: Array) -> Array:
    n = len(front)
    distance = np.zeros(n)
    if n <= 2:
        distance[:] = np.inf
        return distance
    vf = values[front]
    for j in range(values.shape[1]):
        order = np.argsort(vf[:, j])
        distance[order[[0, -1]]] = np.inf
        span = vf[order[-1], j] - vf[order[0], j]
        if span > 0:
            distance[order[1:-1]] += (vf[order[2:], j] - vf[order[:-2], j]) / span
    return distance


def _select_nsga(values: Array, feasible: Array, size: int, violation: Array | None = None) -> Array:
    selected: list[int] = []
    for front in _fast_fronts(values, feasible, violation):
        if len(selected) + len(front) <= size:
            selected.extend(front.tolist())
        else:
            crowd = _crowding(values, front)
            chosen = front[np.argsort(-crowd)[: size - len(selected)]]
            selected.extend(chosen.tolist())
            break
    return np.asarray(selected, dtype=int)


def _hypervolume_2d(values: Array, feasible: Array, reference: Array) -> float:
    vals = values[feasible]
    if not len(vals):
        return 0.0
    vals = vals[_nondominated_mask(vals)]
    vals = vals[np.argsort(vals[:, 0])]
    hv = 0.0
    y = reference[1]
    for x_i, y_i in vals:
        if x_i < reference[0] and y_i < y:
            hv += (reference[0] - x_i) * (y - y_i)
            y = y_i
    return float(max(hv, 0.0))


def _evaluate(
    population: Array,
    evaluator: Callable[[Array], tuple[Array, bool] | tuple[Array, bool, float]],
) -> tuple[Array, Array, Array]:
    output = [evaluator(x) for x in population]
    return (
        np.vstack([x[0] for x in output]),
        np.asarray([x[1] for x in output], dtype=bool),
        np.asarray([x[2] if len(x) > 2 else (0.0 if x[1] else 1.0) for x in output], dtype=float),
    )


def _sbx(rng: np.random.Generator, a: Array, b: Array, eta: float = 15.0) -> tuple[Array, Array]:
    u = rng.random(len(a))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    return 0.5 * ((1 + beta) * a + (1 - beta) * b), 0.5 * ((1 - beta) * a + (1 + beta) * b)


def _polynomial_mutation(rng: np.random.Generator, x: Array, lower: Array, upper: Array, eta: float = 20.0) -> Array:
    y = x.copy()
    for j in range(len(y)):
        if rng.random() > 1.0 / len(y):
            continue
        u = rng.random()
        delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        y[j] += delta * (upper[j] - lower[j])
    return np.clip(y, lower, upper)


def run_nsga2(
    evaluator: Callable[[Array], tuple[Array, bool] | tuple[Array, bool, float]],
    lower: Array,
    upper: Array,
    *,
    population_size: int = 50,
    generations: int = 80,
    seed: int = 0,
    reference: Array | None = None,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower, upper, (population_size, len(lower)))
    objectives, feasible, violation = _evaluate(population, evaluator)
    if reference is None:
        reference = np.nanmax(objectives, axis=0) * 1.1
    history = [_hypervolume_2d(objectives, feasible, reference)]
    for _ in range(generations - 1):
        mating = population[rng.integers(0, population_size, population_size)]
        children = []
        for i in range(0, population_size, 2):
            c1, c2 = _sbx(rng, mating[i], mating[(i + 1) % population_size])
            children.extend([
                _polynomial_mutation(rng, c1, lower, upper),
                _polynomial_mutation(rng, c2, lower, upper),
            ])
        children = np.asarray(children[:population_size])
        child_obj, child_feasible, child_violation = _evaluate(children, evaluator)
        combined = np.vstack((population, children))
        combined_obj = np.vstack((objectives, child_obj))
        combined_feasible = np.r_[feasible, child_feasible]
        combined_violation = np.r_[violation, child_violation]
        chosen = _select_nsga(combined_obj, combined_feasible, population_size, combined_violation)
        population, objectives, feasible, violation = (
            combined[chosen],
            combined_obj[chosen],
            combined_feasible[chosen],
            combined_violation[chosen],
        )
        history.append(_hypervolume_2d(objectives, feasible, reference))
    return OptimizationResult("NSGA-II", seed, population, objectives, feasible, violation, population_size * generations, np.asarray(history))


def run_mo_etpso(
    evaluator: Callable[[Array], tuple[Array, bool] | tuple[Array, bool, float]],
    lower: Array,
    upper: Array,
    *,
    population_size: int = 50,
    generations: int = 80,
    inertia: float = 0.7,
    c1: float = 2.05,
    c2: float = 2.05,
    seed: int = 0,
    reference: Array | None = None,
) -> OptimizationResult:
    """Corrected archive-based elitist PSO with Pareto/crowding selection."""

    rng = np.random.default_rng(seed)
    positions = rng.uniform(lower, upper, (population_size, len(lower)))
    velocities = rng.uniform(-1.0, 1.0, positions.shape) * (upper - lower)
    objectives, feasible, violation = _evaluate(positions, evaluator)
    pbest, pbest_obj, pbest_feasible, pbest_violation = positions.copy(), objectives.copy(), feasible.copy(), violation.copy()
    if reference is None:
        reference = np.nanmax(objectives, axis=0) * 1.1
    history = [_hypervolume_2d(objectives, feasible, reference)]
    chi = 2.0 / abs(2.0 - (c1 + c2) - np.sqrt((c1 + c2) ** 2 - 4.0 * (c1 + c2)))
    for _ in range(generations - 1):
        archive_idx = _select_nsga(objectives, feasible, min(population_size, len(positions)), violation)
        archive = positions[archive_idx]
        archive_obj = objectives[archive_idx]
        first = _fast_fronts(archive_obj, feasible[archive_idx], violation[archive_idx])[0]
        leaders = archive[first]
        crowd = _crowding(archive_obj, first)
        finite = np.where(np.isfinite(crowd), crowd, np.nanmax(crowd[np.isfinite(crowd)]) + 1 if np.isfinite(crowd).any() else 1.0)
        weights = finite + 1e-12
        weights = weights / weights.sum()
        chosen_leaders = leaders[rng.choice(len(leaders), population_size, p=weights)]
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = chi * (inertia * velocities + c1 * r1 * (pbest - positions) + c2 * r2 * (chosen_leaders - positions))
        trial = np.clip(positions + velocities, lower, upper)
        trial_obj, trial_feasible, trial_violation = _evaluate(trial, evaluator)
        for i in range(population_size):
            old = pbest_obj[i]
            new = trial_obj[i]
            new_dominates = (
                (trial_feasible[i] and not pbest_feasible[i])
                or (
                    trial_feasible[i] == pbest_feasible[i]
                    and (
                        (not trial_feasible[i] and trial_violation[i] < pbest_violation[i])
                        or (trial_feasible[i] and np.all(new <= old) and np.any(new < old))
                    )
                )
            )
            nondominated_tie = trial_feasible[i] == pbest_feasible[i] and np.isclose(trial_violation[i], pbest_violation[i]) and not (
                np.all(old <= new) and np.any(old < new)
            )
            if new_dominates or (nondominated_tie and rng.random() < 0.5):
                pbest[i], pbest_obj[i], pbest_feasible[i], pbest_violation[i] = trial[i], new, trial_feasible[i], trial_violation[i]
        combined = np.vstack((positions, trial))
        combined_obj = np.vstack((objectives, trial_obj))
        combined_feasible = np.r_[feasible, trial_feasible]
        combined_violation = np.r_[violation, trial_violation]
        selected = _select_nsga(combined_obj, combined_feasible, population_size, combined_violation)
        positions, objectives, feasible, violation = (
            combined[selected],
            combined_obj[selected],
            combined_feasible[selected],
            combined_violation[selected],
        )
        velocities = np.vstack((velocities, velocities))[selected]
        pbest, pbest_obj, pbest_feasible, pbest_violation = (
            np.vstack((pbest, pbest))[selected],
            np.vstack((pbest_obj, pbest_obj))[selected],
            np.r_[pbest_feasible, pbest_feasible][selected],
            np.r_[pbest_violation, pbest_violation][selected],
        )
        history.append(_hypervolume_2d(objectives, feasible, reference))
    return OptimizationResult("MO-ETPSO-R", seed, positions, objectives, feasible, violation, population_size * generations, np.asarray(history))
