#!/usr/bin/env python3
"""Run the new coupled size--shape cases C034--C036 for Paper A.

The archived Opt_CBv2 campaigns stop at separate size and shape studies.  This
script deliberately uses new case identifiers and records every seed so that
the new evidence cannot be confused with the public archive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cbopt.evaluator import evaluate_design  # noqa: E402
from cbopt.optimizers import run_mo_etpso  # noqa: E402


LOWER = np.asarray([0.1] * 5 + [0.0, 0.0, 4.8, 2.2, 0.15])
UPPER = np.asarray([10.0] * 5 + [1.0, 1.0, 7.9, 4.0, 0.25])
OUT = ROOT / "data" / "paper_a" / "opt_cbv2_final_fronts"
RESULTS = ROOT / "results" / "paper_a" / "coupled_size_shape"


def details(position: np.ndarray):
    shape = position[:7]
    wavelength, amplitude, thickness = map(float, position[7:10])
    return evaluate_design(
        shape,
        sample_size=320,
        wavelength_mm=wavelength,
        amplitude_mm=amplitude,
        thickness_mm=thickness,
        radius_limit_mm=0.9,
    )


def objectives(case: int, position: np.ndarray) -> tuple[np.ndarray, bool, float]:
    result = details(position)
    area = result.area_per_wavelength_mm
    inertia = result.inertia_per_wavelength_mm3
    modulus = result.effective_transverse_modulus_mpa
    if case == 34:
        values = np.asarray([area / 5.0, 1.0e3 / max(modulus, 1.0e-12)])
    elif case == 35:
        values = np.asarray([area / 5.0, 1.0 / max(inertia, 1.0e-12)])
    elif case == 36:
        values = np.asarray([100.0 * area / max(modulus, 1.0e-12), area / max(inertia, 1.0e-12)])
    else:
        raise ValueError(case)
    violation = max(0.0, 0.9 - result.minimum_radius_1_mm) + max(
        0.0, 0.9 - result.minimum_radius_2_mm
    )
    return values, bool(result.feasible), float(violation)


def nondominated(values: np.ndarray) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        if not keep[index]:
            continue
        dominates_index = np.all(values <= values[index], axis=1) & np.any(
            values < values[index], axis=1
        )
        if np.any(dominates_index):
            keep[index] = False
    return keep


def run_case(case: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for seed in (3, 11, 29):
        result = run_mo_etpso(
            lambda position: objectives(case, position),
            LOWER,
            UPPER,
            population_size=40,
            generations=50,
            inertia=0.7,
            seed=seed,
            reference=np.asarray([10.0, 10.0]),
        )
        for member, (position, values, feasible, violation) in enumerate(
            zip(
                result.positions,
                result.objectives,
                result.feasible,
                result.constraint_violation,
            )
        ):
            detail = details(position)
            records.append(
                {
                    "case": f"C{case:03d}",
                    "seed": seed,
                    "member": member,
                    "xline_all_values": values[0],
                    "yline_all_values": values[1],
                    "X_all_values": list(map(float, position)),
                    "area_per_wavelength_mm": detail.area_per_wavelength_mm,
                    "inertia_per_wavelength_mm3": detail.inertia_per_wavelength_mm3,
                    "effective_Ez_MPa": detail.effective_transverse_modulus_mpa,
                    "radius_1_mm": detail.minimum_radius_1_mm,
                    "radius_2_mm": detail.minimum_radius_2_mm,
                    "feasible": bool(feasible),
                    "constraint_violation": float(violation),
                    "pareto_member_within_seed": bool(result.pareto_mask[member]),
                }
            )
    data = pd.DataFrame(records)
    feasible = data[data.feasible].copy()
    feasible = feasible.drop_duplicates(subset=["xline_all_values", "yline_all_values"])
    mask = nondominated(feasible[["xline_all_values", "yline_all_values"]].to_numpy())
    front = feasible[mask].sort_values("xline_all_values").reset_index(drop=True)
    front["cross_seed_pareto"] = True
    return data, front


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for case in (34, 35, 36):
        all_members, front = run_case(case)
        all_members.to_csv(RESULTS / f"C{case:03d}_all_members.csv", index=False)
        front.to_csv(OUT / f"C{case:03d}_final.csv", index=False)
        print(
            f"C{case:03d}: {len(all_members)} terminal members, "
            f"{int(all_members.feasible.sum())} feasible, {len(front)} cross-seed Pareto members"
        )


if __name__ == "__main__":
    main()
