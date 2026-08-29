"""A corotational assumed-strain Timoshenko beam finite-element model.

Each node has three degrees of freedom: global x displacement, global y
displacement and section rotation. The element energy is evaluated in a
corotated frame, providing geometric nonlinearity. Rigid horizontal platens are
represented through exact unilateral displacement bounds or, optionally, a
penalty regularization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import autograd.numpy as anp
from autograd import grad
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class BeamMaterial:
    elastic_modulus_MPa: float
    shear_modulus_MPa: float
    yield_stress_MPa: float | None = None
    hardening_ratio: float = 0.02


@dataclass(frozen=True)
class BeamSection:
    width_mm: float
    thickness_mm: float
    shear_correction: float = 5.0 / 6.0

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.thickness_mm

    @property
    def inertia_mm4(self) -> float:
        return self.width_mm * self.thickness_mm**3 / 12.0


@dataclass
class SolveResult:
    displacement: np.ndarray
    success: bool
    message: str
    iterations: int
    potential_energy_Nmm: float
    gradient_norm: float
    top_reaction_N: float
    lower_reaction_N: float
    contact_penetration_mm: float
    constraint_reactions_N: np.ndarray
    element_results: dict[str, np.ndarray]
    optimizer_method: str
    primary_success: bool
    primary_message: str
    primary_iterations: int
    fallback_used: bool
    fallback_success: bool
    fallback_message: str
    fallback_iterations: int
    fallback_gradient_norm: float


class CorotationalBeamModel:
    def __init__(
        self,
        nodes: np.ndarray,
        material: BeamMaterial,
        section: BeamSection,
        connectivity: np.ndarray | None = None,
    ) -> None:
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (n, 2)")
        if connectivity is None:
            connectivity = np.column_stack(
                [np.arange(len(nodes) - 1), np.arange(1, len(nodes))]
            )
        connectivity = np.asarray(connectivity, dtype=int)
        if connectivity.ndim != 2 or connectivity.shape[1] != 2:
            raise ValueError("connectivity must have shape (m, 2)")
        if np.any(connectivity < 0) or np.any(connectivity >= len(nodes)):
            raise ValueError("connectivity references an invalid node")
        self.nodes = nodes
        self.connectivity = connectivity
        self.material = material
        self.section = section
        self.n_dof = 3 * len(nodes)
        delta = nodes[connectivity[:, 1]] - nodes[connectivity[:, 0]]
        self.initial_lengths = np.linalg.norm(delta, axis=1)
        if np.any(self.initial_lengths <= 0):
            raise ValueError("zero-length beam element")
        self.initial_angles = np.arctan2(delta[:, 1], delta[:, 0])
        self._self_contact_pair_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def _element_energy(self, full_displacement: anp.ndarray) -> anp.ndarray:
        elastic_modulus = float(self.material.elastic_modulus_MPa)
        shear_modulus = float(self.material.shear_modulus_MPa)
        area = float(self.section.area_mm2)
        inertia = float(self.section.inertia_mm4)
        shear_correction = float(self.section.shear_correction)
        nodal = anp.reshape(full_displacement, (-1, 3))
        current_coordinates = anp.asarray(self.nodes) + nodal[:, :2]
        node_1 = self.connectivity[:, 0]
        node_2 = self.connectivity[:, 1]
        delta = current_coordinates[node_2] - current_coordinates[node_1]
        current_length = anp.sqrt(anp.sum(delta**2, axis=1))
        current_angle = anp.arctan2(delta[:, 1], delta[:, 0])
        initial_length = anp.asarray(self.initial_lengths)
        initial_angle = anp.asarray(self.initial_angles)

        axial_extension = current_length - initial_length
        rotation_1 = nodal[node_1, 2] + initial_angle - current_angle
        rotation_2 = nodal[node_2, 2] + initial_angle - current_angle

        # Constant axial strain and curvature are integrated through the
        # thickness.  A single assumed mean shear strain is integrated once
        # along the element.  This reduced/assumed-strain treatment is used in
        # both elastic and smoothly yielding calculations, avoids the locking
        # of a fully integrated two-node shear field, and makes the declared
        # energy identical to the one differentiated by the solver.
        xi = anp.asarray(
            [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
        )
        gauss_weights = anp.asarray(
            [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]
        )
        half_thickness = 0.5 * float(self.section.thickness_mm)
        z = half_thickness * xi
        thickness_weights = half_thickness * gauss_weights
        axial_strain = axial_extension / initial_length
        curvature = (rotation_2 - rotation_1) / initial_length
        fiber_strain = axial_strain[:, None] - curvature[:, None] * z[None, :]
        if self.material.yield_stress_MPa is None:
            density = 0.5 * elastic_modulus * fiber_strain**2
        else:
            # Monotonic deformation-plasticity regularization. Longitudinal
            # fibre strains are integrated through the thickness; the smooth
            # saturation law approaches the prescribed yield stress and keeps
            # a small hardening tangent.
            yield_stress = float(self.material.yield_stress_MPa)
            hardening = float(self.material.hardening_ratio)
            if yield_stress <= 0.0 or not (0.0 <= hardening <= 1.0):
                raise ValueError(
                    "yield stress must be positive and hardening ratio within [0, 1]"
                )
            scaled = elastic_modulus * fiber_strain / yield_stress
            saturation_energy = (
                elastic_modulus
                * fiber_strain**2
                / (anp.sqrt(1.0 + scaled**2) + 1.0)
            )
            density = (
                hardening * 0.5 * elastic_modulus * fiber_strain**2
                + (1.0 - hardening) * saturation_energy
            )
        longitudinal_energy = (
            float(self.section.width_mm)
            * initial_length
            * anp.sum(density * thickness_weights[None, :], axis=1)
        )
        shear_strain = 0.5 * (rotation_1 + rotation_2)
        shear_energy = (
            0.5
            * shear_correction
            * shear_modulus
            * area
            * initial_length
            * shear_strain**2
        )
        return anp.sum(longitudinal_energy + shear_energy)

    def _contact_energy(
        self,
        full_displacement: anp.ndarray,
        lower_y_mm: float | None,
        upper_y_mm: float | None,
        penalty_N_mm: float,
    ) -> anp.ndarray:
        current_y = anp.asarray(self.nodes[:, 1]) + full_displacement[1::3]
        energy = 0.0
        if lower_y_mm is not None:
            lower_penetration = anp.maximum(float(lower_y_mm) - current_y, 0.0)
            energy = energy + 0.5 * float(penalty_N_mm) * anp.sum(lower_penetration**2)
        if upper_y_mm is not None:
            upper_penetration = anp.maximum(current_y - float(upper_y_mm), 0.0)
            energy = energy + 0.5 * float(penalty_N_mm) * anp.sum(upper_penetration**2)
        return energy

    def _self_contact_energy(
        self,
        full_displacement: anp.ndarray,
        contact_distance_mm: float,
        penalty_N_mm: float,
    ) -> anp.ndarray:
        """Regularized nonlocal contact between non-neighbouring beam segments.

        Contact points are the element midpoints. With the refined meshes used
        for collapse (24 elements per wavelength), this supplies a
        differentiable node-to-segment-scale barrier while excluding adjacent
        elements and initially touching material. It is intended for monotonic
        platen collapse, not friction or adhesive contact.
        """

        nodal = anp.reshape(full_displacement, (-1, 3))
        coordinates = anp.asarray(self.nodes) + nodal[:, :2]
        node_1 = self.connectivity[:, 0]
        node_2 = self.connectivity[:, 1]
        midpoints = 0.5 * (coordinates[node_1] + coordinates[node_2])
        initial_midpoints = 0.5 * (
            self.nodes[self.connectivity[:, 0]] + self.nodes[self.connectivity[:, 1]]
        )
        cache_key = float(contact_distance_mm)
        if cache_key not in self._self_contact_pair_cache:
            pair_i = []
            pair_j = []
            exclusion = 2
            initial_clearance = 1.05 * cache_key
            for first in range(len(self.connectivity)):
                for second in range(first + exclusion + 1, len(self.connectivity)):
                    if np.linalg.norm(initial_midpoints[first] - initial_midpoints[second]) > initial_clearance:
                        pair_i.append(first)
                        pair_j.append(second)
            self._self_contact_pair_cache[cache_key] = (
                np.asarray(pair_i, dtype=int), np.asarray(pair_j, dtype=int)
            )
        pair_i, pair_j = self._self_contact_pair_cache[cache_key]
        if len(pair_i) == 0:
            return 0.0
        delta = midpoints[anp.asarray(pair_i)] - midpoints[anp.asarray(pair_j)]
        distance = anp.sqrt(anp.sum(delta**2, axis=1) + 1.0e-16)
        penetration = anp.maximum(float(contact_distance_mm) - distance, 0.0)
        return 0.5 * float(penalty_N_mm) * anp.sum(penetration**2)

    def solve(
        self,
        fixed_dofs: Mapping[int, float],
        nodal_loads: np.ndarray | None = None,
        lower_y_mm: float | None = None,
        upper_y_mm: float | None = None,
        contact_penalty_N_mm: float = 1e5,
        contact_mode: str = "bounds",
        self_contact_distance_mm: float | None = None,
        self_contact_penalty_N_mm: float = 5.0e3,
        tied_dofs: list[tuple[int, int]] | None = None,
        linear_springs: list[tuple[int, int, float]] | None = None,
        rotational_hinges: list[tuple[int, int, float, float, float]] | None = None,
        initial_displacement: np.ndarray | None = None,
        tolerance: float = 1e-7,
        max_iterations: int = 3000,
    ) -> SolveResult:
        loads = np.zeros(self.n_dof) if nodal_loads is None else np.asarray(nodal_loads, dtype=float)
        if loads.shape != (self.n_dof,):
            raise ValueError("nodal_loads must contain one value per degree of freedom")
        parent = np.arange(self.n_dof)

        def find_root(dof: int) -> int:
            while parent[dof] != dof:
                parent[dof] = parent[parent[dof]]
                dof = parent[dof]
            return int(dof)

        def union(dof_1: int, dof_2: int) -> None:
            root_1 = find_root(dof_1)
            root_2 = find_root(dof_2)
            if root_1 != root_2:
                parent[root_2] = root_1

        for dof_1, dof_2 in tied_dofs or []:
            if not (0 <= int(dof_1) < self.n_dof and 0 <= int(dof_2) < self.n_dof):
                raise ValueError("tied degree of freedom is outside the model")
            union(int(dof_1), int(dof_2))
        roots = np.asarray([find_root(dof) for dof in range(self.n_dof)], dtype=int)

        fixed_by_root: dict[int, float] = {}
        for dof, value in fixed_dofs.items():
            root = int(roots[int(dof)])
            value = float(value)
            if root in fixed_by_root and not np.isclose(fixed_by_root[root], value):
                raise ValueError("conflicting prescribed values within a tied DOF group")
            fixed_by_root[root] = value
        unique_roots = list(dict.fromkeys(roots.tolist()))
        free_roots = [root for root in unique_roots if root not in fixed_by_root]
        free_index_by_root = {root: index + 1 for index, root in enumerate(free_roots)}
        free_map = np.asarray(
            [free_index_by_root.get(int(root), 0) for root in roots], dtype=int
        )
        prescribed = np.asarray(
            [fixed_by_root.get(int(root), 0.0) for root in roots], dtype=float
        )
        fixed_mask = free_map == 0
        if initial_displacement is None:
            initial_full = prescribed.copy()
        else:
            initial_full = np.asarray(initial_displacement, dtype=float).copy()
            if initial_full.shape != (self.n_dof,):
                raise ValueError("initial_displacement has the wrong size")
            initial_full[fixed_mask] = prescribed[fixed_mask]
        initial_free = np.asarray(
            [initial_full[roots == root].mean() for root in free_roots], dtype=float
        )

        if contact_mode not in {"bounds", "penalty"}:
            raise ValueError("contact_mode must be 'bounds' or 'penalty'")
        free_bounds = []
        for root in free_roots:
            lower_bound = -np.inf
            upper_bound = np.inf
            for dof in np.flatnonzero(roots == root):
                if contact_mode == "bounds" and dof % 3 == 1:
                    node = dof // 3
                    if lower_y_mm is not None:
                        lower_bound = max(
                            lower_bound, float(lower_y_mm - self.nodes[node, 1])
                        )
                    if upper_y_mm is not None:
                        upper_bound = min(
                            upper_bound, float(upper_y_mm - self.nodes[node, 1])
                        )
            if lower_bound > upper_bound:
                raise ValueError("tied DOFs produce inconsistent contact bounds")
            free_bounds.append((lower_bound, upper_bound))

        free_map_ag = anp.asarray(free_map)
        prescribed_ag = anp.asarray(prescribed)
        loads_ag = anp.asarray(loads)
        springs = []
        for dof_1, dof_2, stiffness in linear_springs or []:
            if not (0 <= int(dof_1) < self.n_dof and 0 <= int(dof_2) < self.n_dof):
                raise ValueError("spring degree of freedom is outside the model")
            if float(stiffness) < 0.0:
                raise ValueError("spring stiffness must be non-negative")
            springs.append((int(dof_1), int(dof_2), float(stiffness)))

        hinges = []
        for dof_1, dof_2, stiffness, yield_moment, hardening in rotational_hinges or []:
            if not (0 <= int(dof_1) < self.n_dof and 0 <= int(dof_2) < self.n_dof):
                raise ValueError("rotational-hinge degree of freedom is outside the model")
            if int(dof_1) % 3 != 2 or int(dof_2) % 3 != 2:
                raise ValueError("rotational hinges must connect rotational degrees of freedom")
            if float(stiffness) <= 0.0 or float(yield_moment) <= 0.0:
                raise ValueError("rotational-hinge stiffness and yield moment must be positive")
            if not 0.0 <= float(hardening) <= 1.0:
                raise ValueError("rotational-hinge hardening must lie within [0, 1]")
            hinges.append(
                (int(dof_1), int(dof_2), float(stiffness), float(yield_moment), float(hardening))
            )

        def spring_energy(full: anp.ndarray) -> anp.ndarray:
            energy = 0.0
            for dof_1, dof_2, stiffness in springs:
                energy = energy + 0.5 * stiffness * (full[dof_2] - full[dof_1]) ** 2
            return energy

        def hinge_energy(full: anp.ndarray) -> anp.ndarray:
            """Smooth elastic--plastic energy for pre-formed crease lines.

            The relative rotation is measured from the supplied, stress-free
            formed geometry.  Its moment approaches the specified yield
            moment and retains a user-controlled post-yield tangent.
            """
            energy = 0.0
            for dof_1, dof_2, stiffness, yield_moment, hardening in hinges:
                relative_rotation = full[dof_2] - full[dof_1]
                scaled = stiffness * relative_rotation / yield_moment
                saturation = (
                    stiffness
                    * relative_rotation**2
                    / (anp.sqrt(1.0 + scaled**2) + 1.0)
                )
                energy = energy + (
                    hardening * 0.5 * stiffness * relative_rotation**2
                    + (1.0 - hardening) * saturation
                )
            return energy

        def total_potential(free: anp.ndarray) -> anp.ndarray:
            padded_free = anp.concatenate((anp.zeros(1), free))
            full = prescribed_ag + padded_free[free_map_ag]
            return (
                self._element_energy(full)
                + spring_energy(full)
                + hinge_energy(full)
                + (
                    self._contact_energy(
                        full,
                        lower_y_mm=lower_y_mm,
                        upper_y_mm=upper_y_mm,
                        penalty_N_mm=contact_penalty_N_mm,
                    )
                    if contact_mode == "penalty"
                    else 0.0
                )
                + (
                    self._self_contact_energy(
                        full,
                        contact_distance_mm=float(self_contact_distance_mm),
                        penalty_N_mm=float(self_contact_penalty_N_mm),
                    )
                    if self_contact_distance_mm is not None
                    and float(self_contact_distance_mm) > 0.0
                    else 0.0
                )
                - anp.dot(loads_ag, full)
            )

        gradient = grad(total_potential)
        primary = minimize(
            fun=lambda q: float(total_potential(q)),
            x0=initial_free,
            jac=lambda q: np.asarray(gradient(q), dtype=float),
            method="L-BFGS-B",
            bounds=free_bounds,
            options={"ftol": min(float(tolerance), 1.0e-10), "gtol": tolerance, "maxiter": max_iterations, "maxls": 80},
        )

        def projected_gradient_at(values: np.ndarray) -> np.ndarray:
            projected = np.asarray(gradient(values), dtype=float).copy()
            if contact_mode == "bounds":
                for index, ((lower_bound, upper_bound), value) in enumerate(
                    zip(free_bounds, values)
                ):
                    if (
                        np.isfinite(lower_bound)
                        and value <= lower_bound + 2e-7
                        and projected[index] > 0
                    ):
                        projected[index] = 0.0
                    if (
                        np.isfinite(upper_bound)
                        and value >= upper_bound - 2e-7
                        and projected[index] < 0
                    ):
                        projected[index] = 0.0
            return projected

        primary_projected = projected_gradient_at(primary.x)
        primary_norm = float(np.linalg.norm(primary_projected, ord=np.inf))
        result = primary
        optimizer_method = "L-BFGS-B"
        fallback_used = False
        fallback_success = False
        fallback_message = "not invoked"
        fallback_iterations = 0
        fallback_gradient_norm = float("nan")
        # A second algorithm supplies an independent bound-constrained KKT
        # check whenever the primary termination flag is negative.  Successful
        # primary solves are still screened by the explicit projected-gradient
        # acceptance rule at the path level.
        if not bool(primary.success):
            fallback = minimize(
                fun=lambda q: float(total_potential(q)),
                x0=np.asarray(primary.x, dtype=float),
                jac=lambda q: np.asarray(gradient(q), dtype=float),
                method="SLSQP",
                bounds=free_bounds,
                options={
                    "ftol": min(float(tolerance), 1.0e-10),
                    "maxiter": min(int(max_iterations), 120),
                    "disp": False,
                },
            )
            fallback_used = True
            fallback_norm = float(
                np.linalg.norm(projected_gradient_at(fallback.x), ord=np.inf)
            )
            fallback_success = bool(fallback.success)
            fallback_message = str(fallback.message)
            fallback_iterations = int(fallback.nit)
            fallback_gradient_norm = fallback_norm
            if (
                bool(fallback.success)
                or fallback_norm < primary_norm
                or (
                    np.isclose(fallback_norm, primary_norm)
                    and float(fallback.fun) < float(primary.fun)
                )
            ):
                result = fallback
                optimizer_method = "SLSQP"
        padded_free = np.concatenate(([0.0], result.x))
        full = prescribed + padded_free[free_map]
        free_gradient = np.asarray(gradient(result.x), dtype=float)
        projected_gradient = projected_gradient_at(result.x)
        def full_potential(full_vector: anp.ndarray) -> anp.ndarray:
            return (
                self._element_energy(full_vector)
                + spring_energy(full_vector)
                + hinge_energy(full_vector)
                + (
                    self._contact_energy(
                        full_vector,
                        lower_y_mm=lower_y_mm,
                        upper_y_mm=upper_y_mm,
                        penalty_N_mm=contact_penalty_N_mm,
                    )
                    if contact_mode == "penalty"
                    else 0.0
                )
                - anp.dot(loads_ag, full_vector)
            )
        full_gradient = np.asarray(grad(full_potential)(full), dtype=float)
        constraint_reactions = np.zeros(self.n_dof)
        constraint_reactions[fixed_mask] = -full_gradient[fixed_mask]
        current_y = self.nodes[:, 1] + full[1::3]
        upper_penetration = (
            np.clip(current_y - upper_y_mm, 0.0, None)
            if upper_y_mm is not None
            else np.zeros(len(current_y))
        )
        lower_penetration = (
            np.clip(lower_y_mm - current_y, 0.0, None)
            if lower_y_mm is not None
            else np.zeros(len(current_y))
        )
        if contact_mode == "bounds":
            contact_tolerance = 2e-7
            top_active = (
                current_y >= float(upper_y_mm) - contact_tolerance
                if upper_y_mm is not None
                else np.zeros(len(current_y), dtype=bool)
            )
            lower_active = (
                current_y <= float(lower_y_mm) + contact_tolerance
                if lower_y_mm is not None
                else np.zeros(len(current_y), dtype=bool)
            )
            # A tied displacement group has one generalized reaction equal to
            # the sum of the member gradients. Summing absolute nodal
            # gradients would incorrectly count the internal tie forces.
            root_gradient: dict[int, float] = {}
            for root in unique_roots:
                if root in free_index_by_root:
                    root_gradient[root] = float(
                        free_gradient[free_index_by_root[root] - 1]
                    )
                else:
                    root_gradient[root] = float(full_gradient[roots == root].sum())
            # Sum signed generalized reactions on each rigid surface.  Free
            # unilateral-contact roots already have the correct KKT sign;
            # prescribed crown/trough roots represent perfect attachments and
            # may carry an opposing local contribution.  Clipping each root
            # separately would discard that contribution and can create a
            # spurious top--bottom force imbalance even though the global
            # internal-force sum is zero.
            top_reaction = 0.0
            lower_reaction = 0.0
            for root in unique_roots:
                vertical_dofs = [
                    dof for dof in np.flatnonzero(roots == root) if dof % 3 == 1
                ]
                if not vertical_dofs:
                    continue
                contact_nodes = np.asarray(vertical_dofs, dtype=int) // 3
                generalized_gradient = root_gradient[root]
                if top_active[contact_nodes].any():
                    top_reaction -= generalized_gradient
                if lower_active[contact_nodes].any():
                    lower_reaction += generalized_gradient
            penetration = float(max(upper_penetration.max(initial=0.0), lower_penetration.max(initial=0.0)))
        else:
            top_reaction = float(contact_penalty_N_mm * upper_penetration.sum())
            lower_contact_reaction = float(contact_penalty_N_mm * lower_penetration.sum())
            lower_constraint_reaction = float(
                np.abs(constraint_reactions[1::3]).sum()
            )
            lower_reaction = lower_contact_reaction + lower_constraint_reaction
            penetration = float(max(upper_penetration.max(initial=0.0), lower_penetration.max(initial=0.0)))
        return SolveResult(
            displacement=full,
            success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            potential_energy_Nmm=float(result.fun),
            gradient_norm=float(np.linalg.norm(projected_gradient, ord=np.inf)),
            top_reaction_N=top_reaction,
            lower_reaction_N=lower_reaction,
            contact_penetration_mm=penetration,
            constraint_reactions_N=constraint_reactions,
            element_results=self.element_results(full),
            optimizer_method=optimizer_method,
            primary_success=bool(primary.success),
            primary_message=str(primary.message),
            primary_iterations=int(primary.nit),
            fallback_used=fallback_used,
            fallback_success=fallback_success,
            fallback_message=fallback_message,
            fallback_iterations=fallback_iterations,
            fallback_gradient_norm=fallback_gradient_norm,
        )

    def element_results(self, displacement: np.ndarray) -> dict[str, np.ndarray]:
        elastic_modulus = float(self.material.elastic_modulus_MPa)
        area = float(self.section.area_mm2)
        inertia = float(self.section.inertia_mm4)
        half_thickness = 0.5 * float(self.section.thickness_mm)
        axial_strain = []
        curvature = []
        axial_force = []
        moment_1 = []
        moment_2 = []
        shear_strain = []
        shear_force = []
        stress_top_1 = []
        stress_bottom_1 = []
        stress_top_2 = []
        stress_bottom_2 = []
        for element_index, (node_1, node_2) in enumerate(self.connectivity):
            dof_1 = 3 * int(node_1)
            dof_2 = 3 * int(node_2)
            initial_length = self.initial_lengths[element_index]
            initial_angle = self.initial_angles[element_index]
            current_1 = self.nodes[node_1] + displacement[dof_1 : dof_1 + 2]
            current_2 = self.nodes[node_2] + displacement[dof_2 : dof_2 + 2]
            delta = current_2 - current_1
            current_length = np.linalg.norm(delta)
            current_angle = np.arctan2(delta[1], delta[0])
            rotations = np.array(
                [
                    displacement[dof_1 + 2] + initial_angle - current_angle,
                    displacement[dof_2 + 2] + initial_angle - current_angle,
                ]
            )
            strain = (current_length - initial_length) / initial_length
            kappa = (rotations[1] - rotations[0]) / initial_length
            xi = np.array([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459])
            weights = np.array([0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851])
            z = half_thickness * xi
            thickness_weights = half_thickness * weights
            quadrature_strain = strain - kappa * z
            outer_strain = np.array(
                [strain + kappa * half_thickness, strain - kappa * half_thickness]
            )

            def stress_law(values: np.ndarray) -> np.ndarray:
                if self.material.yield_stress_MPa is None:
                    return elastic_modulus * values
                yield_stress = float(self.material.yield_stress_MPa)
                hardening = float(self.material.hardening_ratio)
                scaled = elastic_modulus * values / yield_stress
                return (
                    hardening * elastic_modulus * values
                    + (1.0 - hardening)
                    * elastic_modulus
                    * values
                    / np.sqrt(1.0 + scaled**2)
                )

            quadrature_stress = stress_law(quadrature_strain)
            outer_stress = stress_law(outer_strain)
            force = float(self.section.width_mm) * float(
                np.sum(quadrature_stress * thickness_weights)
            )
            moment = -float(self.section.width_mm) * float(
                np.sum(quadrature_stress * z * thickness_weights)
            )
            moments = np.array([moment, moment])
            stresses = np.array(
                [outer_stress[0], outer_stress[1], outer_stress[0], outer_stress[1]]
            )
            gamma = 0.5 * float(rotations[0] + rotations[1])
            shear = (
                float(self.section.shear_correction)
                * float(self.material.shear_modulus_MPa)
                * area
                * gamma
            )
            axial_strain.append(strain)
            curvature.append(kappa)
            axial_force.append(force)
            moment_1.append(moments[0])
            moment_2.append(moments[1])
            shear_strain.append(gamma)
            shear_force.append(shear)
            stress_top_1.append(stresses[0])
            stress_bottom_1.append(stresses[1])
            stress_top_2.append(stresses[2])
            stress_bottom_2.append(stresses[3])
        return {
            "axial_strain": np.asarray(axial_strain),
            "curvature_1_mm": np.asarray(curvature),
            "axial_force_N": np.asarray(axial_force),
            "moment_node_1_Nmm": np.asarray(moment_1),
            "moment_node_2_Nmm": np.asarray(moment_2),
            "assumed_shear_strain": np.asarray(shear_strain),
            "shear_force_N": np.asarray(shear_force),
            "stress_top_node_1_MPa": np.asarray(stress_top_1),
            "stress_bottom_node_1_MPa": np.asarray(stress_bottom_1),
            "stress_top_node_2_MPa": np.asarray(stress_top_2),
            "stress_bottom_node_2_MPa": np.asarray(stress_bottom_2),
        }
