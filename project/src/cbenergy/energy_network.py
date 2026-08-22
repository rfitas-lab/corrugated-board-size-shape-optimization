"""A compact structured neural potential with energy--force consistency."""

from __future__ import annotations

from dataclasses import dataclass

import autograd.numpy as anp
from autograd import grad
import numpy as np
from scipy.optimize import minimize


def _softplus(x: anp.ndarray) -> anp.ndarray:
    return anp.maximum(x, 0.0) + anp.log1p(anp.exp(-anp.abs(x)))


@dataclass
class EnergyNetwork:
    parameters: np.ndarray
    input_size: int
    hidden_size: int
    # Generic response scales. In Paper A the caller supplies dimensionless
    # potential and its conjugate force; the historical attribute names are
    # retained for backward-compatible model serialization.
    energy_scale_Nmm: float
    reaction_scale_N: float
    strain_scale: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def _unpack(self, parameters: anp.ndarray | None = None):
        p = self.parameters if parameters is None else parameters
        n_in = self.input_size
        n_h = self.hidden_size
        cursor = 0
        w1 = anp.reshape(p[cursor : cursor + n_in * n_h], (n_in, n_h))
        cursor += n_in * n_h
        b1 = p[cursor : cursor + n_h]
        cursor += n_h
        w2 = anp.reshape(p[cursor : cursor + n_h * n_h], (n_h, n_h))
        cursor += n_h * n_h
        b2 = p[cursor : cursor + n_h]
        cursor += n_h
        w3 = anp.reshape(p[cursor : cursor + n_h], (n_h, 1))
        cursor += n_h
        b3 = p[cursor : cursor + 1]
        return w1, b1, w2, b2, w3, b3

    def normalized_energy(
        self, features: anp.ndarray, parameters: anp.ndarray | None = None
    ) -> anp.ndarray:
        standardized = (features[:, :-1] - anp.asarray(self.feature_mean)) / anp.asarray(
            self.feature_scale
        )
        features = anp.concatenate((standardized, features[:, -1:]), axis=1)
        w1, b1, w2, b2, w3, b3 = self._unpack(parameters)
        h1 = anp.tanh(anp.dot(features, w1) + b1)
        h2 = anp.tanh(anp.dot(h1, w2) + b2)
        coefficient = _softplus(anp.dot(h2, w3)[:, 0] + b3[0])
        normalized_strain = features[:, -1]
        return normalized_strain**2 * coefficient

    def energy_Nmm(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.normalized_energy(anp.asarray(features))) * self.energy_scale_Nmm

    def reaction_N(self, features: np.ndarray, height_mm: np.ndarray) -> np.ndarray:
        """Differentiate the learned potential with respect to platen travel."""

        features = np.asarray(features, dtype=float)
        height_mm = np.asarray(height_mm, dtype=float)
        values = np.zeros(len(features), dtype=float)
        for index, row in enumerate(features):
            fixed = anp.asarray(row[:-1])

            def scalar_energy(normalized_strain):
                complete = anp.concatenate((fixed, anp.reshape(normalized_strain, (1,))))
                return self.normalized_energy(anp.reshape(complete, (1, -1)))[0]

            derivative = grad(scalar_energy)(row[-1])
            values[index] = (
                self.energy_scale_Nmm
                * float(derivative)
                / (height_mm[index] * self.strain_scale)
            )
        return values

    def save(self, path) -> None:
        np.savez(
            path,
            parameters=self.parameters,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            energy_scale_Nmm=self.energy_scale_Nmm,
            reaction_scale_N=self.reaction_scale_N,
            strain_scale=self.strain_scale,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
        )

    @classmethod
    def load(cls, path) -> "EnergyNetwork":
        data = np.load(path)
        return cls(
            parameters=data["parameters"],
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            energy_scale_Nmm=float(data["energy_scale_Nmm"]),
            reaction_scale_N=float(data["reaction_scale_N"]),
            strain_scale=float(data["strain_scale"]),
            feature_mean=data["feature_mean"],
            feature_scale=data["feature_scale"],
        )


def _initial_parameters(input_size: int, hidden_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale_1 = np.sqrt(2.0 / (input_size + hidden_size))
    scale_h = np.sqrt(1.0 / hidden_size)
    return np.r_[
        rng.normal(0.0, scale_1, input_size * hidden_size),
        np.zeros(hidden_size),
        rng.normal(0.0, scale_h, hidden_size * hidden_size),
        np.zeros(hidden_size),
        rng.normal(0.0, scale_h, hidden_size),
        np.array([-1.0]),
    ]


def fit_energy_network(
    features: np.ndarray,
    energy_Nmm: np.ndarray,
    reaction_N: np.ndarray,
    height_mm: np.ndarray,
    case_ids: np.ndarray,
    *,
    hidden_size: int = 10,
    seed: int = 0,
    max_iterations: int = 1200,
) -> tuple[EnergyNetwork, dict[str, np.ndarray | float | str]]:
    """Fit energy and its displacement derivative on ordered load paths.

    The interval-secant term constrains energy increments to the average FEM
    reaction. The architecture enforces zero energy and zero force at zero
    compression through its quadratic strain prefactor.  This is supervised,
    energy-consistent regression; no differential-equation residual is used.
    """

    features = np.asarray(features, dtype=float)
    energy_Nmm = np.asarray(energy_Nmm, dtype=float)
    reaction_N = np.asarray(reaction_N, dtype=float)
    height_mm = np.asarray(height_mm, dtype=float)
    case_ids = np.asarray(case_ids)
    energy_scale = float(max(np.quantile(energy_Nmm, 0.95), 1e-6))
    reaction_scale = float(max(np.quantile(reaction_N, 0.95), 1e-6))
    target_energy = energy_Nmm / energy_scale
    pair_previous = []
    pair_current = []
    for case in np.unique(case_ids):
        indices = np.flatnonzero(case_ids == case)
        indices = indices[np.argsort(features[indices, -1])]
        pair_previous.extend(indices[:-1])
        pair_current.extend(indices[1:])
    previous = np.asarray(pair_previous, dtype=int)
    current = np.asarray(pair_current, dtype=int)
    delta_strain = (features[current, -1] - features[previous, -1]) * 0.25
    target_average_reaction = 0.5 * (reaction_N[current] + reaction_N[previous])
    path_height = height_mm[current]

    feature_mean = features[:, :-1].mean(axis=0)
    feature_scale = features[:, :-1].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    network = EnergyNetwork(
        parameters=_initial_parameters(features.shape[1], hidden_size, seed),
        input_size=features.shape[1],
        hidden_size=hidden_size,
        energy_scale_Nmm=energy_scale,
        reaction_scale_N=reaction_scale,
        strain_scale=0.25,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    features_ag = anp.asarray(features)
    target_energy_ag = anp.asarray(target_energy)

    def loss(parameters: anp.ndarray) -> anp.ndarray:
        predicted = network.normalized_energy(features_ag, parameters)
        energy_loss = anp.mean((predicted - target_energy_ag) ** 2)
        predicted_average_reaction = (
            energy_scale
            * (predicted[current] - predicted[previous])
            / (anp.asarray(path_height) * anp.asarray(delta_strain))
        )
        force_loss = anp.mean(
            ((predicted_average_reaction - anp.asarray(target_average_reaction)) / reaction_scale) ** 2
        )
        monotonic_penalty = anp.mean(anp.maximum(-predicted_average_reaction, 0.0) ** 2) / (
            reaction_scale**2
        )
        regularization = 1e-3 * anp.mean(parameters**2)
        return energy_loss + force_loss + 2.0 * monotonic_penalty + regularization

    gradient = grad(loss)
    result = minimize(
        fun=lambda p: float(loss(p)),
        x0=network.parameters,
        jac=lambda p: np.asarray(gradient(p), dtype=float),
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    network.parameters = np.asarray(result.x)
    predicted_energy = network.energy_Nmm(features)
    predicted_average_reaction = (
        (predicted_energy[current] - predicted_energy[previous])
        / (path_height * delta_strain)
    )
    diagnostics = {
        "success": str(result.success),
        "message": str(result.message),
        "iterations": float(result.nit),
        "final_loss": float(result.fun),
        "pair_current": current,
        "pair_previous": previous,
        "predicted_average_reaction_N": predicted_average_reaction,
        "target_average_reaction_N": target_average_reaction,
    }
    return network, diagnostics
