"""W5 Benchmark: Weight heterogeneity — 5 isotropic Gaussians with geometric decay weights."""

import torch
import numpy as np
from benchmark.base import EnergyFunction, _gmm_energy


class W5Energy(EnergyFunction):
    """5-mode 2D GMM on pentagon vertices with unequal weights.

    Weights: geometric decay w_k = r^(k-1) / Z, r=0.5
    -> (0.516, 0.258, 0.129, 0.065, 0.032)
    All modes isotropic with sigma=0.8.
    Pentagon radius=8.0, adjacent-mode distance ≈ 9.4.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        r = config.get("mode_weights", {}).get("r", 0.5)
        radius = config.get("mode_positions", {}).get("radius", 8.0)
        sigma = config.get("mode_covariance", {}).get("sigma", 0.8)

        # Pentagon centers
        angles = torch.linspace(0, 2 * np.pi, self._K + 1)[:-1]
        self._centers = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles),
        ], dim=1)  # (5, 2)

        # Geometric decay weights
        raw = torch.tensor([r ** k for k in range(self._K)], dtype=torch.float32)
        self._weights = raw / raw.sum()
        self._log_weights = torch.log(self._weights)

        # Isotropic covariance (diagonal representation)
        self._covariances = torch.full((self._K, self._dim), sigma ** 2)

        self._sigma = sigma

    def _energy_impl(self, x: torch.Tensor) -> torch.Tensor:
        return _gmm_energy(x, self._centers, self._covariances, self._log_weights)

    def _sample_from_modes(self, mode_idx: torch.Tensor) -> torch.Tensor:
        n = mode_idx.shape[0]
        centers = self._centers.cpu()[mode_idx]
        return centers + self._sigma * torch.randn(n, self._dim)
