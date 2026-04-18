"""C5 Benchmark: Covariance heterogeneity — 5 Gaussians with mixed covariance structures."""

import torch
import numpy as np
from benchmark.base import EnergyFunction, _gmm_energy


class C5Energy(EnergyFunction):
    """5-mode 2D GMM on pentagon vertices with equal weights, heterogeneous covariances.

    Modes 1,2,3 (idx 0,1,2): wide isotropic, sigma_wide=0.8
    Mode 4 (idx 3): tight isotropic, sigma_tight=0.15
    Mode 5 (idx 4): anisotropic, diag(sigma_wide^2, sigma_tight^2)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        radius = config.get("mode_positions", {}).get("radius", 8.0)
        sigma_wide = config.get("mode_covariance", {}).get("sigma_wide", 0.8)
        sigma_tight = config.get("mode_covariance", {}).get("sigma_tight", 0.15)

        # Pentagon centers
        angles = torch.linspace(0, 2 * np.pi, self._K + 1)[:-1]
        self._centers = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles),
        ], dim=1)  # (5, 2)

        # Equal weights
        self._weights = torch.ones(self._K) / self._K
        self._log_weights = torch.log(self._weights)

        # Heterogeneous diagonal covariances: (K, D)
        self._covariances = torch.zeros(self._K, self._dim)
        # Modes 0,1,2: wide isotropic
        for k in range(3):
            self._covariances[k] = sigma_wide ** 2
        # Mode 3: tight isotropic
        self._covariances[3] = sigma_tight ** 2
        # Mode 4: anisotropic
        self._covariances[4, 0] = sigma_wide ** 2
        self._covariances[4, 1] = sigma_tight ** 2

        self._sigma_wide = sigma_wide
        self._sigma_tight = sigma_tight

    def _energy_impl(self, x: torch.Tensor) -> torch.Tensor:
        return _gmm_energy(x, self._centers, self._covariances, self._log_weights)

    def _sample_from_modes(self, mode_idx: torch.Tensor) -> torch.Tensor:
        n = mode_idx.shape[0]
        centers = self._centers.cpu()[mode_idx]
        # Per-mode std from diagonal covariance
        stds = torch.sqrt(self._covariances.cpu()[mode_idx])  # (n, dim)
        return centers + stds * torch.randn(n, self._dim)
