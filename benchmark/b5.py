"""B5 Benchmark: Barrier heterogeneity — cluster(1-3) near + isolated(4,5) far."""

import torch
from benchmark.base import EnergyFunction, _gmm_energy


class B5Energy(EnergyFunction):
    """5-mode 2D GMM with barrier heterogeneity.

    Cluster modes 1,2,3 (idx 0,1,2): positions (-3,0), (0,0), (3,0)
        pairwise sep=3.0, rho_sep_near ≈ 1.5
    Isolated modes 4,5 (idx 3,4): positions (0,12), (0,-12)
        distance to cluster ≈ 12, rho_sep_far ≈ 6.0

    Equal weights, isotropic sigma=0.8.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        sigma = config.get("mode_covariance", {}).get("sigma", 0.8)

        # Fixed positions per spec
        positions = config.get("mode_positions", {})
        cluster = positions.get("cluster", [[-3, 0], [0, 0], [3, 0]])
        isolated = positions.get("isolated", [[0, 12], [0, -12]])
        all_pos = cluster + isolated

        self._centers = torch.tensor(all_pos, dtype=torch.float32)  # (5, 2)

        # Equal weights
        self._weights = torch.ones(self._K) / self._K
        self._log_weights = torch.log(self._weights)

        # Isotropic covariance
        self._covariances = torch.full((self._K, self._dim), sigma ** 2)

        self._sigma = sigma

    def _energy_impl(self, x: torch.Tensor) -> torch.Tensor:
        return _gmm_energy(x, self._centers, self._covariances, self._log_weights)

    def _sample_from_modes(self, mode_idx: torch.Tensor) -> torch.Tensor:
        n = mode_idx.shape[0]
        centers = self._centers.cpu()[mode_idx]
        return centers + self._sigma * torch.randn(n, self._dim)
