"""Abstract base class for energy functions (GMM benchmarks)."""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import numpy as np


class EnergyFunction(ABC):
    """Base class for all benchmark energy functions.

    Convention: p(x) ∝ exp(-E(x)), so E(x) = -log p(x) + const.
    """

    def __init__(self, config: dict):
        self._dim = config["dim"]
        self._K = config["K"]
        # Subclasses must set: _centers (K, dim), _covariances, _log_weights
        self._centers: torch.Tensor = None
        self._covariances: torch.Tensor = None  # (K, dim, dim) or (K, dim) for diag
        self._log_weights: torch.Tensor = None  # (K,)
        self._weights: torch.Tensor = None  # (K,)

    @property
    def K(self) -> int:
        return self._K

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mode_weights(self) -> torch.Tensor:
        """(K,) tensor of w_k."""
        return self._weights

    @property
    def mode_centers(self) -> torch.Tensor:
        """(K, dim) tensor of mode centers."""
        return self._centers

    def _to_device(self, device):
        """Move all tensors to device."""
        self._centers = self._centers.to(device)
        self._covariances = self._covariances.to(device)
        self._log_weights = self._log_weights.to(device)
        self._weights = self._weights.to(device)
        # Subclasses with extra tensors should override and call super

    def _ensure_device(self, x: torch.Tensor):
        if self._centers.device != x.device:
            self._to_device(x.device)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute E(x) for x of shape (batch, dim). Returns (batch,)."""
        self._ensure_device(x)
        return self._energy_impl(x)

    @abstractmethod
    def _energy_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Subclass implements energy computation here."""
        ...

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∇E(x) via autograd. Returns (batch, dim)."""
        with torch.enable_grad():
            x_ = x.clone().detach().requires_grad_(True)
            E = self.energy(x_)
            grad_E = torch.autograd.grad(E.sum(), x_, create_graph=False)[0]
        return grad_E

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score = -∇E(x) = ∇ log p(x)."""
        return -self.gradient(x)

    def restricted(self, subset_S: List[int]) -> 'EnergyFunction':
        """Return a new EnergyFunction for p_S (renormalized to subset).

        Args:
            subset_S: list of mode indices (0-indexed) to keep.
        """
        return RestrictedEnergyFunction(self, subset_S)

    def mode_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Return mode index k in {0, ..., K-1} for each point via nearest center.

        Args:
            x: (batch, dim)
        Returns:
            (batch,) tensor of int mode indices
        """
        self._ensure_device(x)
        dists = torch.cdist(x, self._centers)  # (batch, K)
        return dists.argmin(dim=1)

    def get_ref_samples(self, n: int = 10000) -> torch.Tensor:
        """Ancestral sampling from the mixture."""
        mode_idx = torch.multinomial(self._weights.cpu(), n, replacement=True)
        samples = self._sample_from_modes(mode_idx)
        return samples

    @abstractmethod
    def _sample_from_modes(self, mode_idx: torch.Tensor) -> torch.Tensor:
        """Sample points given assigned mode indices. Returns (n, dim)."""
        ...


class RestrictedEnergyFunction(EnergyFunction):
    """Wrapper: restricts a parent energy to a subset of modes with renormalized weights."""

    def __init__(self, parent: EnergyFunction, subset_S: List[int]):
        # Don't call super().__init__ with config; set attributes directly
        self._dim = parent.dim
        self._parent = parent
        self._subset = sorted(subset_S)
        self._K = len(self._subset)

        # Extract subset centers and covariances
        self._centers = parent._centers[self._subset].clone()
        self._covariances = parent._covariances[self._subset].clone()

        # Renormalize weights
        sub_weights = parent._weights[self._subset].clone()
        sub_weights = sub_weights / sub_weights.sum()
        self._weights = sub_weights
        self._log_weights = torch.log(sub_weights)

        # Copy any extra attributes needed for sampling
        self._parent_for_sampling = parent

    def _energy_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Energy using only subset modes."""
        # Delegate to parent's GMM formula but with subset tensors
        return _gmm_energy(x, self._centers, self._covariances, self._log_weights)

    def _sample_from_modes(self, mode_idx: torch.Tensor) -> torch.Tensor:
        """Map local mode indices to parent mode indices, then sample."""
        parent_idx = torch.tensor([self._subset[i] for i in mode_idx])
        return self._parent_for_sampling._sample_from_modes(parent_idx)

    def mode_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Mode assignment within the restricted subset (0-indexed within subset)."""
        self._ensure_device(x)
        dists = torch.cdist(x, self._centers)
        return dists.argmin(dim=1)

    def restricted(self, subset_S: List[int]) -> 'EnergyFunction':
        """Further restrict (indices relative to current subset)."""
        parent_indices = [self._subset[i] for i in subset_S]
        return RestrictedEnergyFunction(self._parent, parent_indices)


def _gmm_energy(
    x: torch.Tensor,
    centers: torch.Tensor,
    covariances: torch.Tensor,
    log_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute GMM energy E(x) = -log Σ_k w_k N(x; μ_k, Σ_k).

    Handles both diagonal and full covariance.

    Args:
        x: (B, D)
        centers: (K, D)
        covariances: (K, D) for diagonal, or (K, D, D) for full
        log_weights: (K,)

    Returns:
        (B,) energies
    """
    diff = x.unsqueeze(1) - centers.unsqueeze(0)  # (B, K, D)

    if covariances.ndim == 2:
        # Diagonal covariance: (K, D)
        var = covariances.unsqueeze(0)  # (1, K, D)
        log_det = covariances.log().sum(dim=-1)  # (K,)
        mahal = (diff ** 2 / var).sum(dim=-1)  # (B, K)
    else:
        # Full covariance: (K, D, D)
        K = centers.shape[0]
        # Compute precision and log-det per mode
        prec = torch.linalg.inv(covariances)  # (K, D, D)
        log_det = torch.linalg.slogdet(covariances)[1]  # (K,)
        # Mahalanobis: diff^T @ prec @ diff
        # diff: (B, K, D) -> (B, K, D, 1)
        mahal = torch.einsum('bkd,kde,bke->bk', diff, prec, diff)  # (B, K)

    log_probs = -0.5 * mahal - 0.5 * log_det + log_weights.unsqueeze(0)  # (B, K)
    return -torch.logsumexp(log_probs, dim=1)  # (B,)
