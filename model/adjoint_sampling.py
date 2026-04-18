"""Adjoint Sampling (AS) implementation.

Ported from Stein_ASBS AdjointVEMatcher + ScoreGradTermCost.

For VE-SDE with no state cost, the adjoint is constant in time:
  adjoint(t) = adjoint(1) = ∇E(X_1) + ∇log p^base_1(X_1)

Target: u*_target = -adjoint
Training: sample (x0, x1) from controlled SDE, sample random t,
  interpolate via bridge posterior, regress u_θ(x_t, t) → -adjoint.
"""

import torch
import torch.nn as nn

from model.base import CategoryASampler, Trajectories
from model.sde import sdeint


class AdjointSampling(CategoryASampler):
    """Adjoint Sampling for VE-SDE.

    Efficient boundary-only computation: adjoint is time-constant
    when base SDE has no drift and state cost is zero.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Source distribution parameters for score computation
        # Source: N(0, I) -> after VE-SDE at t=1: N(0, sigma_max^2 - sigma_min^2 + 1)
        # p^base_1 score: (mu1 - x) / var1
        sigma_min = self.ref_sde.sigma_min
        sigma_max = self.ref_sde.sigma_max
        # Source is N(0, I), so mu0=0, var0=1
        # At t=1: var1 = var0 + ∫_0^1 g²(s) ds = 1 + sigma_max² - sigma_min²
        self.source_var = 1.0  # isotropic N(0,I) source
        self.var1 = self.source_var + sigma_max ** 2 * (1 - (sigma_max / sigma_min) ** (-2))

    def _compute_adjoint1(self, x1: torch.Tensor, energy_fn) -> torch.Tensor:
        """Compute adjoint at t=1: ∇E(X_1) + ∇log p^base_1(X_1).

        For VE-SDE from N(0,I):
          ∇log p^base_1(x) = (0 - x) / var1 = -x / var1
        """
        grad_E = energy_fn.gradient(x1)  # ∇E(x1)
        score_base = -x1 / self.var1     # ∇log p^base_1(x1)
        return grad_E + score_base

    def compute_loss_from_boundaries(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        energy_fn,
        n_loss_samples: int = 1,
    ) -> torch.Tensor:
        """Efficient loss computation using boundary samples only.

        Sample random t, interpolate x_t via bridge posterior,
        regress u_θ(x_t, t) → -adjoint1.
        """
        B, D = x0.shape
        device = x0.device

        adjoint1 = self._compute_adjoint1(x1, energy_fn)
        target = -adjoint1  # (B, D)

        total_loss = 0.0
        for _ in range(n_loss_samples):
            # Random time
            t = torch.rand(B, 1, device=device)

            # Bridge posterior: sample x_t ~ p_t(x | x0, x1)
            x_t = self.ref_sde.sample_posterior(t, x0, x1)

            # Controller prediction
            u_pred = self.controller(t, x_t)

            # L2 loss
            loss = ((u_pred - target) ** 2).mean()
            total_loss = total_loss + loss

        return total_loss / n_loss_samples

    def train_step(self, batch_size: int, energy_fn, device: torch.device) -> torch.Tensor:
        """Single training step: sample trajectories, compute loss.

        Uses efficient boundary-only sampling for VE-SDE.
        """
        # Sample initial points
        x0 = torch.randn(batch_size, self.dim, device=device)
        timesteps = self._get_timesteps(device)

        # Forward SDE to get boundary points only
        x_final = sdeint(self.controlled_sde, x0, timesteps, return_all=False)

        # Compute loss using boundary samples
        return self.compute_loss_from_boundaries(x0, x_final, energy_fn)

    # --- Required ABC methods (less efficient, for evaluation/compatibility) ---

    def compute_target(self, trajectories: Trajectories, energy_fn):
        """Compute target as list of per-timestep targets.

        For VE-SDE: target is constant = -adjoint1 at all timesteps.
        """
        x1 = trajectories.terminal
        adjoint1 = self._compute_adjoint1(x1, energy_fn)
        target = -adjoint1  # (B, D)

        # Replicate across timesteps
        T = len(trajectories.timesteps) - 1
        return [target] * T  # list of T copies of (B, D)
