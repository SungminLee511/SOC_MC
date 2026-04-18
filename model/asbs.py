"""Adjoint Schrödinger Bridge Sampler (ASBS) implementation.

Ported from Stein_ASBS CorrectorMatcher + AdjointVEMatcher + CorrectorGradTermCost.

ASBS alternates two matching steps per outer iteration:
  1. Corrector matching: learn h(x1) ≈ ∇log p_1^base(x1|x0) (bridge score)
  2. Adjoint matching: learn u_θ(x,t) with target = -(∇E(x1) + h(x1))

The corrector replaces the analytic base-score term in AS.
"""

import torch
import torch.nn as nn

from model.base import CategoryASampler, Trajectories
from model.networks import FourierMLP
from model.sde import sdeint


class ASBS(CategoryASampler):
    """Adjoint Schrödinger Bridge Sampler for VE-SDE.

    Adds a corrector network that learns the bridge score,
    enabling iterative refinement (IPF-style).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        corr_cfg = config.get("corrector", {})

        # Corrector network: learns ∇log p_1(x1 | x0) under current policy
        self.corrector = FourierMLP(
            dim=self.dim,
            hidden_dims=corr_cfg.get("hidden_dims", [128, 128, 128, 128]),
            channels=corr_cfg.get("channels", 128),
            activation=corr_cfg.get("activation", "silu"),
            time_embed_dim=corr_cfg.get("time_embed_dim", 64),
        )

        # Source variance for base score
        sigma_min = self.ref_sde.sigma_min
        sigma_max = self.ref_sde.sigma_max
        self.source_var = 1.0
        self.var1 = self.source_var + sigma_max ** 2 * (1 - (sigma_max / sigma_min) ** (-2))

        # Track IPF stage
        self._is_init_stage = True

    def set_init_stage(self, is_init: bool):
        """Toggle between init stage (zero corrector) and iterative stage."""
        self._is_init_stage = is_init

    def _compute_adjoint1(self, x1: torch.Tensor, energy_fn) -> torch.Tensor:
        """Compute adjoint at t=1.

        Init stage: ∇E(x1) only (zero corrector, like first IPF step).
        Iterative stage: ∇E(x1) + h(x1) where h is learned corrector.
        """
        grad_E = energy_fn.gradient(x1)

        if self._is_init_stage:
            return grad_E
        else:
            t1 = torch.ones(x1.shape[0], 1, device=x1.device)
            with torch.no_grad():
                corr = self.corrector(t1, x1)
            return grad_E + corr

    def corrector_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Corrector matching: regress corrector(t1, x1) → cond_score(x0, t1, x1).

        Bridge score: ∇log p_{1|0}(x1|x0) = (mu1 - x1) / var1
        For VE-SDE from x0: mu1 = x0, var1 = ∫g²ds
        Actually uses ref_sde.sample_posterior logic.
        """
        B = x0.shape[0]
        device = x0.device

        t1 = torch.ones(B, 1, device=device)

        # Conditional score: ∇log p_{t=1|0}(x1 | x0) = (x0 - x1) / var_total
        # For VE-SDE: var_total = sigma_max² - sigma_min² (total variance added)
        total_var = self.ref_sde.total_var
        cond_score = (x0 - x1) / total_var  # (B, D)

        # Corrector prediction
        corr_pred = self.corrector(t1, x1)

        return ((corr_pred - cond_score) ** 2).mean()

    def adjoint_loss(self, x0: torch.Tensor, x1: torch.Tensor, energy_fn) -> torch.Tensor:
        """Adjoint matching: regress u_θ(x_t, t) → -adjoint1.

        Same as AS but uses corrector instead of analytic base score.
        """
        B, D = x0.shape
        device = x0.device

        adjoint1 = self._compute_adjoint1(x1, energy_fn)
        target = -adjoint1

        # Random time, bridge posterior
        t = torch.rand(B, 1, device=device)
        x_t = self.ref_sde.sample_posterior(t, x0, x1)

        u_pred = self.controller(t, x_t)
        return ((u_pred - target) ** 2).mean()

    def train_step(
        self,
        batch_size: int,
        energy_fn,
        device: torch.device,
        step_type: str = "adjoint",
    ) -> torch.Tensor:
        """Single training step.

        Args:
            step_type: "corrector" or "adjoint"
        """
        x0 = torch.randn(batch_size, self.dim, device=device)
        timesteps = self._get_timesteps(device)

        if step_type == "corrector":
            # Corrector step: use ref_sde in init stage, controlled_sde otherwise
            sde = self.ref_sde if self._is_init_stage else self.controlled_sde
            x_final = sdeint(sde, x0, timesteps, return_all=False)
            return self.corrector_loss(x0, x_final)
        else:
            # Adjoint step
            x_final = sdeint(self.controlled_sde, x0, timesteps, return_all=False)
            return self.adjoint_loss(x0, x_final, energy_fn)

    # --- Required ABC methods ---

    def compute_target(self, trajectories: Trajectories, energy_fn):
        """Compute target for base class loss() method."""
        x1 = trajectories.terminal
        adjoint1 = self._compute_adjoint1(x1, energy_fn)
        target = -adjoint1
        T = len(trajectories.timesteps) - 1
        return [target] * T

    def parameters(self):
        """Return all trainable parameters (controller + corrector)."""
        return list(self.controller.parameters()) + list(self.corrector.parameters())

    def state_dict(self) -> dict:
        return {
            "controller": self.controller.state_dict(),
            "corrector": self.corrector.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.controller.load_state_dict(state["controller"])
        self.corrector.load_state_dict(state["corrector"])
