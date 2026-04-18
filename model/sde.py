"""SDE definitions and Euler-Maruyama integration.

Ported from Stein_ASBS/adjoint_samplers/components/sde.py.
Only VE-SDE + ControlledSDE + sdeint needed for 2D benchmarks.
"""

from __future__ import annotations
from typing import List

import numpy as np
import torch
import torch.nn as nn


class VESDE(nn.Module):
    """Variance Exploding SDE: dX_t = g(t) dW_t

    g(t) = sigma_min * (sigma_max/sigma_min)^(1-t) * sqrt(2*log(sigma_max/sigma_min))

    Convention: t ∈ [0, 1], X_0 ~ source (e.g. N(0, I)), X_1 ~ target.
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 3.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max / sigma_min
        self.total_var = sigma_max ** 2 - sigma_min ** 2

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def diff(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) diffusion coefficient."""
        return (
            self.sigma_min
            * (self.sigma_diff ** (1 - t))
            * ((2 * np.log(self.sigma_diff)) ** 0.5)
        )

    def diffsquare_integral(self, t: torch.Tensor) -> torch.Tensor:
        """∫_0^t g²(s) ds"""
        return self.sigma_max ** 2 * (1 - self.sigma_diff ** (-2 * t))

    def sample_posterior(self, t, x0, x1, z=None):
        """Sample from bridge posterior p_t(x | x0, x1).

        Args:
            t: (B, 1), x0: (B, D), x1: (B, D)
        Returns:
            xt: (B, D)
        """
        B, D = x0.shape
        t_reparam = self.diffsquare_integral(t) / self.total_var
        if z is None:
            z = torch.randn_like(x0)
        mean = (1 - t_reparam) * x0 + t_reparam * x1
        coeff = self.total_var * t_reparam * (1 - t_reparam)
        coeff = coeff.clamp(min=0)
        return mean + torch.sqrt(coeff) * z


class ControlledSDE(nn.Module):
    """Controlled SDE: dX_t = (b(t,x) + g(t)² u_θ(t,x)) dt + g(t) dW_t"""

    def __init__(self, ref_sde: VESDE, controller: nn.Module):
        super().__init__()
        self.ref_sde = ref_sde
        self.controller = controller

    def diff(self, t: torch.Tensor) -> torch.Tensor:
        return self.ref_sde.diff(t)

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = self.diff(t)
        return self.ref_sde.drift(t, x) + (g ** 2) * self.controller(t, x)

    def sample_posterior(self, t, x0, x1, z=None):
        return self.ref_sde.sample_posterior(t, x0, x1, z=z)


@torch.no_grad()
def sdeint(
    sde: nn.Module,
    state0: torch.Tensor,
    timesteps: torch.Tensor,
    return_all: bool = False,
) -> torch.Tensor | List[torch.Tensor]:
    """Forward Euler-Maruyama integration.

    Args:
        sde: must have .drift(t, x) and .diff(t)
        state0: (B, D) initial state
        timesteps: (T,) time grid
        return_all: if True, return list of all states; else just final state

    Returns:
        Final state (B, D) or list of T states.
    """
    sde.eval()
    state = state0.clone()
    states = [state0] if return_all else None

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        drift = sde.drift(t, state) * dt
        noise = sde.diff(t) * dt.sqrt() * torch.randn_like(state)
        state = state + drift + noise

        if return_all:
            states.append(state)

    return states if return_all else state
