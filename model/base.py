"""Abstract base class for Category (a) SOC samplers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch
import torch.nn as nn
import numpy as np

from model.networks import FourierMLP
from model.sde import VESDE, ControlledSDE, sdeint


class Trajectories:
    """Container for sampled trajectories and intermediate data."""

    def __init__(
        self,
        states: list[torch.Tensor],
        timesteps: torch.Tensor,
        terminal: torch.Tensor,
    ):
        self.states = states       # list of (B, D) at each timestep
        self.timesteps = timesteps  # (T,)
        self.terminal = terminal    # (B, D) = states[-1]


class CategoryASampler(ABC, nn.Module):
    """Base class for category (a) SOC samplers.

    All category (a) samplers share:
    - VE-SDE diffusion schedule
    - FourierMLP controller u_theta(x, t)
    - L2 residual loss between controller and target
    - Forward SDE sampling via Euler-Maruyama
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("controller", {})
        diff_cfg = config.get("diffusion", {})

        self.dim = config.get("dim", 2)
        self.n_timesteps = config.get("n_timesteps", 100)

        # Diffusion schedule
        self.ref_sde = VESDE(
            sigma_min=diff_cfg.get("sigma_min", 0.01),
            sigma_max=diff_cfg.get("sigma_max", 3.0),
        )

        # Controller network
        self.controller = FourierMLP(
            dim=self.dim,
            hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
            channels=model_cfg.get("channels", 128),
            activation=model_cfg.get("activation", "silu"),
            time_embed_dim=model_cfg.get("time_embed_dim", 64),
        )

        # Controlled SDE
        self.controlled_sde = ControlledSDE(self.ref_sde, self.controller)

    def _get_timesteps(self, device: torch.device) -> torch.Tensor:
        """Uniform time grid [0, 1] with n_timesteps points."""
        return torch.linspace(0, 1, self.n_timesteps, device=device)

    @torch.no_grad()
    def sample_trajectories(self, batch_size: int, device: torch.device = None) -> Trajectories:
        """Generate trajectories from current policy Q_theta.

        Samples X_0 ~ N(0, I), integrates forward to X_1.
        """
        if device is None:
            device = next(self.parameters()).device

        x0 = torch.randn(batch_size, self.dim, device=device)
        timesteps = self._get_timesteps(device)

        states = sdeint(self.controlled_sde, x0, timesteps, return_all=True)

        return Trajectories(
            states=states,
            timesteps=timesteps,
            terminal=states[-1],
        )

    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Sample terminal points only. Returns (B, D)."""
        if device is None:
            device = next(self.parameters()).device
        x0 = torch.randn(batch_size, self.dim, device=device)
        timesteps = self._get_timesteps(device)
        return sdeint(self.controlled_sde, x0, timesteps, return_all=False)

    @abstractmethod
    def compute_target(self, trajectories: Trajectories, energy_fn) -> torch.Tensor:
        """Compute u*_target along trajectories using current parameters.

        Returns target drift values for the loss computation.
        Subclasses implement this with their specific target formulation.
        """
        ...

    def loss(self, trajectories: Trajectories, energy_fn) -> torch.Tensor:
        """Compute L2 residual loss between controller and target.

        L = E[∫_0^1 |u_θ(X_t, t) - u*_target(X_t, t)|² dt]
        """
        target = self.compute_target(trajectories, energy_fn)
        # target shape: (B, T-1, D) or computed per-step

        timesteps = trajectories.timesteps
        total_loss = torch.tensor(0.0, device=timesteps.device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - t
            x_t = trajectories.states[i]

            # Controller output at this step
            u_theta = self.controller(t, x_t)

            # Target at this step (subclass provides)
            u_target = target[i] if isinstance(target, list) else target[:, i, :]

            # L2 residual, time-weighted
            residual = (u_theta - u_target).pow(2).sum(dim=-1).mean()
            total_loss = total_loss + residual * dt

        return total_loss

    def parameters(self) -> Iterable[torch.Tensor]:
        """Return trainable parameters (controller only)."""
        return self.controller.parameters()

    def state_dict(self) -> dict:
        return self.controller.state_dict()

    def load_state_dict(self, state: dict):
        self.controller.load_state_dict(state)
