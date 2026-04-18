"""Time-conditioned MLP controller network.

Ported from Stein_ASBS/adjoint_samplers/components/model.py (FourierMLP + TimeEmbed).
"""

from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters of a module."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimeEmbed(nn.Module):
    """Fourier time embedding: sinusoidal features → 2-layer MLP."""

    def __init__(
        self,
        dim_out: int,
        activation: nn.Module = nn.SiLU(),
        num_layers: int = 2,
        channels: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.activation = activation
        self.register_buffer(
            "timestep_coeff",
            torch.linspace(start=0.1, end=100, steps=channels).unsqueeze(0),
            persistent=False,
        )
        self.timestep_phase = nn.Parameter(torch.randn(1, channels))
        self.hidden_layer = nn.ModuleList(
            [nn.Linear(2 * channels, channels)]
            + [nn.Linear(channels, channels) for _ in range(num_layers - 2)]
        )
        self.out_layer = nn.Linear(channels, dim_out)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, 1) -> (B, dim_out)"""
        t = t.view(-1, 1).float()
        sin_embed = torch.sin(self.timestep_coeff * t + self.timestep_phase)
        cos_embed = torch.cos(self.timestep_coeff * t + self.timestep_phase)
        embed = torch.cat([sin_embed, cos_embed], dim=1)
        for layer in self.hidden_layer:
            embed = self.activation(layer(embed))
        return self.out_layer(embed)


class FourierMLP(nn.Module):
    """Time-conditioned MLP with Fourier time embedding.

    u_theta(x, t) -> drift correction of shape (B, dim).

    Architecture: input_embed(x) + time_embed(t) -> hidden layers -> output.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Optional[list] = None,
        channels: int = 128,
        activation: str = "silu",
        time_embed_dim: int = 64,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.dim = dim

        act_fn = _get_activation(activation)

        # Time embedding
        self.time_embed = TimeEmbed(
            dim_out=channels,
            activation=act_fn,
            num_layers=2,
            channels=time_embed_dim,
        )

        # Input embedding
        self.input_embed = nn.Linear(dim, channels)

        # Hidden layers
        if hidden_dims is None:
            hidden_dims = [channels] * 4
        layers = []
        in_dim = channels
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.activation = act_fn

        # Output
        self.out_layer = nn.Linear(in_dim, dim)
        if zero_init_output:
            zero_module(self.out_layer)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: scalar or (B,) or (B,1) — time
            x: (B, dim) — spatial input
        Returns:
            (B, dim) — drift correction
        """
        t = t.view(-1, 1).expand(x.shape[0], 1).float()
        embed_t = self.time_embed(t)
        embed_x = self.input_embed(x)
        embed = embed_x + embed_t
        for layer in self.hidden_layers:
            embed = layer(self.activation(embed))
        return self.out_layer(self.activation(embed))


def _get_activation(name: str) -> nn.Module:
    activations = {
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    return activations[name]
