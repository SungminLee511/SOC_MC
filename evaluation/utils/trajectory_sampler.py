"""Trajectory sampling utility for evaluation."""

import torch
from evaluation.utils.checkpoint_loader import load_sampler_from_checkpoint


def sample_from_checkpoint(
    exp_id: str,
    n_samples: int = 10000,
    epoch: str = "final",
    result_dir: str = "result",
    device: str = "cpu",
    seed: int = 12345,
) -> torch.Tensor:
    """Sample terminal points from a trained checkpoint.

    Uses fixed eval seed for reproducibility.

    Returns:
        (n_samples, dim) tensor of terminal points
    """
    torch.manual_seed(seed)
    sampler, _ = load_sampler_from_checkpoint(exp_id, epoch, result_dir, device)

    with torch.no_grad():
        samples = sampler.sample(n_samples, device=torch.device(device))

    return samples.cpu()


def sample_trajectories_from_checkpoint(
    exp_id: str,
    n_samples: int = 10000,
    epoch: str = "final",
    result_dir: str = "result",
    device: str = "cpu",
    seed: int = 12345,
):
    """Sample full trajectories from a trained checkpoint.

    Returns:
        Trajectories object with states, timesteps, terminal
    """
    torch.manual_seed(seed)
    sampler, _ = load_sampler_from_checkpoint(exp_id, epoch, result_dir, device)

    with torch.no_grad():
        trajs = sampler.sample_trajectories(n_samples, device=torch.device(device))

    return trajs
