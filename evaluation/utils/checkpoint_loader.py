"""Checkpoint loading utility."""

import os
import glob
import torch
import yaml

from model.adjoint_sampling import AdjointSampling
from model.asbs import ASBS


SAMPLER_REGISTRY = {
    "adjoint_sampling": AdjointSampling,
    "as": AdjointSampling,
    "asbs": ASBS,
}


def load_checkpoint(exp_id: str, epoch: str = "final", result_dir: str = "result"):
    """Load a checkpoint and its config by experiment ID.

    Args:
        exp_id: e.g. 'goal1_as_w5_S01_pretrain_seed0'
        epoch: 'final' (last checkpoint) or int epoch number
        result_dir: base result directory

    Returns:
        (state_dict, config_dict)
    """
    ckpt_dir = os.path.join(result_dir, "checkpoints", exp_id)

    # Load config
    config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Find checkpoint file
    if epoch == "final":
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        ckpt_path = ckpt_files[-1]
    else:
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return state_dict, config


def load_sampler_from_checkpoint(exp_id: str, epoch: str = "final", result_dir: str = "result", device: str = "cpu"):
    """Load a fully instantiated sampler from checkpoint.

    Returns:
        (sampler, exp_config)
    """
    state_dict, exp_config = load_checkpoint(exp_id, epoch, result_dir)

    model_config_path = exp_config["model"]["config"]
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)

    sampler_name = model_cfg["sampler"]
    cls = SAMPLER_REGISTRY[sampler_name]
    sampler = cls(model_cfg)
    sampler.load_state_dict(state_dict)
    sampler = sampler.to(device)
    sampler.eval()

    return sampler, exp_config
