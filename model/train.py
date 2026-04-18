"""Shared training script for all category (a) samplers.

Usage:
    python model/train.py --exp_config result/experiments/<config>.yaml
"""

import argparse
import csv
import os
import random
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml


# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.adjoint_sampling import AdjointSampling
from model.asbs import ASBS


# ── Sampler registry ──

SAMPLER_REGISTRY = {
    "adjoint_sampling": AdjointSampling,
    "as": AdjointSampling,
    "asbs": ASBS,
}

# ── Benchmark registry ──

BENCHMARK_REGISTRY = {
    "W5": ("benchmark.w5", "W5Energy"),
    "C5": ("benchmark.c5", "C5Energy"),
    "B5": ("benchmark.b5", "B5Energy"),
}


def load_benchmark(bench_config_path: str):
    """Load benchmark energy function from config."""
    with open(bench_config_path) as f:
        bench_cfg = yaml.safe_load(f)
    name = bench_cfg["benchmark"]
    mod_name, cls_name = BENCHMARK_REGISTRY[name]
    import importlib
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(bench_cfg)


def load_sampler(model_config_path: str):
    """Load sampler from model config."""
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    sampler_name = model_cfg["sampler"]
    cls = SAMPLER_REGISTRY[sampler_name]
    return cls(model_cfg), model_cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_mode_weights(sampler, energy_fn, n_samples: int, device: torch.device):
    """Sample and estimate mode weights α_k. Returns (K,) array + MWD."""
    sampler.eval()
    with torch.no_grad():
        samples = sampler.sample(n_samples, device=device)
    assignments = energy_fn.mode_assignment(samples)
    K = energy_fn.K
    counts = torch.zeros(K)
    for k in range(K):
        counts[k] = (assignments == k).sum().float()
    alpha = counts / counts.sum()

    # MWD = max_k |alpha_k - w_k|
    target_weights = energy_fn.mode_weights.cpu()
    mwd = (alpha - target_weights).abs().max().item()

    return alpha.tolist(), mwd


class CSVLogger:
    """Simple CSV logger matching the spec."""

    def __init__(self, log_path: str, K: int = 5):
        self.log_path = log_path
        self.K = K
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, "w", newline="")
        alpha_cols = [f"alpha_{k+1}" for k in range(K)]
        self.writer = csv.writer(self.file)
        self.writer.writerow(["epoch", "event_type", "loss"] + alpha_cols + ["mwd", "notes"])
        self.file.flush()

    def log_train(self, epoch: int, loss: float, notes: str = ""):
        row = [epoch, "train", f"{loss:.6f}"] + [""] * self.K + ["", notes]
        self.writer.writerow(row)
        self.file.flush()

    def log_eval(self, epoch: int, alphas: list, mwd: float, notes: str = ""):
        row = [epoch, "eval", ""] + [f"{a:.4f}" for a in alphas] + [f"{mwd:.4f}", notes]
        self.writer.writerow(row)
        self.file.flush()

    def log_checkpoint(self, epoch: int, notes: str = ""):
        row = [epoch, "checkpoint", ""] + [""] * self.K + ["", notes]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def build_optimizer(sampler, model_cfg: dict):
    """Build optimizer from model config."""
    opt_cfg = model_cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-4)

    if name == "adam":
        return torch.optim.Adam(sampler.parameters(), lr=lr)
    elif name == "sgd":
        momentum = opt_cfg.get("momentum", 0.0)
        return torch.optim.SGD(sampler.parameters(), lr=lr, momentum=momentum)
    elif name == "adamw":
        return torch.optim.AdamW(sampler.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train(exp_config_path: str):
    """Main training entry point."""
    with open(exp_config_path) as f:
        exp_cfg = yaml.safe_load(f)

    # ── Setup ──
    seed = exp_cfg.get("seed", 0)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = exp_cfg.get("n_epochs", 500)

    # ── Load model and benchmark ──
    model_config_path = exp_cfg["model"]["config"]
    bench_config_path = exp_cfg["benchmark"]["config"]

    sampler, model_cfg = load_sampler(model_config_path)
    energy_fn = load_benchmark(bench_config_path)

    # ── Apply subset restriction if specified ──
    subset_cfg = exp_cfg.get("subset", {})
    subset_S = subset_cfg.get("S", None)
    full_target = subset_cfg.get("full_target", True)

    # For training: use restricted energy if subset specified and not full_target
    if subset_S is not None and not full_target:
        train_energy = energy_fn.restricted(subset_S)
    else:
        train_energy = energy_fn

    # For evaluation: always use full energy
    eval_energy = energy_fn

    # ── Load checkpoint if specified ──
    init_cfg = exp_cfg.get("init", {})
    from_ckpt = init_cfg.get("from_checkpoint", None)
    if from_ckpt and os.path.exists(from_ckpt):
        ckpt = torch.load(from_ckpt, map_location=device, weights_only=True)
        sampler.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {from_ckpt}")

        # Add initialization noise if specified
        noise_level = init_cfg.get("noise_level", 0.0)
        if noise_level > 0:
            with torch.no_grad():
                for p in sampler.parameters():
                    p.add_(noise_level * torch.randn_like(p))
            print(f"Added noise: level={noise_level}")

    sampler = sampler.to(device)

    # ── Optimizer ──
    optimizer = build_optimizer(sampler, model_cfg)

    # Override LR if experiment config specifies it
    exp_lr = exp_cfg.get("optimizer", {}).get("lr", None)
    if exp_lr is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = exp_lr

    # ── Logging and checkpointing ──
    log_cfg = exp_cfg.get("logging", {})
    checkpoint_every = log_cfg.get("checkpoint_every", 100)
    eval_every = log_cfg.get("eval_every", 0)  # 0 = no eval during training
    n_eval_samples = log_cfg.get("n_eval_samples", 10000)

    output_cfg = exp_cfg.get("output", {})
    ckpt_dir = output_cfg.get("checkpoint_dir", "result/checkpoints/default/")
    log_file = output_cfg.get("log_file", "result/logs/default.csv")

    os.makedirs(ckpt_dir, exist_ok=True)
    logger = CSVLogger(log_file, K=eval_energy.K)

    # Save experiment config copy
    shutil.copy2(exp_config_path, os.path.join(ckpt_dir, "config.yaml"))

    # ── Training config ──
    train_cfg = model_cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 256)
    clip_grad = train_cfg.get("clip_grad_norm", None)

    # ASBS-specific
    is_asbs = isinstance(sampler, ASBS)
    if is_asbs:
        corr_steps = train_cfg.get("corrector_steps_per_epoch", 1)
        adj_steps = train_cfg.get("adjoint_steps_per_epoch", 1)
        ipf_init_epochs = train_cfg.get("ipf_init_epochs", 50)

    # ── Training loop ──
    print(f"Training: {n_epochs} epochs, batch_size={batch_size}, device={device}")
    print(f"Sampler: {sampler.__class__.__name__}, Benchmark: {train_energy.__class__.__name__}")
    if subset_S is not None:
        print(f"Subset S={subset_S}, full_target={full_target}")

    for epoch in range(n_epochs):
        sampler.train()

        if is_asbs:
            # ASBS: alternate corrector and adjoint steps
            sampler.set_init_stage(epoch < ipf_init_epochs)

            epoch_loss = 0.0
            for _ in range(corr_steps):
                optimizer.zero_grad()
                closs = sampler.train_step(batch_size, train_energy, device, step_type="corrector")
                closs.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(sampler.parameters(), clip_grad)
                optimizer.step()

            for _ in range(adj_steps):
                optimizer.zero_grad()
                aloss = sampler.train_step(batch_size, train_energy, device, step_type="adjoint")
                aloss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(sampler.parameters(), clip_grad)
                optimizer.step()
                epoch_loss = aloss.item()
        else:
            # AS: single train step
            optimizer.zero_grad()
            loss = sampler.train_step(batch_size, train_energy, device)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(sampler.parameters(), clip_grad)
            optimizer.step()
            epoch_loss = loss.item()

        # Log training loss
        notes = "start" if epoch == 0 else ""
        logger.log_train(epoch, epoch_loss, notes)

        # Periodic evaluation
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            alphas, mwd = evaluate_mode_weights(sampler, eval_energy, n_eval_samples, device)
            logger.log_eval(epoch, alphas, mwd)
            print(f"  Epoch {epoch}: loss={epoch_loss:.4f}, MWD={mwd:.4f}, α={[f'{a:.3f}' for a in alphas]}")

        # Periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0 or epoch == n_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save(sampler.state_dict(), ckpt_path)
            logger.log_checkpoint(epoch, f"saved {ckpt_path}")

        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}")

    logger.close()
    print(f"Done. Checkpoints: {ckpt_dir}, Logs: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()
    train(args.exp_config)
