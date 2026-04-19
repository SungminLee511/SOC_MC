#!/usr/bin/env python3
"""Generate Goal 5 ablation configs.

Ablation axes:
  5.1 Optimizer: sgd, sgd_momentum (0.9), adam (baseline)
  5.2 Schedule: constant (baseline), cosine, warmup
  5.3 Batch size: 64, 256 (baseline), 1024

For each (checkpoint, ablation), use eta_mult=2.0 (at theoretical boundary),
5 seeds. Only non-baseline settings generate new configs.
"""

import os, csv, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "result" / "experiments"

# Read checkpoint selection
ckpts = []
with open(ROOT / "evaluation" / "tables" / "goal5_checkpoint_selection.csv") as f:
    for row in csv.DictReader(f):
        ckpts.append(row)

# Ablation definitions: (tag, overrides)
# Baseline is adam + constant + bs256 — skip those
ablations = [
    # Optimizer ablations (fixed eta_mult=2.0, constant schedule, bs=256)
    ("opt_sgd", {"optimizer": {"name": "sgd"}, "training": {"schedule": "constant", "batch_size": 256}}),
    ("opt_sgd_mom", {"optimizer": {"name": "sgd", "momentum": 0.9}, "training": {"schedule": "constant", "batch_size": 256}}),
    # Schedule ablations (adam, fixed eta_mult=2.0, bs=256)
    ("sched_cosine", {"optimizer": {"name": "adam"}, "training": {"schedule": "cosine", "batch_size": 256}}),
    ("sched_warmup", {"optimizer": {"name": "adam"}, "training": {"schedule": "warmup", "batch_size": 256}}),
    # Batch size ablations (adam, constant, fixed eta_mult=2.0)
    ("bs64", {"optimizer": {"name": "adam"}, "training": {"schedule": "constant", "batch_size": 64}}),
    ("bs1024", {"optimizer": {"name": "adam"}, "training": {"schedule": "constant", "batch_size": 1024}}),
]

N_SEEDS = 5
ETA_MULT = 2.0  # at theoretical boundary

count = 0
for ckpt in ckpts:
    sampler = ckpt["sampler"]
    bench = ckpt["benchmark"]
    subset = ckpt["subset"]
    eta_0 = float(ckpt["eta_0"])
    lr = eta_0 * ETA_MULT

    S_list = [int(d) for d in subset.replace("S", "")]
    ckpt_path = f"result/checkpoints/goal1_{sampler}_{bench}_{subset}_pretrain_seed0/epoch_500.pt"

    for abl_tag, overrides in ablations:
        for seed in range(N_SEEDS):
            exp_id = f"goal5abl_{sampler}_{bench}_{subset}_{abl_tag}_seed{seed}"
            cfg = {
                "goal": 5,
                "stage": "ablation",
                "seed": seed,
                "n_epochs": 300,
                "model": {"config": f"model/configs/{sampler}_default.yaml"},
                "benchmark": {"config": f"benchmark/configs/{bench}.yaml"},
                "subset": {"S": S_list, "full_target": True},
                "init": {"from_checkpoint": ckpt_path, "noise_level": 0.0},
                "training": {
                    "override_lr": lr,
                    **overrides.get("training", {}),
                },
                "logging": {"checkpoint_every": 300, "eval_every": 20, "n_eval_samples": 10000},
                "output": {
                    "checkpoint_dir": f"result/checkpoints/{exp_id}/",
                    "log_file": f"result/logs/{exp_id}.csv",
                },
            }
            if "optimizer" in overrides:
                cfg["optimizer"] = overrides["optimizer"]

            out_path = EXP_DIR / f"{exp_id}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            count += 1

print(f"Generated {count} ablation configs")
# 13 checkpoints x 6 ablations x 5 seeds = 390
