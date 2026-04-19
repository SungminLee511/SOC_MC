#!/usr/bin/env python3
"""Goal 5: Step-Size Escape — Generate experiment configs.

Reads goal3_spectral_summary.csv, selects up to 3 stable checkpoints per
(sampler, benchmark) pair, computes eta_0 = 1/lambda_max, and writes YAML
configs for a step-size sweep with 5 seeds each.

Outputs:
  result/experiments/goal5_*.yaml
  evaluation/tables/goal5_checkpoint_selection.csv
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TBL_DIR = ROOT / "evaluation" / "tables"
EXP_DIR = ROOT / "result" / "experiments"

EXP_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

SPECTRAL_CSV = TBL_DIR / "goal3_spectral_summary.csv"

P_STAB_THRESH = 0.5
MAX_PER_GROUP = 3
SEEDS = list(range(5))
ETA_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]

N_EPOCHS = 300
CHECKPOINT_EVERY = 300
EVAL_EVERY = 20
N_EVAL_SAMPLES = 10000

# ── Helpers ─────────────────────────────────────────────────────────────
def parse_subset_indices(sname):
    """'S0123' -> [0, 1, 2, 3]"""
    return [int(c) for c in sname[1:]]


def eta_mult_str(m):
    """Format multiplier for filenames: 0.5 -> '0.5', 1.0 -> '1.0'."""
    return f"{m:g}"


def make_yaml(*, goal, stage, seed, n_epochs, sampler, bench, subset_name,
              S_list, from_checkpoint, override_lr, checkpoint_dir, log_file):
    """Build YAML config string."""
    S_str = "[" + ", ".join(str(i) for i in S_list) + "]"
    return f"""\
goal: {goal}
stage: {stage}
seed: {seed}
n_epochs: {n_epochs}

model:
  config: model/configs/{sampler}_default.yaml

benchmark:
  config: benchmark/configs/{bench}.yaml

subset:
  S: {S_str}
  full_target: true

init:
  from_checkpoint: {from_checkpoint}
  noise_level: 0.0

training:
  override_lr: {override_lr}

logging:
  checkpoint_every: {CHECKPOINT_EVERY}
  eval_every: {EVAL_EVERY}
  n_eval_samples: {N_EVAL_SAMPLES}

output:
  checkpoint_dir: {checkpoint_dir}
  log_file: {log_file}
"""


# ── Main ────────────────────────────────────────────────────────────────
def main():
    if not SPECTRAL_CSV.exists():
        print(f"ERROR: {SPECTRAL_CSV} not found"); sys.exit(1)

    df = pd.read_csv(SPECTRAL_CSV)
    print(f"Loaded {len(df)} rows from spectral summary")

    # Filter stable checkpoints
    stable = df[df["p_stab"] >= P_STAB_THRESH].copy()
    print(f"Stable checkpoints (p_stab >= {P_STAB_THRESH}): {len(stable)}")

    # Select up to MAX_PER_GROUP per (sampler, benchmark), sorted by p_stab desc then L_S_star asc
    stable = stable.sort_values(["p_stab", "L_S_star"], ascending=[False, True])
    selected = stable.groupby(["sampler", "benchmark"]).head(MAX_PER_GROUP).reset_index(drop=True)
    print(f"Selected checkpoints: {len(selected)}")

    # Save selection table
    sel_rows = []
    for _, row in selected.iterrows():
        lmax = row["lambda_max"]
        eta_0 = 1.0 / lmax
        eta_th = 2.0 / lmax
        sel_rows.append({
            "sampler": row["sampler"],
            "benchmark": row["benchmark"],
            "subset": row["subset"],
            "lambda_max": lmax,
            "eta_0": eta_0,
            "eta_th": eta_th,
            "p_stab": row["p_stab"],
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_path = TBL_DIR / "goal5_checkpoint_selection.csv"
    sel_df.to_csv(sel_path, index=False)
    print(f"Saved checkpoint selection: {sel_path}")

    # Generate configs
    n_configs = 0
    for _, row in selected.iterrows():
        sampler = row["sampler"]
        bench = row["benchmark"]
        subset = row["subset"]
        lmax = row["lambda_max"]
        eta_0 = 1.0 / lmax
        S_list = parse_subset_indices(subset)

        from_ckpt = f"result/checkpoints/goal1_{sampler}_{bench}_{subset}_pretrain_seed0/epoch_500.pt"

        for mult in ETA_MULTS:
            eta_val = mult * eta_0
            m_str = eta_mult_str(mult)
            for seed in SEEDS:
                tag = f"goal5_{sampler}_{bench}_{subset}_eta{m_str}_seed{seed}"
                ckpt_dir = f"result/checkpoints/{tag}/"
                log_file = f"result/logs/{tag}.csv"
                yaml_path = EXP_DIR / f"{tag}.yaml"

                content = make_yaml(
                    goal=5, stage="escape", seed=seed, n_epochs=N_EPOCHS,
                    sampler=sampler, bench=bench, subset_name=subset,
                    S_list=S_list, from_checkpoint=from_ckpt,
                    override_lr=eta_val, checkpoint_dir=ckpt_dir,
                    log_file=log_file,
                )
                yaml_path.write_text(content)
                n_configs += 1

    print(f"Generated {n_configs} YAML configs in {EXP_DIR}")


if __name__ == "__main__":
    main()
