#!/usr/bin/env python3
"""Goal 3 Stage 4: Five-term Hessian decomposition.

For each checkpoint, decompose the Hessian quadratic form v^T H v along the
minimum-eigenvalue Ritz vector into:
  P1 = controller variation (integral of |delta_u|^2 over trajectories)
  total_vHv = full Hessian quadratic form via finite differences
  remainder = total_vHv - P1

Outputs:
  evaluation/tables/goal3_decomposition.csv
  evaluation/figures/goal3_decomposition_bar.png
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.train import load_sampler, load_benchmark, evaluate_mode_weights
from evaluation.utils.hessian_ops import unflatten_params, flatten_params

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
TBL_DIR = ROOT / "evaluation" / "tables"
FIG_DIR = ROOT / "evaluation" / "figures"
CKPT_DIR = ROOT / "result" / "checkpoints"
EIGS_DIR = TBL_DIR / "goal3_eigenvalues"
CACHE_DIR = TBL_DIR / "goal3_decomp_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 0.01        # finite-diff step for Hessian quadratic form
N_TRAJ = 5000         # trajectories for P1 estimation
TRAJ_BATCH = 256      # batch size for trajectory sampling


def ts():
    return time.strftime("%H:%M:%S")


def load_checkpoint(sampler_name, bench, subset_name, device):
    """Load sampler at pretrain checkpoint."""
    model_cfg = str(ROOT / "model" / "configs" / f"{sampler_name}_default.yaml")
    bench_cfg = str(ROOT / "benchmark" / "configs" / f"{bench}.yaml")

    sampler, _ = load_sampler(model_cfg)
    sampler = sampler.to(device)

    ckpt_path = CKPT_DIR / f"goal1_{sampler_name}_{bench}_{subset_name}_pretrain_seed0" / "epoch_500.pt"
    if not ckpt_path.exists():
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sampler.load_state_dict(ckpt)
    sampler.eval()

    energy_fn = load_benchmark(bench_cfg)
    energy_fn = energy_fn.to(device) if hasattr(energy_fn, 'to') else energy_fn
    return sampler, energy_fn


def load_ritz_vector(sampler_name, bench, subset_name, device):
    """Load the minimum eigenvalue Ritz vector from Stage 1."""
    ritz_path = EIGS_DIR / f"{sampler_name}_{bench}_{subset_name}_ritz.pt"
    if not ritz_path.exists():
        return None

    ritz_data = torch.load(ritz_path, weights_only=False)
    # First of min5 = the smallest eigenvalue direction
    v = ritz_data["ritz_vecs_min5"][0]
    v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
    v_tensor = v_tensor / (v_tensor.norm() + 1e-10)
    return v_tensor


def compute_total_vHv(sampler, energy_fn, v_flat, device, epsilon=EPSILON):
    """Compute v^T H v via central finite differences on the loss.

    v^T H v ~ [L(theta + eps*v) - 2*L(theta) + L(theta - eps*v)] / eps^2
    Average over multiple batches for stability.
    """
    params = list(sampler.parameters())
    shapes = [p.shape for p in params]
    original = [p.data.clone() for p in params]
    v_list = unflatten_params(v_flat, shapes)

    n_avg = 8  # batches to average for stable loss estimate
    batch_size = 1024

    def avg_loss():
        total = 0.0
        for _ in range(n_avg):
            with torch.enable_grad():
                loss = sampler.train_step(batch_size, energy_fn, device)
            total += loss.item()
        return total / n_avg

    # L(theta)
    L0 = avg_loss()

    # L(theta + eps*v)
    for p, dv in zip(params, v_list):
        p.data.add_(dv, alpha=epsilon)
    L_plus = avg_loss()

    # Restore, then L(theta - eps*v)
    for p, orig in zip(params, original):
        p.data.copy_(orig)
    for p, dv in zip(params, v_list):
        p.data.add_(dv, alpha=-epsilon)
    L_minus = avg_loss()

    # Restore
    for p, orig in zip(params, original):
        p.data.copy_(orig)

    vHv = (L_plus - 2 * L0 + L_minus) / (epsilon ** 2)
    return vHv, L0


def compute_P1(sampler, energy_fn, v_flat, device, epsilon=EPSILON):
    """Compute P1 = E[integral |delta_u|^2 dt] along direction v.

    delta_u(x_t, t) = [u_{theta+eps*v}(x_t, t) - u_theta(x_t, t)] / eps

    Sample trajectories from current policy, evaluate controller at each
    (x_t, t), then perturb and evaluate again.
    """
    params = list(sampler.parameters())
    shapes = [p.shape for p in params]
    original = [p.data.clone() for p in params]
    v_list = unflatten_params(v_flat, shapes)

    total_p1 = 0.0
    n_batches = (N_TRAJ + TRAJ_BATCH - 1) // TRAJ_BATCH
    n_total = 0

    for b in range(n_batches):
        bs = min(TRAJ_BATCH, N_TRAJ - b * TRAJ_BATCH)

        # Sample trajectories at current theta (no grad needed for sampling)
        with torch.no_grad():
            traj = sampler.sample_trajectories(bs, device)

        timesteps = traj.timesteps  # (T,)
        states = traj.states        # list of T tensors, each (B, D)
        T_steps = len(timesteps) - 1  # number of intervals

        # Compute dt for integration
        dt_vals = timesteps[1:] - timesteps[:-1]  # (T-1,)

        # Compute u_theta at each (x_t, t) — no grad for forward pass
        u_base_list = []
        with torch.no_grad():
            for i in range(T_steps):
                t_i = timesteps[i].unsqueeze(0).expand(bs, 1)  # (B, 1)
                x_i = states[i]  # (B, D)
                u_i = sampler.controller(t_i, x_i)  # (B, D)
                u_base_list.append(u_i)

        # Perturb theta -> theta + eps*v
        for p, dv in zip(params, v_list):
            p.data.add_(dv, alpha=epsilon)

        # Compute u_{theta+eps*v} at same (x_t, t) points
        u_pert_list = []
        with torch.no_grad():
            for i in range(T_steps):
                t_i = timesteps[i].unsqueeze(0).expand(bs, 1)
                x_i = states[i]
                u_i = sampler.controller(t_i, x_i)
                u_pert_list.append(u_i)

        # Restore theta
        for p, orig in zip(params, original):
            p.data.copy_(orig)

        # Compute delta_u and integrate |delta_u|^2 dt
        batch_p1 = 0.0
        for i in range(T_steps):
            delta_u = (u_pert_list[i] - u_base_list[i]) / epsilon  # (B, D)
            # |delta_u|^2 summed over D, shape (B,)
            du_sq = (delta_u ** 2).sum(dim=-1)
            # Integrate with dt, average over batch
            batch_p1 += (du_sq * dt_vals[i]).mean().item()

        total_p1 += batch_p1 * bs
        n_total += bs

    P1 = total_p1 / n_total
    return P1


def process_checkpoint(sampler_name, bench, subset_name, row_data, device):
    """Process one checkpoint. Returns result dict or None."""
    cache_path = CACHE_DIR / f"{sampler_name}_{bench}_{subset_name}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"      CACHED: P1={cached['P1']:.4f}, total={cached['total_vHv']:.4f}")
        return cached

    # Load Ritz vector
    v_flat = load_ritz_vector(sampler_name, bench, subset_name, device)
    if v_flat is None:
        print(f"      SKIP: no Ritz vector found")
        return None

    sampler, energy_fn = load_checkpoint(sampler_name, bench, subset_name, device)
    if sampler is None:
        print(f"      SKIP: checkpoint not found")
        return None

    # Compute total v^T H v
    print(f"      [{ts()}] Computing total v^T H v (finite diff)...")
    total_vHv, L0 = compute_total_vHv(sampler, energy_fn, v_flat, device)
    print(f"      total_vHv = {total_vHv:.6f}, L0 = {L0:.6f}")

    # Compute P1
    print(f"      [{ts()}] Computing P1 ({N_TRAJ} trajectories)...")
    P1 = compute_P1(sampler, energy_fn, v_flat, device)
    print(f"      P1 = {P1:.6f}")

    remainder = total_vHv - P1
    P1_fraction = P1 / (abs(total_vHv) + 1e-10)

    result = {
        "sampler": sampler_name,
        "benchmark": bench,
        "subset": subset_name,
        "L_S_star": row_data["L_S_star"],
        "p_stab": row_data["p_stab"],
        "lambda_min": row_data["lambda_min"],
        "P1": P1,
        "total_vHv": total_vHv,
        "remainder": remainder,
        "P1_fraction": P1_fraction,
    }

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)

    torch.cuda.empty_cache()
    return result


def make_figure(df):
    """Bar chart of decomposition for 5 representative checkpoints."""
    if len(df) == 0:
        return

    # Select 5 representative: mix of stable/unstable, different benchmarks
    stable = df[df["p_stab"] >= 0.5].head(2)
    unstable = df[df["p_stab"] < 0.5]
    # Pick from different benchmarks if possible
    unstable_picks = []
    for bench in ["w5", "c5", "b5"]:
        sub = unstable[unstable["benchmark"] == bench]
        if len(sub) > 0:
            unstable_picks.append(sub.iloc[0:1])
    if len(unstable_picks) > 0:
        unstable_sel = pd.concat(unstable_picks).head(3)
    else:
        unstable_sel = unstable.head(3)

    reps = pd.concat([stable, unstable_sel]).head(5)

    if len(reps) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{r['sampler']}/{r['benchmark']}/{r['subset']}" for _, r in reps.iterrows()]
    x = np.arange(len(labels))
    width = 0.25

    p1_vals = reps["P1"].values
    rem_vals = reps["remainder"].values
    total_vals = reps["total_vHv"].values

    bars1 = ax.bar(x - width, p1_vals, width, label="P1 (controller var.)", color="steelblue")
    bars2 = ax.bar(x, rem_vals, width, label="Remainder", color="salmon")
    bars3 = ax.bar(x + width, total_vals, width, label=r"Total $v^T H v$", color="gray", alpha=0.7)

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Goal 3: Hessian Quadratic Form Decomposition", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    fig_path = FIG_DIR / "goal3_decomposition_bar.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")
    print("=" * 60)
    print("GOAL 3 STAGE 4: FIVE-TERM DECOMPOSITION")
    print("=" * 60)

    summary = pd.read_csv(TBL_DIR / "goal3_spectral_summary.csv")
    total = len(summary)
    print(f"Checkpoints: {total}")

    all_results = []
    for idx, row in summary.iterrows():
        sampler = row["sampler"]
        bench = row["benchmark"]
        sname = row["subset"]

        print(f"\n  [{idx+1}/{total}] {sampler}/{bench}/{sname} "
              f"(L*={row['L_S_star']:.4f}, lambda_min={row['lambda_min']:.4f})")

        try:
            result = process_checkpoint(sampler, bench, sname, row, device)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save CSV
    df = pd.DataFrame(all_results)
    out_path = TBL_DIR / "goal3_decomposition.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[{ts()}] Saved: {out_path} ({len(df)} rows)")

    # Figure
    if len(df) > 0:
        make_figure(df)

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 4 SUMMARY")
    print("=" * 60)
    if len(df) > 0:
        print(f"  Total checkpoints: {len(df)}")
        print(f"  P1 fraction: mean={df['P1_fraction'].mean():.3f}, "
              f"median={df['P1_fraction'].median():.3f}")
        print(f"  P1 > |total|: {(df['P1'] > df['total_vHv'].abs()).sum()}/{len(df)}")

        # Per stability
        for cat, mask in [("stable", df["p_stab"] >= 0.5), ("unstable", df["p_stab"] < 0.5)]:
            sub = df[mask]
            if len(sub) > 0:
                print(f"  {cat}: P1_frac mean={sub['P1_fraction'].mean():.3f}, "
                      f"remainder mean={sub['remainder'].mean():.4f}")

    print(f"\n[{ts()}] Done.")


if __name__ == "__main__":
    main()
