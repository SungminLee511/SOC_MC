#!/usr/bin/env python3
"""Goal 3 Stage 3: Revival-subspace projection.

For each checkpoint, sample random parameter directions, perturb, measure
dead-mode revival, orthogonalize revival directions via QR, then run Lanczos
on the projected Hessian to get lambda_min^rev.

Outputs:
  evaluation/tables/goal3_revival_projection.csv
  evaluation/figures/goal3_loss_vs_lambda_min_rev.png
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.train import load_sampler, load_benchmark, evaluate_mode_weights
from evaluation.utils.hessian_ops import (
    compute_extreme_eigenvalues, hvp_flat, make_averaged_loss_fn,
    lanczos, unflatten_params, flatten_params,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
TBL_DIR = ROOT / "evaluation" / "tables"
FIG_DIR = ROOT / "evaluation" / "figures"
CKPT_DIR = ROOT / "result" / "checkpoints"
EIGS_DIR = TBL_DIR / "goal3_eigenvalues"
CACHE_DIR = TBL_DIR / "goal3_revival_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 5
N_DIR = 200           # random directions to sample
EPSILON = 0.05        # perturbation scale (larger than stage 2's 0.01)
REVIVAL_THRESH = 0.005  # |delta_alpha_k| threshold for dead modes
N_EVAL = 10000        # samples for mode weight estimation
BATCH_SIZE = 1024     # for HVP computation
N_HVP_AVG = 8         # batches to average per HVP


def ts():
    """Timestamp string."""
    return time.strftime("%H:%M:%S")


def get_subset_indices(subset_name):
    """Parse 'S012' -> (0, 1, 2)."""
    digits = subset_name.replace("S", "")
    return tuple(int(d) for d in digits)


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


def perturb_and_measure(sampler, energy_fn, v_flat, epsilon, device):
    """Perturb params by eps*v, measure mode weights, restore."""
    params = list(sampler.parameters())
    shapes = [p.shape for p in params]
    original = [p.data.clone() for p in params]

    v_list = unflatten_params(v_flat, shapes)
    for p, dv in zip(params, v_list):
        p.data.add_(dv, alpha=epsilon)

    alphas, _ = evaluate_mode_weights(sampler, energy_fn, N_EVAL, device)

    for p, orig in zip(params, original):
        p.data.copy_(orig)

    return np.array(alphas)


def find_revival_directions(sampler, energy_fn, device, subset_name):
    """Sample N_DIR random directions, return those that revive dead modes."""
    S = set(get_subset_indices(subset_name))
    S_comp = sorted(set(range(K)) - S)

    if len(S_comp) == 0:
        return np.zeros((0, 0)), 0

    params = list(sampler.parameters())
    dim = sum(p.numel() for p in params)

    # Baseline mode weights
    alpha_base = np.array(evaluate_mode_weights(sampler, energy_fn, N_EVAL, device)[0])

    revival_dirs = []
    for i in range(N_DIR):
        v = torch.randn(dim, device=device)
        v = v / v.norm()

        alpha_pert = perturb_and_measure(sampler, energy_fn, v, EPSILON, device)
        delta_alpha = alpha_pert - alpha_base

        # Check dead-mode revival
        max_dead_delta = max(abs(delta_alpha[k]) for k in S_comp)
        if max_dead_delta > REVIVAL_THRESH:
            revival_dirs.append(v.cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"      [{ts()}] {i+1}/{N_DIR} dirs, {len(revival_dirs)} revival so far")

    n_revival = len(revival_dirs)
    if n_revival == 0:
        return np.zeros((dim, 0)), 0

    # Stack and orthogonalize via QR
    V = np.array(revival_dirs).T  # (dim, n_revival)
    Q, R = np.linalg.qr(V, mode='reduced')
    # Remove near-zero columns (numerically dependent)
    diag = np.abs(np.diag(R))
    keep = diag > 1e-8
    Q = Q[:, keep]

    return Q, n_revival


def projected_lanczos(sampler, energy_fn, Q, device):
    """Run Lanczos on projected Hessian Q^T H Q.

    Q: (dim, r) orthonormal basis for revival subspace.
    Returns eigenvalues of the projected Hessian.
    """
    r = Q.shape[1]
    T_lanczos = min(r, 30)

    Q_tensor = torch.tensor(Q, dtype=torch.float32, device=device)
    params = list(sampler.parameters())

    def projected_hvp(v_proj):
        """v_proj is r-dim. Returns Q^T H Q v_proj."""
        # Lift to full space: w = Q @ v_proj
        w = Q_tensor @ v_proj  # (dim,)
        # Full HVP
        loss_fn = make_averaged_loss_fn(sampler, energy_fn, BATCH_SIZE, N_HVP_AVG, device)
        Hw = hvp_flat(loss_fn, params, w)  # (dim,)
        # Project back
        return Q_tensor.T @ Hw  # (r,)

    eig_vals, ritz_vecs = lanczos(projected_hvp, r, T_lanczos, device)
    return eig_vals, ritz_vecs


def process_checkpoint(sampler_name, bench, subset_name, row_data, device):
    """Process one checkpoint. Returns result dict or None."""
    # Check cache
    cache_path = CACHE_DIR / f"{sampler_name}_{bench}_{subset_name}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"      CACHED: revival_dim={cached['revival_dim']}, "
              f"lambda_min_rev={cached.get('lambda_min_rev', 'N/A')}")
        return cached

    sampler, energy_fn = load_checkpoint(sampler_name, bench, subset_name, device)
    if sampler is None:
        print(f"      SKIP: checkpoint not found")
        return None

    # Find revival directions
    print(f"      [{ts()}] Sampling {N_DIR} random directions...")
    Q, n_revival_dirs = find_revival_directions(sampler, energy_fn, device, subset_name)
    revival_dim = Q.shape[1]

    result = {
        "sampler": sampler_name,
        "benchmark": bench,
        "subset": subset_name,
        "L_S_star": row_data["L_S_star"],
        "p_stab": row_data["p_stab"],
        "lambda_min_full": row_data["lambda_min"],
        "lambda_min_rev": float("nan"),
        "revival_dim": revival_dim,
        "n_revival_dirs": n_revival_dirs,
    }

    if revival_dim >= 2:
        print(f"      [{ts()}] Running projected Lanczos (dim={revival_dim})...")
        try:
            eig_vals, _ = projected_lanczos(sampler, energy_fn, Q, device)
            result["lambda_min_rev"] = float(eig_vals[0])
            print(f"      lambda_min_rev = {eig_vals[0]:.6f}")
        except Exception as e:
            print(f"      Projected Lanczos ERROR: {e}")
    else:
        print(f"      Revival dim={revival_dim} < 2, skipping projected Lanczos")

    # Save cache
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)

    torch.cuda.empty_cache()
    return result


def make_figure(df):
    """Scatter plot: L_S* vs lambda_min_rev, colored by stability."""
    valid = df.dropna(subset=["lambda_min_rev"])
    if len(valid) == 0:
        print("  No valid lambda_min_rev values, skipping figure")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    stable = valid[valid["p_stab"] >= 0.5]
    unstable = valid[valid["p_stab"] < 0.5]

    for sub, color, label in [
        (stable, "green", "Stable (p_stab >= 0.5)"),
        (unstable, "red", "Unstable (p_stab < 0.5)"),
    ]:
        if len(sub) == 0:
            continue
        for sampler, marker in [("as", "o"), ("asbs", "s")]:
            ss = sub[sub["sampler"] == sampler]
            if len(ss) == 0:
                continue
            ax.scatter(
                ss["L_S_star"], ss["lambda_min_rev"],
                c=color, marker=marker, s=80, edgecolors="k",
                label=f"{sampler.upper()} {label}", zorder=5,
            )

    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\hat{\mathcal{L}}_S^*$", fontsize=12)
    ax.set_ylabel(r"$\lambda_{\min}^{\mathrm{rev}}$", fontsize=12)
    ax.set_title("Goal 3: Loss vs Revival-Subspace Minimum Eigenvalue", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = FIG_DIR / "goal3_loss_vs_lambda_min_rev.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")
    print("=" * 60)
    print("GOAL 3 STAGE 3: REVIVAL-SUBSPACE PROJECTION")
    print("=" * 60)

    summary = pd.read_csv(TBL_DIR / "goal3_spectral_summary.csv")
    total = len(summary)
    print(f"Checkpoints: {total}")

    all_results = []
    for idx, row in summary.iterrows():
        sampler = row["sampler"]
        bench = row["benchmark"]
        sname = row["subset"]
        lmin = row["lambda_min"]

        print(f"\n  [{idx+1}/{total}] {sampler}/{bench}/{sname} "
              f"(L*={row['L_S_star']:.4f}, lambda_min={lmin:.4f})")

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
    out_path = TBL_DIR / "goal3_revival_projection.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[{ts()}] Saved: {out_path} ({len(df)} rows)")

    # Figure
    if len(df) > 0:
        make_figure(df)

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 3 SUMMARY")
    print("=" * 60)
    if len(df) > 0:
        has_rev = df[df["revival_dim"] >= 2]
        no_rev = df[df["revival_dim"] < 2]
        print(f"  Checkpoints with revival subspace (dim>=2): {len(has_rev)}/{len(df)}")
        print(f"  Checkpoints without revival subspace: {len(no_rev)}/{len(df)}")
        if len(has_rev) > 0:
            print(f"  Revival dim: mean={has_rev['revival_dim'].mean():.1f}, "
                  f"max={has_rev['revival_dim'].max()}")
            valid_rev = has_rev.dropna(subset=["lambda_min_rev"])
            if len(valid_rev) > 0:
                n_neg = (valid_rev["lambda_min_rev"] < 0).sum()
                print(f"  Negative lambda_min_rev: {n_neg}/{len(valid_rev)}")

    print(f"\n[{ts()}] Done.")


if __name__ == "__main__":
    main()
