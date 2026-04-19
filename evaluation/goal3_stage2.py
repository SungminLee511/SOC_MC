#!/usr/bin/env python3
"""Goal 3 Stage 2: Eigenvector analysis — classify revival vs surviving.

For each checkpoint from Stage 1, load top-5 min and max Ritz vectors,
perturb θ → θ + ε_p * v_i, sample mode weights, classify as revival or surviving.

Outputs:
  evaluation/tables/goal3_eigvec_analysis.csv
  evaluation/figures/goal3_eigvec_classification.png
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.train import load_sampler, load_benchmark, evaluate_mode_weights
from evaluation.utils.hessian_ops import unflatten_params

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
TBL_DIR = ROOT / "evaluation" / "tables"
FIG_DIR = ROOT / "evaluation" / "figures"
CKPT_DIR = ROOT / "result" / "checkpoints"
EIGS_DIR = TBL_DIR / "goal3_eigenvalues"

EPSILON_P = 0.01           # perturbation scale
REVIVAL_THRESHOLD = 0.01   # |δα_k| > this → revival
N_EVAL = 10000             # samples for mode weight estimation
K = 5


def get_subset_indices(subset_name):
    """Parse 'S012' → (0, 1, 2)."""
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
    return sampler, energy_fn


def perturb_and_measure(sampler, energy_fn, v_flat, epsilon, device):
    """Perturb params by ε*v, measure mode weights, restore params."""
    params = list(sampler.parameters())
    shapes = [p.shape for p in params]

    # Save original
    original = [p.data.clone() for p in params]

    # Perturb
    v_list = unflatten_params(v_flat, shapes)
    for p, dv in zip(params, v_list):
        p.data.add_(dv, alpha=epsilon)

    # Measure
    alphas, _ = evaluate_mode_weights(sampler, energy_fn, N_EVAL, device)

    # Restore
    for p, orig in zip(params, original):
        p.data.copy_(orig)

    return np.array(alphas)


def analyze_checkpoint(sampler_name, bench, subset_name, device):
    """Analyze top-5 min/max eigenvectors for one checkpoint."""
    S = get_subset_indices(subset_name)
    S_comp = set(range(K)) - set(S)  # dead modes

    # Load Ritz vectors
    ritz_path = EIGS_DIR / f"{sampler_name}_{bench}_{subset_name}_ritz.pt"
    eigs_path = EIGS_DIR / f"{sampler_name}_{bench}_{subset_name}.json"
    if not ritz_path.exists() or not eigs_path.exists():
        return []

    ritz_data = torch.load(ritz_path, weights_only=False)
    with open(eigs_path) as f:
        eigs_data = json.load(f)

    eigenvalues = eigs_data["eigenvalues"]

    # Load model
    sampler, energy_fn = load_checkpoint(sampler_name, bench, subset_name, device)
    if sampler is None:
        return []

    # Baseline mode weights
    alpha_base = np.array(evaluate_mode_weights(sampler, energy_fn, N_EVAL, device)[0])

    rows = []
    for group, vecs_key, eig_indices in [
        ("min5", "ritz_vecs_min5", range(5)),
        ("max5", "ritz_vecs_max5", range(len(eigenvalues)-5, len(eigenvalues))),
    ]:
        vecs = ritz_data[vecs_key]
        for i, (vec, eig_idx) in enumerate(zip(vecs, eig_indices)):
            v_flat = torch.tensor(vec, dtype=torch.float32, device=device)
            # Normalize
            v_flat = v_flat / (v_flat.norm() + 1e-10)

            # Perturb and measure
            alpha_pert = perturb_and_measure(sampler, energy_fn, v_flat, EPSILON_P, device)
            delta_alpha = alpha_pert - alpha_base

            # Check revival: any dead mode gains weight
            revival_deltas = [abs(delta_alpha[k]) for k in S_comp]
            max_revival = max(revival_deltas) if revival_deltas else 0.0
            is_revival = max_revival > REVIVAL_THRESHOLD

            eig_val = eigenvalues[eig_idx] if eig_idx < len(eigenvalues) else float("nan")

            rows.append({
                "sampler": sampler_name,
                "benchmark": bench,
                "subset": subset_name,
                "eigvec_group": group,
                "eigvec_idx": i,
                "eigenvalue": eig_val,
                "delta_alpha": delta_alpha.tolist(),
                "max_revival_delta": max_revival,
                "classification": "revival" if is_revival else "surviving",
            })

    torch.cuda.empty_cache()
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("GOAL 3 STAGE 2: EIGENVECTOR ANALYSIS")
    print("=" * 60)

    # Load spectral summary for checkpoint list
    summary = pd.read_csv(TBL_DIR / "goal3_spectral_summary.csv")
    total = len(summary)
    print(f"Checkpoints: {total}")

    all_rows = []
    for idx, row in summary.iterrows():
        sampler = row["sampler"]
        bench = row["benchmark"]
        sname = row["subset"]
        lmin = row["lambda_min"]

        print(f"  [{idx+1}/{total}] {sampler}/{bench}/{sname} (λ_min={lmin:.4f})")

        try:
            rows = analyze_checkpoint(sampler, bench, sname, device)
            all_rows.extend(rows)
            n_rev = sum(1 for r in rows if r["classification"] == "revival")
            n_surv = len(rows) - n_rev
            print(f"    {n_rev} revival, {n_surv} surviving (of {len(rows)} eigvecs)")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Save results
    df = pd.DataFrame(all_rows)
    # Convert delta_alpha list to string for CSV
    df["delta_alpha"] = df["delta_alpha"].apply(str)
    out_path = TBL_DIR / "goal3_eigvec_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.name} ({len(df)} rows)")

    # ── Figure: eigenvalue vs classification ──
    if len(df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        revival = df[df["classification"] == "revival"]
        surviving = df[df["classification"] == "surviving"]
        ax.scatter(revival["eigenvalue"], revival["max_revival_delta"],
                  c="red", label=f"Revival ({len(revival)})", s=30, alpha=0.6)
        ax.scatter(surviving["eigenvalue"], surviving["max_revival_delta"],
                  c="blue", label=f"Surviving ({len(surviving)})", s=30, alpha=0.6)
        ax.axhline(REVIVAL_THRESHOLD, color="k", linestyle="--", alpha=0.5, label=f"threshold={REVIVAL_THRESHOLD}")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Eigenvalue", fontsize=12)
        ax.set_ylabel("|δα| (max dead-mode revival)", fontsize=12)
        ax.set_title("Eigenvector Classification: Revival vs Surviving", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = FIG_DIR / "goal3_eigvec_classification.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {fig_path.name}")

    # ── Summary stats ──
    print("\n" + "=" * 60)
    print("STAGE 2 SUMMARY")
    print("=" * 60)
    if len(df) > 0:
        for group in ["min5", "max5"]:
            sub = df[df["eigvec_group"] == group]
            n_rev = (sub["classification"] == "revival").sum()
            print(f"  {group}: {n_rev}/{len(sub)} revival ({100*n_rev/max(len(sub),1):.1f}%)")

        # Per stability category
        stable_subs = set(summary[summary["p_stab"] >= 0.5]["subset"])
        for cat, mask in [("stable", df["subset"].isin(stable_subs)),
                          ("unstable", ~df["subset"].isin(stable_subs))]:
            sub = df[mask]
            n_rev = (sub["classification"] == "revival").sum()
            print(f"  {cat} checkpoints: {n_rev}/{len(sub)} revival eigvecs")

    print("\nDone.")


if __name__ == "__main__":
    main()
