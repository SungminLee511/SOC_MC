#!/usr/bin/env python3
"""Goal 3 Stage 1: Extreme eigenvalues via Lanczos at selected checkpoints.

Auto-selects ~10 checkpoints per (benchmark, sampler) from Goal 2 results,
runs Lanczos T=50 on each, saves spectral data.

Outputs:
  evaluation/tables/goal3_spectral_summary.csv
  evaluation/tables/goal3_eigenvalues_{sampler}_{bench}_{subset}.json
  evaluation/figures/goal3_loss_vs_lambda_min.png
"""

import os, sys, json, yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.train import load_sampler, load_benchmark
from evaluation.utils.hessian_ops import compute_extreme_eigenvalues

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
TBL_DIR = ROOT / "evaluation" / "tables"
FIG_DIR = ROOT / "evaluation" / "figures"
CKPT_DIR = ROOT / "result" / "checkpoints"
EIGS_DIR = ROOT / "evaluation" / "tables" / "goal3_eigenvalues"
EIGS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = ["w5", "c5", "b5"]
SAMPLERS = ["as", "asbs"]
K = 5

# Hessian parameters (from experimental plan)
BATCH_SIZE = 1024
N_HVP_AVG = 8
T_LANCZOS = 50


def select_checkpoints(sampler, bench):
    """Auto-select ~10 checkpoints from Goal 2 ranked table."""
    tbl = TBL_DIR / f"goal2_ranked_{sampler}_{bench}.csv"
    if not tbl.exists():
        print(f"  WARNING: {tbl.name} not found, skipping")
        return []

    df = pd.read_csv(tbl).sort_values("L_S_star")

    # 3 most stable (lowest L_S*, p_stab >= 0.5)
    stable = df[df["p_stab"] >= 0.5].head(3)
    # 3 most unstable (highest L_S*)
    unstable = df.tail(3)
    # 4 mid-range
    mid_idx = len(df) // 2
    mid = df.iloc[max(0, mid_idx-2):mid_idx+2]

    selected = pd.concat([stable, mid, unstable]).drop_duplicates("subset")
    return selected.to_dict("records")


def load_checkpoint(sampler_name, bench, subset_name, device):
    """Load sampler at pretrain checkpoint."""
    # Model config
    model_cfg_path = str(ROOT / "model" / "configs" / f"{sampler_name}_default.yaml")
    bench_cfg_path = str(ROOT / "benchmark" / "configs" / f"{bench}.yaml")

    sampler, _ = load_sampler(model_cfg_path)
    sampler = sampler.to(device)

    # Load checkpoint
    ckpt_path = CKPT_DIR / f"goal1_{sampler_name}_{bench}_{subset_name}_pretrain_seed0" / "epoch_500.pt"
    if not ckpt_path.exists():
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sampler.load_state_dict(ckpt)
    sampler.eval()

    # Load benchmark (full target for Hessian at collapsed state)
    energy_fn = load_benchmark(bench_cfg_path)
    energy_fn = energy_fn.to(device) if hasattr(energy_fn, 'to') else energy_fn

    return sampler, energy_fn


def run_convergence_check(sampler, energy_fn, device, subset_name):
    """Quick convergence check: T=40 vs T=50."""
    results_40 = compute_extreme_eigenvalues(
        sampler, energy_fn, BATCH_SIZE, N_HVP_AVG, 40, device
    )
    results_50 = compute_extreme_eigenvalues(
        sampler, energy_fn, BATCH_SIZE, N_HVP_AVG, 50, device
    )

    lmin_diff = abs(results_40["lambda_min"] - results_50["lambda_min"])
    lmax_diff = abs(results_40["lambda_max"] - results_50["lambda_max"])
    lmin_rel = lmin_diff / (abs(results_50["lambda_min"]) + 1e-10)
    lmax_rel = lmax_diff / (abs(results_50["lambda_max"]) + 1e-10)

    converged = lmin_rel < 0.05 and lmax_rel < 0.05
    return converged, results_50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("GOAL 3 STAGE 1: EXTREME EIGENVALUES (LANCZOS T=50)")
    print("=" * 60)

    all_rows = []
    total_ckpts = 0
    done_ckpts = 0

    # Count total first
    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            ckpts = select_checkpoints(sampler, bench)
            total_ckpts += len(ckpts)

    print(f"Total checkpoints to process: {total_ckpts}")
    print()

    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            combo = f"{sampler.upper()}/{bench.upper()}"
            ckpts = select_checkpoints(sampler, bench)
            print(f"\n── {combo}: {len(ckpts)} checkpoints ──")

            for rec in ckpts:
                sname = rec["subset"]
                L_star = rec["L_S_star"]
                p_stab = rec["p_stab"]
                done_ckpts += 1

                print(f"  [{done_ckpts}/{total_ckpts}] {sname} (L*={L_star:.4f}, p_stab={p_stab:.1f})")

                # Check if already computed
                eigs_path = EIGS_DIR / f"{sampler}_{bench}_{sname}.json"
                if eigs_path.exists():
                    with open(eigs_path) as f:
                        results = json.load(f)
                    print(f"    CACHED: λ_min={results['lambda_min']:.6f}, λ_max={results['lambda_max']:.6f}")
                    all_rows.append({
                        "sampler": sampler, "benchmark": bench, "subset": sname,
                        "L_S_star": L_star, "p_stab": p_stab,
                        "lambda_min": results["lambda_min"],
                        "lambda_max": results["lambda_max"],
                        "converged": results.get("converged", True),
                        "dim": results["dim"],
                    })
                    continue

                # Load model
                model, energy_fn = load_checkpoint(sampler, bench, sname, device)
                if model is None:
                    print(f"    SKIP: checkpoint not found")
                    continue

                try:
                    # Run with convergence check
                    converged, results = run_convergence_check(model, energy_fn, device, sname)
                    results["converged"] = converged

                    # Save full eigenvalue data
                    # Don't save ritz vectors to JSON (too large) — save separately if needed
                    save_data = {k: v for k, v in results.items()
                                 if k not in ("ritz_vecs_min5", "ritz_vecs_max5")}
                    save_data["converged"] = converged
                    with open(eigs_path, "w") as f:
                        json.dump(save_data, f, indent=2)

                    # Save Ritz vectors as .pt
                    ritz_path = EIGS_DIR / f"{sampler}_{bench}_{sname}_ritz.pt"
                    torch.save({
                        "ritz_vecs_min5": results["ritz_vecs_min5"],
                        "ritz_vecs_max5": results["ritz_vecs_max5"],
                    }, ritz_path)

                    conv_str = "✓" if converged else "✗"
                    print(f"    λ_min={results['lambda_min']:.6f}, λ_max={results['lambda_max']:.6f} "
                          f"[conv={conv_str}, dim={results['dim']}]")

                    all_rows.append({
                        "sampler": sampler, "benchmark": bench, "subset": sname,
                        "L_S_star": L_star, "p_stab": p_stab,
                        "lambda_min": results["lambda_min"],
                        "lambda_max": results["lambda_max"],
                        "converged": converged,
                        "dim": results["dim"],
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_rows.append({
                        "sampler": sampler, "benchmark": bench, "subset": sname,
                        "L_S_star": L_star, "p_stab": p_stab,
                        "lambda_min": float("nan"), "lambda_max": float("nan"),
                        "converged": False, "dim": 0,
                    })

                # Clear GPU cache
                torch.cuda.empty_cache()

    # Save summary table
    summary_df = pd.DataFrame(all_rows)
    summary_path = TBL_DIR / "goal3_spectral_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary table: {summary_path.name} ({len(summary_df)} rows)")

    # ── Figure: L_S* vs lambda_min ──
    if len(summary_df) > 0 and not summary_df["lambda_min"].isna().all():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, bench in zip(axes, BENCHMARKS):
            for sampler, marker, color in [("as", "o", "blue"), ("asbs", "s", "red")]:
                sub = summary_df[(summary_df["benchmark"] == bench) &
                                 (summary_df["sampler"] == sampler)]
                if len(sub) == 0:
                    continue
                stable = sub[sub["p_stab"] >= 0.5]
                unstable = sub[sub["p_stab"] < 0.5]
                ax.scatter(stable["L_S_star"], stable["lambda_min"],
                          c="green", marker=marker, s=80, edgecolors="k",
                          label=f"{sampler.upper()} stable", zorder=5)
                ax.scatter(unstable["L_S_star"], unstable["lambda_min"],
                          c="red", marker=marker, s=80, edgecolors="k",
                          label=f"{sampler.upper()} unstable", zorder=5)
            ax.axhline(0, color="k", linestyle="--", alpha=0.5)
            ax.set_xlabel(r"$\hat{\mathcal{L}}_S^*$")
            ax.set_ylabel(r"$\lambda_{\min}$")
            ax.set_title(f"{bench.upper()}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Goal 3: L_S* vs λ_min (Lanczos T=50)", fontsize=13)
        fig.tight_layout()
        fig_path = FIG_DIR / "goal3_loss_vs_lambda_min.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Figure: {fig_path.name}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("STAGE 1 SUMMARY")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        cat = "STABLE" if row["p_stab"] >= 0.5 else "UNSTAB"
        sign = "+" if row["lambda_min"] >= 0 else "-"
        print(f"  {row['sampler']}/{row['benchmark']}/{row['subset']:8s} "
              f"L*={row['L_S_star']:.4f} λ_min={row['lambda_min']:+.6f} "
              f"λ_max={row['lambda_max']:.6f} [{cat}] λ_min_sign={sign}")

    # Theory check
    n_psd_stable = len(summary_df[(summary_df["p_stab"] >= 0.5) & (summary_df["lambda_min"] >= -1e-3)])
    n_stable = len(summary_df[summary_df["p_stab"] >= 0.5])
    n_neg_unstable = len(summary_df[(summary_df["p_stab"] < 0.5) & (summary_df["lambda_min"] < -1e-3)])
    n_unstable = len(summary_df[summary_df["p_stab"] < 0.5])

    print(f"\n  P3 check: PSD at stable: {n_psd_stable}/{n_stable}")
    print(f"  P4 check: negative λ at unstable: {n_neg_unstable}/{n_unstable}")
    print("\nDone.")


if __name__ == "__main__":
    main()
