#!/usr/bin/env python3
"""Goal 2: Predictivity Ranking — Does L_S* predict which subsets are stable?

Reads Goal 1 pretrain + stability logs, computes rankings, correlations,
threshold detection, and generates all figures/tables.

Outputs:
  evaluation/tables/goal2_*.csv
  evaluation/figures/goal2_*.png
  evaluation/goal2_metrics.json   (machine-readable summary)
"""

import os, json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "result" / "logs"
FIG_DIR = ROOT / "evaluation" / "figures"
TBL_DIR = ROOT / "evaluation" / "tables"
MET_PATH = ROOT / "evaluation" / "goal2_metrics.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = ["w5", "c5", "b5"]
SAMPLERS = ["as", "asbs"]
SEEDS = list(range(5))
K = 5

# Stability classification thresholds (from experimental plan)
MWD_TOL = 0.05        # stable if MWD < this for all eval points after warmup
MWD_ESCAPE = 0.15     # escaped if MWD > this for 3 consecutive evals
WARMUP_EPOCH = 20
MAX_EPOCH = 200

# Full target weights per benchmark (used to compute S-collapsed target)
def get_benchmark_weights(bench):
    """Return full target weight vector for each benchmark."""
    if bench == "w5":
        r = 0.5
        w = np.array([r**k for k in range(K)])
        return w / w.sum()
    else:  # c5, b5: equal weights
        return np.ones(K) / K

def get_collapsed_target(subset_indices, bench):
    """Return α^(S): target weights for S-collapsed state."""
    w = get_benchmark_weights(bench)
    alpha_S = np.zeros(K)
    for i in subset_indices:
        alpha_S[i] = w[i]
    if alpha_S.sum() > 0:
        alpha_S = alpha_S / alpha_S.sum()
    return alpha_S

# ── Subset enumeration ─────────────────────────────────────────────────
def all_subsets(k=5):
    """All proper non-trivial subsets of {0,...,k-1}."""
    subs = []
    for size in range(1, k):
        for combo in combinations(range(k), size):
            subs.append(combo)
    return subs

def subset_name(s):
    return "S" + "".join(str(i) for i in s)

ALL_SUBSETS = all_subsets(K)
ALL_SUBSET_NAMES = [subset_name(s) for s in ALL_SUBSETS]

# ── Data loading ────────────────────────────────────────────────────────
def load_pretrain_loss(sampler, bench, sname):
    """Load final training loss from pretrain CSV."""
    fname = LOG_DIR / f"goal1_{sampler}_{bench}_{sname}_pretrain_seed0.csv"
    if not fname.exists():
        return np.nan
    df = pd.read_csv(fname)
    train = df[df["event_type"] == "train"]
    if len(train) == 0:
        return np.nan
    # Average last 10 epochs for robustness
    return train["loss"].iloc[-10:].mean()


def load_stability_run(sampler, bench, sname, seed, subset_indices=None):
    """Load stability CSV, recompute MWD vs S-collapsed target, classify."""
    fname = LOG_DIR / f"goal1_{sampler}_{bench}_{sname}_stability_seed{seed}.csv"
    if not fname.exists():
        return None, "missing"
    df = pd.read_csv(fname)
    evals = df[df["event_type"] == "eval"].copy()
    if len(evals) == 0:
        return None, "no_evals"

    evals = evals.sort_values("epoch")
    alpha_cols = [f"alpha_{k+1}" for k in range(K)]

    # Recompute MWD against S-collapsed target (not full target)
    if subset_indices is not None:
        alpha_S = get_collapsed_target(subset_indices, bench)
        alphas = evals[alpha_cols].values.astype(float)
        mwd_recomputed = np.abs(alphas - alpha_S[np.newaxis, :]).max(axis=1)
        evals = evals.copy()
        evals["mwd"] = mwd_recomputed

    mwd_series = evals[["epoch", "mwd"]].reset_index(drop=True)

    # Classification
    post_warmup = mwd_series[mwd_series["epoch"] >= WARMUP_EPOCH]
    if len(post_warmup) == 0:
        return mwd_series, "ambiguous"

    # Stable: all MWD < tol after warmup
    if (post_warmup["mwd"] < MWD_TOL).all():
        return mwd_series, "stable"

    # Escaped: 3 consecutive MWD > escape threshold
    mwd_vals = post_warmup["mwd"].values
    consec = 0
    for v in mwd_vals:
        if v > MWD_ESCAPE:
            consec += 1
            if consec >= 3:
                return mwd_series, "escaped"
        else:
            consec = 0

    return mwd_series, "ambiguous"


# ── Analysis ────────────────────────────────────────────────────────────
def analyze_combo(sampler, bench):
    """Full Goal 2 analysis for one (sampler, benchmark) combo."""
    rows = []
    for s, sname in zip(ALL_SUBSETS, ALL_SUBSET_NAMES):
        loss = load_pretrain_loss(sampler, bench, sname)
        classifications = []
        for seed in SEEDS:
            _, cls = load_stability_run(sampler, bench, sname, seed, subset_indices=s)
            classifications.append(cls)

        n_stable = sum(1 for c in classifications if c == "stable")
        n_escaped = sum(1 for c in classifications if c == "escaped")
        n_ambig = sum(1 for c in classifications if c == "ambiguous")
        p_stab = n_stable / len(SEEDS)

        rows.append({
            "subset": sname,
            "S_size": len(s),
            "L_S_star": loss,
            "n_stable": n_stable,
            "n_escaped": n_escaped,
            "n_ambiguous": n_ambig,
            "p_stab": p_stab,
            "classifications": ",".join(classifications),
        })

    df = pd.DataFrame(rows).sort_values("L_S_star")
    return df


def threshold_detection(df):
    """Sweep threshold on L_S* to maximize classification accuracy."""
    losses = df["L_S_star"].dropna().sort_values().values
    best_acc, best_tau = 0, losses[0]
    accs = []
    for tau in losses:
        pred_stable = df["L_S_star"] < tau
        actual_stable = df["p_stab"] >= 0.5
        acc = (pred_stable == actual_stable).mean()
        accs.append((tau, acc))
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
    return best_tau, best_acc, accs


# ── Figures ─────────────────────────────────────────────────────────────
def fig_predictivity_scatter(df, sampler, bench):
    """Figure 4: L_S* vs p_stab scatter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = df["S_size"].values
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for sz in sorted(df["S_size"].unique()):
        mask = df["S_size"] == sz
        ax.scatter(df.loc[mask, "L_S_star"], df.loc[mask, "p_stab"],
                   c=colors.get(sz, "gray"), label=f"|S|={sz}", s=60, alpha=0.8,
                   edgecolors="k", linewidths=0.5)
    ax.set_xlabel(r"$\hat{\mathcal{L}}_S^*$ (pretrain loss)", fontsize=12)
    ax.set_ylabel(r"$p_{\mathrm{stab}}$ (stability proportion)", fontsize=12)
    ax.set_title(f"Predictivity Scatter — {sampler.upper()} / {bench.upper()}", fontsize=13)
    ax.legend(title="|S|")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / f"goal2_scatter_{sampler}_{bench}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_threshold_detection(accs_list, best_tau, sampler, bench):
    """Figure 5: Classification accuracy vs threshold."""
    fig, ax = plt.subplots(figsize=(7, 4))
    taus, accs = zip(*accs_list)
    ax.plot(taus, accs, "b-", linewidth=1.5)
    ax.axvline(best_tau, color="r", linestyle="--", label=f"best τ={best_tau:.3f}")
    ax.set_xlabel(r"Threshold $\hat{\tau}$", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(f"Threshold Detection — {sampler.upper()} / {bench.upper()}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / f"goal2_threshold_{sampler}_{bench}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_per_size_stratification(df, sampler, bench):
    """Figure 6: Predictivity scatter stratified by |S|."""
    sizes = sorted(df["S_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 4), sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    for ax, sz in zip(axes, sizes):
        sub = df[df["S_size"] == sz]
        ax.scatter(sub["L_S_star"], sub["p_stab"], s=60, alpha=0.8, edgecolors="k", linewidths=0.5)
        ax.set_xlabel(r"$\hat{\mathcal{L}}_S^*$")
        ax.set_title(f"|S|={sz} (n={len(sub)})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        # Per-size Spearman
        if len(sub) >= 3:
            rho, pval = spearmanr(sub["L_S_star"], sub["p_stab"])
            ax.text(0.05, 0.05, f"ρ={rho:.2f}", transform=ax.transAxes, fontsize=10)
    axes[0].set_ylabel(r"$p_{\mathrm{stab}}$")
    fig.suptitle(f"Per-|S| Stratification — {sampler.upper()} / {bench.upper()}", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"goal2_stratified_{sampler}_{bench}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_loss_histogram(df, sampler, bench):
    """Figure 3: Distribution of L_S* values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["L_S_star"].dropna(), bins=20, edgecolor="k", alpha=0.7)
    ax.set_xlabel(r"$\hat{\mathcal{L}}_S^*$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Pretrain Loss Distribution — {sampler.upper()} / {bench.upper()}", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / f"goal2_loss_hist_{sampler}_{bench}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def fig_cross_sampler_comparison(all_results):
    """Table 4 as figure: Spearman ρ and τ across all combos."""
    rows = []
    for (sampler, bench), res in all_results.items():
        rows.append({
            "Sampler": sampler.upper(),
            "Benchmark": bench.upper(),
            "Spearman_rho": res["spearman_rho"],
            "Spearman_p": res["spearman_pval"],
            "Best_tau": res["best_tau"],
            "Best_acc": res["best_acc"],
        })
    df = pd.DataFrame(rows)
    path = TBL_DIR / "goal2_cross_sampler.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}")

    # Also as a figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    fig.suptitle("Cross-Sampler Comparison (Goal 2)", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "goal2_cross_sampler_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GOAL 2: PREDICTIVITY RANKING ANALYSIS")
    print("=" * 60)

    all_results = {}
    all_metrics = {}

    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            combo = f"{sampler.upper()}/{bench.upper()}"
            print(f"\n── {combo} ──")

            # Step 1: Load & rank
            df = analyze_combo(sampler, bench)

            # Save ranked table
            tbl_path = TBL_DIR / f"goal2_ranked_{sampler}_{bench}.csv"
            df.to_csv(tbl_path, index=False)
            print(f"  Table: {tbl_path.name} ({len(df)} subsets)")

            # Step 2: Spearman correlation
            valid = df.dropna(subset=["L_S_star"])
            if len(valid) >= 3:
                rho, pval = spearmanr(valid["L_S_star"], valid["p_stab"])
            else:
                rho, pval = np.nan, np.nan
            print(f"  Spearman ρ = {rho:.4f} (p = {pval:.4e})")

            # Step 3: Threshold detection
            best_tau, best_acc, accs = threshold_detection(valid)
            print(f"  Best threshold τ = {best_tau:.4f}, accuracy = {best_acc:.4f}")

            # Step 4: Quick stability summary
            n_stable = (df["p_stab"] >= 0.5).sum()
            n_unstable = (df["p_stab"] < 0.5).sum()
            print(f"  Stable subsets (p≥0.5): {n_stable}/{len(df)}")
            print(f"  Unstable subsets (p<0.5): {n_unstable}/{len(df)}")

            # Figures
            fig_predictivity_scatter(df, sampler, bench)
            fig_threshold_detection(accs, best_tau, sampler, bench)
            fig_per_size_stratification(df, sampler, bench)
            fig_loss_histogram(df, sampler, bench)

            res = {
                "spearman_rho": float(rho),
                "spearman_pval": float(pval),
                "best_tau": float(best_tau),
                "best_acc": float(best_acc),
                "n_stable": int(n_stable),
                "n_unstable": int(n_unstable),
                "n_subsets": len(df),
            }
            all_results[(sampler, bench)] = res
            all_metrics[f"{sampler}_{bench}"] = res

    # Cross-sampler comparison
    print(f"\n── Cross-Sampler Summary ──")
    fig_cross_sampler_comparison(all_results)

    # Save all metrics
    with open(MET_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics: {MET_PATH.name}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, m in all_metrics.items():
        print(f"  {key}: ρ={m['spearman_rho']:.3f}, τ={m['best_tau']:.3f}, "
              f"acc={m['best_acc']:.3f}, stable={m['n_stable']}/{m['n_subsets']}")

    # Falsification check
    print("\n── Falsification Checks ──")
    for key, m in all_metrics.items():
        flags = []
        if abs(m["spearman_rho"]) < 0.3:
            flags.append("F2.1 TRIGGERED: |ρ| < 0.3")
        if m["spearman_rho"] > 0.3:
            flags.append("F2.2 TRIGGERED: ρ > +0.3 (inverted)")
        if m["best_acc"] < 0.6:
            flags.append("F2.3 WARNING: weak threshold (acc < 0.6)")
        if flags:
            for fl in flags:
                print(f"  ⚠ {key}: {fl}")
        else:
            print(f"  ✅ {key}: no falsification flags")

    print("\nDone.")


if __name__ == "__main__":
    main()
