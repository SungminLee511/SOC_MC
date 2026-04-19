#!/usr/bin/env python3
"""Goal 1: Stability Test Analysis

Reads stability and pretrain logs, classifies runs, and generates
all figures, tables, and metrics for Goal 1.

Outputs:
  evaluation/tables/goal1_stability_matrix.csv
  evaluation/figures/goal1_mwd_trajectories.png
  evaluation/figures/goal1_stability_by_size.png
  evaluation/figures/goal1_loss_histogram.png
  evaluation/figures/goal1_noise_ablation.png
  evaluation/goal1_metrics.json
"""

import os, json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "result" / "logs"
FIG_DIR = ROOT / "evaluation" / "figures"
TBL_DIR = ROOT / "evaluation" / "tables"
MET_PATH = ROOT / "evaluation" / "goal1_metrics.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = ["w5", "c5", "b5"]
SAMPLERS = ["as", "asbs"]
SEEDS = list(range(5))
K = 5

# Stability classification thresholds (from experimental plan)
MWD_TOL = 0.05        # stable: MWD < this for all eval points after warmup
MWD_ESCAPE = 0.15     # escaped: MWD > this for 3+ consecutive eval points
WARMUP_EPOCH = 20
MAX_EPOCH = 200

# Noise ablation levels — noise=0 is baseline (stability logs)
# File suffix mapping: 0.0 -> stability, 0.01 -> noise001, 0.05 -> noise005, 0.1 -> noise01
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
NOISE_SUFFIX = {0.0: None, 0.01: "noise001", 0.05: "noise005", 0.1: "noise01"}

# Mode separation ablation
RHO_SEPS = [3, 4, 5, 7]


def ts():
    """Current KST timestamp string."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


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

# ── Easy/Hard classification per benchmark ─────────────────────────────
# W5: easy = subsets of {0,1,2}; hard = contains mode 3 or 4
# C5: easy = subsets of {0,1,2}; hard = contains mode 3 or 4
# B5: easy = subsets of {0,1,2}; hard = contains mode 3 or 4
EASY_MODES = {0, 1, 2}
HARD_MODES = {3, 4}


def classify_easy_hard(subset_indices):
    """Return 'easy' if subset only contains EASY_MODES, else 'hard'."""
    s_set = set(subset_indices)
    if s_set.issubset(EASY_MODES):
        return "easy"
    return "hard"


# ── Data loading ────────────────────────────────────────────────────────
def load_stability_run(sampler, bench, sname, seed, noise_suffix=None):
    """Load a stability CSV and return (mwd_series, classification).

    noise_suffix: None -> standard stability log
                  str  -> noise ablation log (e.g. 'noise001')
    """
    if noise_suffix is None:
        fname = LOG_DIR / f"goal1_{sampler}_{bench}_{sname}_stability_seed{seed}.csv"
    else:
        fname = LOG_DIR / f"goal1_{sampler}_{bench}_{sname}_{noise_suffix}_seed{seed}.csv"

    if not fname.exists():
        return None, "missing"

    try:
        df = pd.read_csv(fname)
    except Exception as e:
        warnings.warn(f"Failed to read {fname}: {e}")
        return None, "missing"

    evals = df[df["event_type"] == "eval"].copy()
    if len(evals) == 0:
        return None, "no_evals"

    evals = evals.sort_values("epoch").reset_index(drop=True)

    # MWD column already in CSV; use it directly
    if "mwd" not in evals.columns:
        return None, "no_mwd"

    mwd_series = evals[["epoch", "mwd"]].copy()
    mwd_series["mwd"] = pd.to_numeric(mwd_series["mwd"], errors="coerce")

    # Classification using eval points in [WARMUP_EPOCH, MAX_EPOCH]
    post_warmup = mwd_series[mwd_series["epoch"] >= WARMUP_EPOCH].dropna(subset=["mwd"])
    if len(post_warmup) == 0:
        return mwd_series, "ambiguous"

    mwd_vals = post_warmup["mwd"].values

    # Stable: all MWD < MWD_TOL
    if (mwd_vals < MWD_TOL).all():
        return mwd_series, "stable"

    # Escaped: 3 consecutive MWD > MWD_ESCAPE
    consec = 0
    for v in mwd_vals:
        if v > MWD_ESCAPE:
            consec += 1
            if consec >= 3:
                return mwd_series, "escaped"
        else:
            consec = 0

    return mwd_series, "ambiguous"


def load_pretrain_loss(sampler, bench, sname):
    """Load final pretrain loss (L_S*) from pretrain CSV."""
    fname = LOG_DIR / f"goal1_{sampler}_{bench}_{sname}_pretrain_seed0.csv"
    if not fname.exists():
        return np.nan
    try:
        df = pd.read_csv(fname)
    except Exception as e:
        warnings.warn(f"Failed to read {fname}: {e}")
        return np.nan

    train = df[df["event_type"] == "train"]
    if len(train) == 0:
        return np.nan
    # Use last epoch's loss
    return float(train["loss"].iloc[-1])


def final_mwd(mwd_series):
    """Return MWD at the last eval epoch."""
    if mwd_series is None or len(mwd_series) == 0:
        return np.nan
    last = mwd_series.dropna(subset=["mwd"])
    if len(last) == 0:
        return np.nan
    return float(last["mwd"].iloc[-1])


# ── Table 1: Stability Matrix ───────────────────────────────────────────
def build_stability_matrix():
    """Build per-(subset, bench, sampler) stability proportions."""
    print(f"[{ts()}] Building stability matrix...")
    rows = []
    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            for s, sname in zip(ALL_SUBSETS, ALL_SUBSET_NAMES):
                eh = classify_easy_hard(s)
                classifications = []
                for seed in SEEDS:
                    _, cls = load_stability_run(sampler, bench, sname, seed)
                    classifications.append(cls)

                n_stable = sum(1 for c in classifications if c == "stable")
                n_escaped = sum(1 for c in classifications if c == "escaped")
                n_ambig = sum(1 for c in classifications if c == "ambiguous")
                n_missing = sum(1 for c in classifications if c in ("missing", "no_evals", "no_mwd"))
                valid = len(SEEDS) - n_missing
                p_stab = n_stable / max(valid, 1)

                rows.append({
                    "benchmark": bench,
                    "sampler": sampler,
                    "subset": sname,
                    "S_size": len(s),
                    "easy_hard": eh,
                    "n_stable": n_stable,
                    "n_escaped": n_escaped,
                    "n_ambiguous": n_ambig,
                    "n_missing": n_missing,
                    "p_stab": round(p_stab, 4),
                    "classifications": ",".join(classifications),
                })

    df = pd.DataFrame(rows)
    path = TBL_DIR / "goal1_stability_matrix.csv"
    df.to_csv(path, index=False)
    print(f"[{ts()}] Saved {path}")
    return df


# ── Figure 1: MWD Trajectories ─────────────────────────────────────────
def fig_mwd_trajectories(stability_df):
    """2×3 grid: one subplot per (bench, sampler). All 30 subsets overlaid."""
    print(f"[{ts()}] Generating MWD trajectory figure...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    # axes[row=sampler, col=bench]
    SAMPLER_ROW = {"as": 0, "asbs": 1}
    BENCH_COL = {"w5": 0, "c5": 1, "b5": 2}
    COLOR = {"easy": "#2196F3", "hard": "#F44336"}
    ALPHA_EASY = 0.5
    ALPHA_HARD = 0.4

    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            ax = axes[SAMPLER_ROW[sampler]][BENCH_COL[bench]]
            subset_rows = stability_df[
                (stability_df["benchmark"] == bench) &
                (stability_df["sampler"] == sampler)
            ]

            plotted_easy = False
            plotted_hard = False

            for _, row in subset_rows.iterrows():
                sname = row["subset"]
                eh = row["easy_hard"]
                color = COLOR[eh]

                # Average MWD trajectory over seeds
                traj_list = []
                for seed in SEEDS:
                    mwd_series, _ = load_stability_run(sampler, bench, sname, seed)
                    if mwd_series is not None and len(mwd_series) > 0:
                        traj_list.append(mwd_series.set_index("epoch")["mwd"])

                if len(traj_list) == 0:
                    continue

                combined = pd.concat(traj_list, axis=1)
                mean_mwd = combined.mean(axis=1)
                epochs = mean_mwd.index.values

                lw = 1.0
                alpha = ALPHA_EASY if eh == "easy" else ALPHA_HARD
                label = None
                if eh == "easy" and not plotted_easy:
                    label = "easy"
                    plotted_easy = True
                elif eh == "hard" and not plotted_hard:
                    label = "hard"
                    plotted_hard = True

                ax.plot(epochs, mean_mwd.values, color=color, alpha=alpha,
                        linewidth=lw, label=label)

            # Reference lines
            ax.axhline(MWD_TOL, color="green", linestyle="--", linewidth=0.8, alpha=0.7,
                       label=f"stable thr ({MWD_TOL})")
            ax.axhline(MWD_ESCAPE, color="orange", linestyle="--", linewidth=0.8, alpha=0.7,
                       label=f"escape thr ({MWD_ESCAPE})")

            ax.set_title(f"{bench.upper()} / {sampler.upper()}", fontsize=11)
            ax.set_xlim(0, MAX_EPOCH)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

            if SAMPLER_ROW[sampler] == 1:
                ax.set_xlabel("Epoch", fontsize=10)
            if BENCH_COL[bench] == 0:
                ax.set_ylabel("MWD (mean over seeds)", fontsize=10)

    fig.suptitle("Goal 1: MWD Trajectories — All Subsets per (Benchmark, Sampler)", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "goal1_mwd_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts()}] Saved {path}")


# ── Figure 2: Stability by Subset Size ─────────────────────────────────
def fig_stability_by_size(stability_df):
    """6 subplots: stability proportion stratified by |S|."""
    print(f"[{ts()}] Generating stability-by-size figure...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    SAMPLER_ROW = {"as": 0, "asbs": 1}
    BENCH_COL = {"w5": 0, "c5": 1, "b5": 2}

    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            ax = axes[SAMPLER_ROW[sampler]][BENCH_COL[bench]]
            sub = stability_df[
                (stability_df["benchmark"] == bench) &
                (stability_df["sampler"] == sampler)
            ]

            sizes = sorted(sub["S_size"].unique())
            easy_means, hard_means, easy_stds, hard_stds = [], [], [], []

            for sz in sizes:
                sz_sub = sub[sub["S_size"] == sz]
                easy_p = sz_sub[sz_sub["easy_hard"] == "easy"]["p_stab"]
                hard_p = sz_sub[sz_sub["easy_hard"] == "hard"]["p_stab"]
                easy_means.append(easy_p.mean() if len(easy_p) > 0 else np.nan)
                hard_means.append(hard_p.mean() if len(hard_p) > 0 else np.nan)
                easy_stds.append(easy_p.std() if len(easy_p) > 0 else 0.0)
                hard_stds.append(hard_p.std() if len(hard_p) > 0 else 0.0)

            x = np.arange(len(sizes))
            width = 0.35

            bars_easy = ax.bar(x - width / 2, easy_means, width, yerr=easy_stds,
                               label="easy", color="#2196F3", alpha=0.8,
                               capsize=4, error_kw={"linewidth": 1})
            bars_hard = ax.bar(x + width / 2, hard_means, width, yerr=hard_stds,
                               label="hard", color="#F44336", alpha=0.8,
                               capsize=4, error_kw={"linewidth": 1})

            ax.set_xticks(x)
            ax.set_xticklabels([f"|S|={sz}" for sz in sizes])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{bench.upper()} / {sampler.upper()}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            if SAMPLER_ROW[sampler] == 1:
                ax.set_xlabel("Subset Size", fontsize=10)
            if BENCH_COL[bench] == 0:
                ax.set_ylabel("Stability Proportion", fontsize=10)

    fig.suptitle("Goal 1: Stability Proportion by Subset Size", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "goal1_stability_by_size.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts()}] Saved {path}")


# ── Figure 3: L_S* Histogram ───────────────────────────────────────────
def fig_loss_histogram():
    """Distribution of L_S* (final pretrain loss) per (bench, sampler)."""
    print(f"[{ts()}] Generating L_S* histogram figure...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    SAMPLER_ROW = {"as": 0, "asbs": 1}
    BENCH_COL = {"w5": 0, "c5": 1, "b5": 2}
    COLOR = {"easy": "#2196F3", "hard": "#F44336"}

    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            ax = axes[SAMPLER_ROW[sampler]][BENCH_COL[bench]]

            easy_losses = []
            hard_losses = []

            for s, sname in zip(ALL_SUBSETS, ALL_SUBSET_NAMES):
                loss = load_pretrain_loss(sampler, bench, sname)
                if np.isnan(loss):
                    continue
                eh = classify_easy_hard(s)
                if eh == "easy":
                    easy_losses.append(loss)
                else:
                    hard_losses.append(loss)

            if easy_losses:
                ax.hist(easy_losses, bins=15, color=COLOR["easy"], alpha=0.6,
                        label=f"easy (n={len(easy_losses)})", edgecolor="k", linewidth=0.4)
            if hard_losses:
                ax.hist(hard_losses, bins=15, color=COLOR["hard"], alpha=0.6,
                        label=f"hard (n={len(hard_losses)})", edgecolor="k", linewidth=0.4)

            ax.set_title(f"{bench.upper()} / {sampler.upper()}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            if SAMPLER_ROW[sampler] == 1:
                ax.set_xlabel(r"$L_{S}^*$ (final pretrain loss)", fontsize=10)
            if BENCH_COL[bench] == 0:
                ax.set_ylabel("Count", fontsize=10)

    fig.suptitle(r"Goal 1: Distribution of $L_S^*$ Values (30 Subsets per Panel)", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "goal1_loss_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts()}] Saved {path}")


# ── Figure 4: Noise Ablation ───────────────────────────────────────────
def fig_noise_ablation():
    """MWD at final epoch vs noise level, grouped by easy/hard subsets."""
    print(f"[{ts()}] Generating noise ablation figure...")

    # Collect data: for each (bench, sampler, subset, noise_level),
    # get mean final MWD over seeds
    records = []
    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            for s, sname in zip(ALL_SUBSETS, ALL_SUBSET_NAMES):
                eh = classify_easy_hard(s)
                for noise in NOISE_LEVELS:
                    suffix = NOISE_SUFFIX[noise]
                    final_mwds = []
                    for seed in SEEDS:
                        mwd_series, cls = load_stability_run(
                            sampler, bench, sname, seed,
                            noise_suffix=suffix
                        )
                        fm = final_mwd(mwd_series)
                        if not np.isnan(fm):
                            final_mwds.append(fm)
                    if final_mwds:
                        records.append({
                            "benchmark": bench,
                            "sampler": sampler,
                            "subset": sname,
                            "easy_hard": eh,
                            "noise": noise,
                            "mean_final_mwd": np.mean(final_mwds),
                            "std_final_mwd": np.std(final_mwds),
                            "n_seeds": len(final_mwds),
                        })

    if len(records) == 0:
        print(f"[{ts()}] WARNING: No noise ablation data found; skipping figure.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    SAMPLER_ROW = {"as": 0, "asbs": 1}
    BENCH_COL = {"w5": 0, "c5": 1, "b5": 2}
    COLOR = {"easy": "#2196F3", "hard": "#F44336"}
    MARKER = {"easy": "o", "hard": "s"}

    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            ax = axes[SAMPLER_ROW[sampler]][BENCH_COL[bench]]
            sub = df[(df["benchmark"] == bench) & (df["sampler"] == sampler)]

            for eh in ["easy", "hard"]:
                eh_sub = sub[sub["easy_hard"] == eh]
                if eh_sub.empty:
                    continue

                # Aggregate over all subsets in this (bench, sampler, eh) group
                grp = eh_sub.groupby("noise").agg(
                    mean_mwd=("mean_final_mwd", "mean"),
                    std_mwd=("mean_final_mwd", "std"),
                    n=("mean_final_mwd", "count"),
                ).reset_index()

                ax.errorbar(
                    grp["noise"], grp["mean_mwd"],
                    yerr=grp["std_mwd"],
                    fmt=MARKER[eh] + "-",
                    color=COLOR[eh],
                    label=eh,
                    capsize=4,
                    linewidth=1.5,
                    markersize=7,
                )

            ax.axhline(MWD_TOL, color="green", linestyle="--", linewidth=0.8,
                       alpha=0.7, label=f"stable ({MWD_TOL})")
            ax.axhline(MWD_ESCAPE, color="orange", linestyle="--", linewidth=0.8,
                       alpha=0.7, label=f"escape ({MWD_ESCAPE})")

            ax.set_xscale("symlog", linthresh=0.005)
            ax.set_xticks(NOISE_LEVELS)
            ax.set_xticklabels([str(n) for n in NOISE_LEVELS], fontsize=8)
            ax.set_title(f"{bench.upper()} / {sampler.upper()}", fontsize=11)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            if SAMPLER_ROW[sampler] == 1:
                ax.set_xlabel("Noise Level", fontsize=10)
            if BENCH_COL[bench] == 0:
                ax.set_ylabel("Mean Final MWD", fontsize=10)

    fig.suptitle("Goal 1: Noise Ablation — Final MWD vs Noise Level (easy vs hard subsets)", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "goal1_noise_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts()}] Saved {path}")
    return df


# ── Metrics JSON ───────────────────────────────────────────────────────
def build_metrics(stability_df):
    """Compute aggregate metrics and return as dict."""
    metrics = {}

    for bench in BENCHMARKS:
        for sampler in SAMPLERS:
            key = f"{sampler}_{bench}"
            sub = stability_df[
                (stability_df["benchmark"] == bench) &
                (stability_df["sampler"] == sampler)
            ]

            n_total = len(sub)
            n_stable_majority = int((sub["p_stab"] >= 0.5).sum())
            n_unstable = int((sub["p_stab"] < 0.5).sum())
            n_missing = int(sub["n_missing"].sum())

            easy_sub = sub[sub["easy_hard"] == "easy"]
            hard_sub = sub[sub["easy_hard"] == "hard"]

            metrics[key] = {
                "n_subsets": n_total,
                "n_stable_majority": n_stable_majority,
                "n_unstable_majority": n_unstable,
                "n_missing_seeds": n_missing,
                "overall_p_stab": round(float(sub["p_stab"].mean()), 4),
                "easy_p_stab_mean": round(float(easy_sub["p_stab"].mean()), 4) if len(easy_sub) > 0 else None,
                "hard_p_stab_mean": round(float(hard_sub["p_stab"].mean()), 4) if len(hard_sub) > 0 else None,
                "p_stab_by_size": {
                    str(sz): round(float(sub[sub["S_size"] == sz]["p_stab"].mean()), 4)
                    for sz in sorted(sub["S_size"].unique())
                },
            }

    return metrics


def compute_falsification_flags(metrics):
    """Print falsification check results for Goal 1."""
    print(f"\n[{ts()}] Falsification Checks:")
    for key, m in metrics.items():
        flags = []
        if m["overall_p_stab"] < 0.5:
            flags.append("F1.1 TRIGGERED: overall p_stab < 0.5")
        easy = m.get("easy_p_stab_mean")
        hard = m.get("hard_p_stab_mean")
        if easy is not None and hard is not None and easy <= hard:
            flags.append("F1.2 TRIGGERED: easy p_stab <= hard p_stab")
        if m["n_missing_seeds"] > 50:
            flags.append(f"F1.3 WARNING: {m['n_missing_seeds']} missing seeds")
        if flags:
            for fl in flags:
                print(f"  ! {key}: {fl}")
        else:
            print(f"  OK {key}: no falsification flags")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"GOAL 1: STABILITY ANALYSIS   [{ts()}]")
    print("=" * 60)

    # Table 1: Stability matrix
    stability_df = build_stability_matrix()

    # Figure 1: MWD trajectories
    fig_mwd_trajectories(stability_df)

    # Figure 2: Stability by subset size
    fig_stability_by_size(stability_df)

    # Figure 3: L_S* histogram
    fig_loss_histogram()

    # Figure 4: Noise ablation
    noise_df = fig_noise_ablation()

    # Metrics JSON
    print(f"\n[{ts()}] Computing metrics...")
    metrics = build_metrics(stability_df)
    with open(MET_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{ts()}] Saved {MET_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, m in metrics.items():
        print(
            f"  {key}: overall_p_stab={m['overall_p_stab']:.3f}, "
            f"stable={m['n_stable_majority']}/{m['n_subsets']}, "
            f"easy={m['easy_p_stab_mean']}, hard={m['hard_p_stab_mean']}"
        )

    compute_falsification_flags(metrics)

    print(f"\n[{ts()}] Done.")


if __name__ == "__main__":
    main()
