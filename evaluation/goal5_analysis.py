#!/usr/bin/env python3
"""Goal 5: Step-Size Escape — Post-training analysis.

For each (checkpoint, eta, seed) combination:
  1. Load stability CSV log
  2. Classify run: no_escape / escape / oscillation
  3. Compute P_escape(eta) per checkpoint
  4. Find critical eta* (smallest eta where P_escape > 0.5)
  5. Compare to theoretical eta_th = 2/lambda_max

Outputs:
  evaluation/tables/goal5_escape_results.csv
  evaluation/tables/goal5_escape_probability.csv
  evaluation/tables/goal5_critical_stepsize.csv
  evaluation/figures/goal5_escape_curves.png
  evaluation/figures/goal5_critical_comparison.png
  evaluation/goal5_metrics.json
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "result" / "logs"
FIG_DIR = ROOT / "evaluation" / "figures"
TBL_DIR = ROOT / "evaluation" / "tables"
MET_PATH = ROOT / "evaluation" / "goal5_metrics.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

K = 5
SEEDS = list(range(5))
ETA_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]

# Classification thresholds
MWD_NO_ESCAPE = 0.05    # no_escape if MWD < this throughout
MWD_ESCAPE = 0.15       # escape if MWD > this for 3 consecutive evals
MWD_OSCILLATION = 0.10  # oscillation if MWD crosses this multiple times
P_ESCAPE_THRESH = 0.5   # critical eta* threshold

SELECTION_CSV = TBL_DIR / "goal5_checkpoint_selection.csv"

# ── Target weight helpers (copied from goal2_analysis.py) ──────────────
def get_benchmark_weights(bench):
    """Return full target weight vector for each benchmark."""
    if bench == "w5":
        r = 0.5
        w = np.array([r**k for k in range(K)])
        return w / w.sum()
    else:  # c5, b5: equal weights
        return np.ones(K) / K


def get_collapsed_target(subset_indices, bench):
    """Return alpha^(S): target weights for S-collapsed state."""
    w = get_benchmark_weights(bench)
    alpha_S = np.zeros(K)
    for i in subset_indices:
        alpha_S[i] = w[i]
    if alpha_S.sum() > 0:
        alpha_S = alpha_S / alpha_S.sum()
    return alpha_S


def parse_subset_indices(sname):
    """'S0123' -> [0, 1, 2, 3]"""
    return [int(c) for c in sname[1:]]


def eta_mult_str(m):
    return f"{m:g}"


# ── Data loading & classification ──────────────────────────────────────
def load_and_classify(sampler, bench, subset, eta_mult, seed):
    """Load goal5 log CSV, recompute MWD vs S-collapsed target, classify."""
    m_str = eta_mult_str(eta_mult)
    tag = f"goal5_{sampler}_{bench}_{subset}_eta{m_str}_seed{seed}"
    fname = LOG_DIR / f"{tag}.csv"

    if not fname.exists():
        return None, "missing"

    try:
        df = pd.read_csv(fname)
    except Exception as e:
        print(f"  WARNING: Could not read {fname}: {e}")
        return None, "error"

    evals = df[df["event_type"] == "eval"].copy()
    if len(evals) == 0:
        return None, "no_evals"

    evals = evals.sort_values("epoch")
    alpha_cols = [f"alpha_{k+1}" for k in range(K)]

    # Recompute MWD against S-collapsed target
    subset_indices = parse_subset_indices(subset)
    alpha_S = get_collapsed_target(subset_indices, bench)
    alphas = evals[alpha_cols].values.astype(float)
    mwd_recomputed = np.abs(alphas - alpha_S[np.newaxis, :]).max(axis=1)
    evals = evals.copy()
    evals["mwd"] = mwd_recomputed

    mwd_vals = evals["mwd"].values

    # Classification
    # 1. no_escape: MWD < MWD_NO_ESCAPE throughout
    if (mwd_vals < MWD_NO_ESCAPE).all():
        return evals, "no_escape"

    # 2. escape: MWD > MWD_ESCAPE for 3 consecutive evals
    consec = 0
    for v in mwd_vals:
        if v > MWD_ESCAPE:
            consec += 1
            if consec >= 3:
                return evals, "escape"
        else:
            consec = 0

    # 3. oscillation: MWD crosses MWD_OSCILLATION multiple times
    above = mwd_vals > MWD_OSCILLATION
    crossings = np.sum(np.diff(above.astype(int)) != 0)
    if crossings >= 3:
        return evals, "oscillation"

    # Default: no_escape (didn't meet escape or oscillation criteria)
    return evals, "no_escape"


# ── Main analysis ──────────────────────────────────────────────────────
def main():
    if not SELECTION_CSV.exists():
        print(f"ERROR: {SELECTION_CSV} not found. Run goal5_generate_configs.py first.")
        sys.exit(1)

    sel = pd.read_csv(SELECTION_CSV)
    print(f"Loaded {len(sel)} selected checkpoints")

    # ── Per-run classification ──────────────────────────────────────────
    result_rows = []
    prob_rows = []
    crit_rows = []

    for idx, row in sel.iterrows():
        sampler = row["sampler"]
        bench = row["benchmark"]
        subset = row["subset"]
        lmax = row["lambda_max"]
        eta_0 = row["eta_0"]
        eta_th = row["eta_th"]
        ckpt_label = f"{sampler}/{bench}/{subset}"

        print(f"Processing {ckpt_label} (lambda_max={lmax:.1f}, eta_0={eta_0:.6f}, eta_th={eta_th:.6f})")

        eta_star = None

        for mult in ETA_MULTS:
            eta_val = mult * eta_0
            classifications = []

            for seed in SEEDS:
                _, cls = load_and_classify(sampler, bench, subset, mult, seed)
                classifications.append(cls)
                result_rows.append({
                    "sampler": sampler,
                    "benchmark": bench,
                    "subset": subset,
                    "eta_mult": mult,
                    "eta_value": eta_val,
                    "seed": seed,
                    "classification": cls,
                })

            n_escape = sum(1 for c in classifications if c == "escape")
            n_no_escape = sum(1 for c in classifications if c == "no_escape")
            n_oscillation = sum(1 for c in classifications if c == "oscillation")
            n_valid = sum(1 for c in classifications if c not in ("missing", "error", "no_evals"))
            p_escape = n_escape / n_valid if n_valid > 0 else np.nan

            prob_rows.append({
                "sampler": sampler,
                "benchmark": bench,
                "subset": subset,
                "eta_mult": mult,
                "eta_value": eta_val,
                "n_escape": n_escape,
                "n_no_escape": n_no_escape,
                "n_oscillation": n_oscillation,
                "n_valid": n_valid,
                "P_escape": p_escape,
            })

            # Track smallest eta where P_escape > threshold
            if eta_star is None and p_escape > P_ESCAPE_THRESH:
                eta_star = eta_val

        crit_rows.append({
            "sampler": sampler,
            "benchmark": bench,
            "subset": subset,
            "lambda_max": lmax,
            "eta_0": eta_0,
            "eta_th": eta_th,
            "eta_star": eta_star if eta_star is not None else np.nan,
            "ratio": (eta_star / eta_th) if eta_star is not None else np.nan,
        })

    # ── Save tables ─────────────────────────────────────────────────────
    res_df = pd.DataFrame(result_rows)
    res_path = TBL_DIR / "goal5_escape_results.csv"
    res_df.to_csv(res_path, index=False)
    print(f"Saved: {res_path} ({len(res_df)} rows)")

    prob_df = pd.DataFrame(prob_rows)
    prob_path = TBL_DIR / "goal5_escape_probability.csv"
    prob_df.to_csv(prob_path, index=False)
    print(f"Saved: {prob_path} ({len(prob_df)} rows)")

    crit_df = pd.DataFrame(crit_rows)
    crit_path = TBL_DIR / "goal5_critical_stepsize.csv"
    crit_df.to_csv(crit_path, index=False)
    print(f"Saved: {crit_path} ({len(crit_df)} rows)")

    # ── Figures ─────────────────────────────────────────────────────────
    _plot_escape_curves(prob_df, sel)
    _plot_critical_comparison(crit_df)

    # ── Metrics JSON ────────────────────────────────────────────────────
    n_with_star = crit_df["eta_star"].notna().sum()
    metrics = {
        "n_checkpoints": len(sel),
        "n_total_runs": len(res_df),
        "n_missing": int((res_df["classification"] == "missing").sum()),
        "n_escape": int((res_df["classification"] == "escape").sum()),
        "n_no_escape": int((res_df["classification"] == "no_escape").sum()),
        "n_oscillation": int((res_df["classification"] == "oscillation").sum()),
        "n_checkpoints_with_eta_star": int(n_with_star),
        "mean_ratio_eta_star_over_eta_th": float(crit_df["ratio"].mean()) if n_with_star > 0 else None,
        "median_ratio_eta_star_over_eta_th": float(crit_df["ratio"].median()) if n_with_star > 0 else None,
    }
    with open(MET_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {MET_PATH}")
    print(f"Summary: {n_with_star}/{len(sel)} checkpoints have eta*, "
          f"mean ratio={metrics['mean_ratio_eta_star_over_eta_th']}")


# ── Plotting ────────────────────────────────────────────────────────────
def _plot_escape_curves(prob_df, sel):
    """P_escape vs eta/eta_0 per checkpoint, vertical line at eta_th."""
    checkpoints = sel[["sampler", "benchmark", "subset", "eta_0", "eta_th"]].drop_duplicates()
    n_ckpts = len(checkpoints)

    if n_ckpts == 0:
        print("  No checkpoints to plot escape curves.")
        return

    # Group by (sampler, benchmark) for subplot layout
    groups = checkpoints.groupby(["sampler", "benchmark"])
    n_groups = len(groups)
    ncols = min(3, n_groups)
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, ((sampler, bench), grp) in enumerate(groups):
        ax = axes_flat[ax_idx]
        for _, ckpt in grp.iterrows():
            subset = ckpt["subset"]
            eta_0 = ckpt["eta_0"]
            eta_th = ckpt["eta_th"]
            eta_th_mult = eta_th / eta_0  # = 2.0 always

            mask = ((prob_df["sampler"] == sampler) &
                    (prob_df["benchmark"] == bench) &
                    (prob_df["subset"] == subset))
            sub = prob_df[mask].sort_values("eta_mult")

            if len(sub) == 0:
                continue

            ax.plot(sub["eta_mult"], sub["P_escape"], "o-", label=subset, markersize=4)
            ax.axvline(eta_th_mult, color="red", linestyle="--", alpha=0.5)

        ax.set_xlabel("eta / eta_0")
        ax.set_ylabel("P_escape")
        ax.set_title(f"{sampler} / {bench}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(ax_idx + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Goal 5: Escape Probability vs Step-Size", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "goal5_escape_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_critical_comparison(crit_df):
    """Scatter: eta* vs eta_th."""
    valid = crit_df.dropna(subset=["eta_star"])

    fig, ax = plt.subplots(figsize=(6, 5))

    if len(valid) == 0:
        ax.text(0.5, 0.5, "No checkpoints with eta*\n(all runs may be missing)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        # Color by sampler
        for sampler, grp in valid.groupby("sampler"):
            ax.scatter(grp["eta_th"], grp["eta_star"], label=sampler, s=60, alpha=0.7)

        # Identity line
        all_vals = np.concatenate([valid["eta_th"].values, valid["eta_star"].values])
        lo, hi = all_vals.min() * 0.8, all_vals.max() * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="eta* = eta_th")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_xlabel("eta_th = 2 / lambda_max")
        ax.set_ylabel("eta* (empirical critical step-size)")
        ax.legend()

    ax.set_title("Goal 5: Critical Step-Size Comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "goal5_critical_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
