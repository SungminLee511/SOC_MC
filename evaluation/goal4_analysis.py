#!/usr/bin/env python3
"""Goal 4: L5 Consequence Test.

For each checkpoint, sample random parameter perturbation directions and measure:
  P1 = E[∫|δu|² dt]  (controller variation)
  δα = mode weight change for dead modes

Tests the theoretical lower bound: P1 >= c_u * |δα|²

Outputs:
  evaluation/tables/goal4_l5_results.csv   (per-direction results)
  evaluation/tables/goal4_summary.csv      (per-checkpoint summary)
  evaluation/figures/goal4_P1_vs_delta_alpha.png
  evaluation/figures/goal4_cu_distribution.png
  evaluation/goal4_metrics.json
"""

import os, sys, json, time, traceback
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.train import load_sampler, load_benchmark, evaluate_mode_weights
from model.sde import sdeint
from evaluation.utils.hessian_ops import unflatten_params, flatten_params

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
TBL_DIR = ROOT / "evaluation" / "tables"
FIG_DIR = ROOT / "evaluation" / "figures"
CKPT_DIR = ROOT / "result" / "checkpoints"
CACHE_DIR = TBL_DIR / "goal4_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_DIR = 200          # random directions per checkpoint
EPSILON_PERT = 0.05  # perturbation scale
N_TRAJ = 1000        # trajectories for P1
TRAJ_BATCH = 256     # batch size for trajectory sampling
N_EVAL = 10000       # samples for mode weight estimation
EVAL_BATCH = 2048    # batch for mode weight sampling
DEAD_THRESH = 0.005  # revival threshold for |δα|_max on dead modes
G_MAX = 3.0          # sigma_max from SDE config


def ts():
    return time.strftime("%H:%M:%S")


def load_checkpoint(sampler_name, bench, subset_name, device):
    """Load sampler at pretrain checkpoint."""
    model_cfg = str(ROOT / "model" / "configs" / f"{sampler_name}_default.yaml")
    bench_cfg = str(ROOT / "benchmark" / "configs" / f"{bench}.yaml")

    sampler, _ = load_sampler(model_cfg)
    sampler = sampler.to(device)

    ckpt_path = (CKPT_DIR /
                 f"goal1_{sampler_name}_{bench}_{subset_name}_pretrain_seed0" /
                 "epoch_500.pt")
    if not ckpt_path.exists():
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sampler.load_state_dict(ckpt)
    sampler.eval()

    energy_fn = load_benchmark(bench_cfg)
    energy_fn = energy_fn.to(device) if hasattr(energy_fn, 'to') else energy_fn
    return sampler, energy_fn


def get_dead_modes(alpha, target_weights, threshold=0.01):
    """Return indices of dead modes (alpha_k < threshold)."""
    dead = []
    for k in range(len(alpha)):
        if alpha[k] < threshold:
            dead.append(k)
    return dead


def compute_alpha(sampler, energy_fn, device):
    """Compute mode weights from sampler samples."""
    alpha_list, mwd = evaluate_mode_weights(sampler, energy_fn, N_EVAL, device)
    return np.array(alpha_list)


def random_unit_direction(params, device):
    """Generate a random unit-norm direction in parameter space."""
    v_parts = []
    for p in params:
        v_parts.append(torch.randn_like(p.data))
    # Flatten, normalize, unflatten
    v_flat = torch.cat([v.reshape(-1) for v in v_parts])
    v_flat = v_flat / (v_flat.norm() + 1e-12)
    return v_flat


def compute_P1(sampler, v_flat, device, epsilon):
    """Compute P1 = E[∫|δu/δε|² dt] along direction v.

    Samples trajectories from current policy, evaluates controller at
    original and perturbed parameters, integrates squared difference.
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

        # Sample trajectories at current theta
        with torch.no_grad():
            traj = sampler.sample_trajectories(bs, device)

        timesteps = traj.timesteps   # (T,)
        states = traj.states         # list of T tensors, each (B, D)
        T_steps = len(timesteps) - 1
        dt_vals = timesteps[1:] - timesteps[:-1]  # (T-1,)

        # u_theta at each (x_t, t)
        u_base_list = []
        with torch.no_grad():
            for i in range(T_steps):
                t_i = timesteps[i]
                x_i = states[i]
                u_i = sampler.controller(t_i, x_i)
                u_base_list.append(u_i)

        # Perturb: theta + eps * v
        for p, dv in zip(params, v_list):
            p.data.add_(dv, alpha=epsilon)

        # u_{theta+eps*v}
        u_pert_list = []
        with torch.no_grad():
            for i in range(T_steps):
                t_i = timesteps[i]
                x_i = states[i]
                u_i = sampler.controller(t_i, x_i)
                u_pert_list.append(u_i)

        # Restore theta
        for p, orig in zip(params, original):
            p.data.copy_(orig)

        # Integrate |δu|² dt
        batch_p1 = 0.0
        for i in range(T_steps):
            delta_u = (u_pert_list[i] - u_base_list[i]) / epsilon
            du_sq = (delta_u ** 2).sum(dim=-1)  # (B,)
            batch_p1 += (du_sq * dt_vals[i]).mean().item()

        total_p1 += batch_p1 * bs
        n_total += bs

    return total_p1 / n_total


def compute_perturbed_alpha(sampler, energy_fn, v_flat, device, epsilon):
    """Compute mode weights after perturbing params by eps*v."""
    params = list(sampler.parameters())
    shapes = [p.shape for p in params]
    original = [p.data.clone() for p in params]
    v_list = unflatten_params(v_flat, shapes)

    # Perturb
    for p, dv in zip(params, v_list):
        p.data.add_(dv, alpha=epsilon)

    alpha_pert = compute_alpha(sampler, energy_fn, device)

    # Restore
    for p, orig in zip(params, original):
        p.data.copy_(orig)

    return alpha_pert


def process_checkpoint(sampler_name, bench, subset_name, device):
    """Process one checkpoint: N_DIR random directions.

    Returns list of per-direction result dicts, or None on failure.
    """
    cache_path = CACHE_DIR / f"{sampler_name}_{bench}_{subset_name}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"      CACHED ({len(cached)} dirs)")
        return cached

    sampler, energy_fn = load_checkpoint(sampler_name, bench, subset_name, device)
    if sampler is None:
        print(f"      SKIP: checkpoint not found")
        return None

    # Baseline alpha
    alpha_base = compute_alpha(sampler, energy_fn, device)
    target_w = energy_fn.mode_weights.cpu().numpy()
    dead_modes = get_dead_modes(alpha_base, target_w)
    n_dead = len(dead_modes)
    print(f"      alpha_base={[f'{a:.3f}' for a in alpha_base]}, "
          f"dead_modes={dead_modes}")

    if n_dead == 0:
        print(f"      SKIP: no dead modes")
        # Still save empty cache so we don't re-check
        with open(cache_path, "w") as f:
            json.dump([], f)
        return []

    results = []
    params = list(sampler.parameters())

    for j in range(N_DIR):
        if (j + 1) % 50 == 0:
            print(f"      [{ts()}] dir {j+1}/{N_DIR}")

        v_flat = random_unit_direction(params, device)

        # P1
        P1 = compute_P1(sampler, v_flat, device, EPSILON_PERT)

        # Perturbed alpha
        alpha_pert = compute_perturbed_alpha(
            sampler, energy_fn, v_flat, device, EPSILON_PERT)

        # Delta alpha on dead modes
        delta_alpha = alpha_pert - alpha_base
        delta_alpha_dead = np.array([delta_alpha[k] for k in dead_modes])
        delta_alpha_dead_norm_sq = float(np.sum(delta_alpha_dead ** 2))
        delta_alpha_dead_max = float(np.max(np.abs(delta_alpha_dead)))
        is_revival = delta_alpha_dead_max > DEAD_THRESH

        results.append({
            "sampler": sampler_name,
            "benchmark": bench,
            "subset": subset_name,
            "dir_idx": j,
            "P1": float(P1),
            "delta_alpha_dead_norm_sq": delta_alpha_dead_norm_sq,
            "delta_alpha_dead_max": float(delta_alpha_dead_max),
            "is_revival": bool(is_revival),
        })

    # Cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    torch.cuda.empty_cache()
    return results


def compute_summary(all_results_df, dead_mode_counts):
    """Per-checkpoint summary: c_u_emp, c_u_theory, violation_frac."""
    summaries = []
    grouped = all_results_df.groupby(["sampler", "benchmark", "subset"])

    for (samp, bench, sub), grp in grouped:
        n_revival = int(grp["is_revival"].sum())
        n_total = len(grp)

        # c_u_emp = min(P1 / |δα|²) over revival directions
        revival = grp[grp["is_revival"] & (grp["delta_alpha_dead_norm_sq"] > 1e-15)]
        if len(revival) > 0:
            ratios = revival["P1"] / revival["delta_alpha_dead_norm_sq"]
            c_u_emp = float(ratios.min())
        else:
            c_u_emp = float("nan")

        # c_u_theory = 1 / (|S^c| * g_max²)
        n_dead = dead_mode_counts.get((samp, bench, sub), 1)
        c_u_theory = 1.0 / (n_dead * G_MAX ** 2)

        # Violation fraction: directions where P1 < c_u_theory * |δα|²
        if len(revival) > 0:
            violations = revival["P1"] < c_u_theory * revival["delta_alpha_dead_norm_sq"]
            violation_frac = float(violations.mean())
        else:
            violation_frac = float("nan")

        ratio = c_u_emp / c_u_theory if not np.isnan(c_u_emp) else float("nan")

        summaries.append({
            "sampler": samp,
            "benchmark": bench,
            "subset": sub,
            "n_dir": n_total,
            "n_revival": n_revival,
            "c_u_emp": c_u_emp,
            "c_u_theory": c_u_theory,
            "ratio_emp_over_theory": ratio,
            "violation_frac": violation_frac,
        })

    return pd.DataFrame(summaries)


def make_scatter_figure(df):
    """Scatter P1 vs |δα|² per benchmark, theory line overlaid."""
    revival = df[df["is_revival"] & (df["delta_alpha_dead_norm_sq"] > 1e-15)].copy()
    if len(revival) == 0:
        print("  No revival directions for scatter plot")
        return

    benchmarks = sorted(revival["benchmark"].unique())
    n_bench = len(benchmarks)
    fig, axes = plt.subplots(1, n_bench, figsize=(6 * n_bench, 5), squeeze=False)

    for idx, bench in enumerate(benchmarks):
        ax = axes[0, idx]
        sub = revival[revival["benchmark"] == bench]

        x = sub["delta_alpha_dead_norm_sq"].values
        y = sub["P1"].values

        # Color by sampler
        for samp, marker in [("as", "o"), ("asbs", "s")]:
            mask = sub["sampler"] == samp
            if mask.any():
                ax.scatter(x[mask], y[mask], marker=marker, alpha=0.4,
                           s=15, label=samp)

        # Theory line: P1 = c_u_theory * |δα|²
        # Use max n_dead across subsets for this bench
        # Simple: use n_dead=1 for tightest bound
        x_line = np.linspace(0, x.max() * 1.1, 100)
        for n_dead, ls in [(1, "-"), (2, "--"), (3, ":")]:
            c_u = 1.0 / (n_dead * G_MAX ** 2)
            ax.plot(x_line, c_u * x_line, ls, color="red", alpha=0.6,
                    label=f"theory |S$^c$|={n_dead}")

        ax.set_xlabel(r"$|\delta\alpha_{\mathrm{dead}}|^2$", fontsize=11)
        ax.set_ylabel(r"$P_1 = E[\int|\delta u|^2\,dt]$", fontsize=11)
        ax.set_title(f"Benchmark: {bench}", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Goal 4: P1 vs Dead-Mode Weight Change", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "goal4_P1_vs_delta_alpha.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {FIG_DIR / 'goal4_P1_vs_delta_alpha.png'}")


def make_cu_histogram(summary_df):
    """Histogram of c_u^emp values."""
    vals = summary_df["c_u_emp"].dropna()
    if len(vals) == 0:
        print("  No c_u_emp values for histogram")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=min(30, max(5, len(vals) // 2)), color="steelblue",
            edgecolor="white", alpha=0.8)

    # Mark theory values
    for n_dead in [1, 2, 3, 4]:
        c_th = 1.0 / (n_dead * G_MAX ** 2)
        ax.axvline(c_th, color="red", linestyle="--", alpha=0.6,
                   label=f"theory |S$^c$|={n_dead}: {c_th:.4f}")

    ax.set_xlabel(r"$c_u^{\mathrm{emp}}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(r"Goal 4: Distribution of $c_u^{\mathrm{emp}}$", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "goal4_cu_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {FIG_DIR / 'goal4_cu_distribution.png'}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")
    print("=" * 60)
    print("GOAL 4: L5 CONSEQUENCE TEST")
    print("=" * 60)

    # Load checkpoint list from goal3 summary
    summary_path = TBL_DIR / "goal3_spectral_summary.csv"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    total = len(summary)
    print(f"Checkpoints from goal3: {total}")
    print(f"Config: N_DIR={N_DIR}, eps={EPSILON_PERT}, N_TRAJ={N_TRAJ}, "
          f"N_EVAL={N_EVAL}, dead_thresh={DEAD_THRESH}")

    all_dir_results = []
    dead_mode_counts = {}

    for idx, row in summary.iterrows():
        samp = row["sampler"]
        bench = row["benchmark"]
        sname = row["subset"]

        print(f"\n  [{idx+1}/{total}] {samp}/{bench}/{sname}")

        try:
            results = process_checkpoint(samp, bench, sname, device)
            if results is not None and len(results) > 0:
                all_dir_results.extend(results)
                # Infer n_dead from results (all share same checkpoint)
                n_revival = sum(1 for r in results if r["is_revival"])
                print(f"      Revivals: {n_revival}/{len(results)}")

                # Count dead modes from cache or re-derive
                # Dead modes count stored implicitly: load checkpoint once more
                # to avoid, we estimate from delta_alpha dimension
                # Actually we can just load and check; but for efficiency
                # use a simpler heuristic or load once
                sampler, energy_fn = load_checkpoint(samp, bench, sname, device)
                if sampler is not None:
                    alpha_base = compute_alpha(sampler, energy_fn, device)
                    target_w = energy_fn.mode_weights.cpu().numpy()
                    n_dead = len(get_dead_modes(alpha_base, target_w))
                    dead_mode_counts[(samp, bench, sname)] = max(n_dead, 1)
                    del sampler, energy_fn
                    torch.cuda.empty_cache()
            elif results is not None:
                print(f"      No dead modes, skipped")
        except Exception as e:
            print(f"      ERROR: {e}")
            traceback.print_exc()

    # ── Save per-direction CSV ──
    if len(all_dir_results) == 0:
        print("\nNo results collected. Exiting.")
        sys.exit(0)

    df = pd.DataFrame(all_dir_results)
    l5_path = TBL_DIR / "goal4_l5_results.csv"
    df.to_csv(l5_path, index=False)
    print(f"\n[{ts()}] Saved: {l5_path} ({len(df)} rows)")

    # ── Per-checkpoint summary ──
    summary_df = compute_summary(df, dead_mode_counts)
    sum_path = TBL_DIR / "goal4_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"[{ts()}] Saved: {sum_path} ({len(summary_df)} rows)")

    # ── Figures ──
    make_scatter_figure(df)
    make_cu_histogram(summary_df)

    # ── Metrics JSON ──
    n_ckpt = len(summary_df)
    n_with_revival = int((summary_df["n_revival"] > 0).sum())
    revival_dirs = df[df["is_revival"]]
    cu_vals = summary_df["c_u_emp"].dropna()
    viol_vals = summary_df["violation_frac"].dropna()

    metrics = {
        "n_checkpoints": n_ckpt,
        "n_checkpoints_with_revival": n_with_revival,
        "total_directions": len(df),
        "total_revival_directions": len(revival_dirs),
        "revival_fraction": float(len(revival_dirs) / len(df)) if len(df) > 0 else 0,
        "c_u_emp_mean": float(cu_vals.mean()) if len(cu_vals) > 0 else None,
        "c_u_emp_median": float(cu_vals.median()) if len(cu_vals) > 0 else None,
        "c_u_emp_min": float(cu_vals.min()) if len(cu_vals) > 0 else None,
        "violation_frac_mean": float(viol_vals.mean()) if len(viol_vals) > 0 else None,
        "violation_frac_max": float(viol_vals.max()) if len(viol_vals) > 0 else None,
        "g_max": G_MAX,
        "epsilon_pert": EPSILON_PERT,
        "n_dir": N_DIR,
        "n_traj": N_TRAJ,
        "n_eval": N_EVAL,
        "dead_threshold": DEAD_THRESH,
    }

    met_path = ROOT / "evaluation" / "goal4_metrics.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{ts()}] Saved: {met_path}")

    # ── Console summary ──
    print("\n" + "=" * 60)
    print("GOAL 4 SUMMARY")
    print("=" * 60)
    print(f"  Checkpoints processed: {n_ckpt}")
    print(f"  Checkpoints with revival: {n_with_revival}")
    print(f"  Total directions: {len(df)}")
    print(f"  Revival directions: {len(revival_dirs)} "
          f"({100*len(revival_dirs)/max(len(df),1):.1f}%)")

    if len(cu_vals) > 0:
        print(f"  c_u^emp: mean={cu_vals.mean():.6f}, "
              f"median={cu_vals.median():.6f}, min={cu_vals.min():.6f}")
        print(f"  c_u^theory (|S^c|=1): {1/(G_MAX**2):.6f}")
        print(f"  Violation frac: mean={viol_vals.mean():.4f}, "
              f"max={viol_vals.max():.4f}")
    else:
        print("  No revival directions found — cannot compute c_u^emp")

    print(f"\n[{ts()}] Done.")


if __name__ == "__main__":
    main()
