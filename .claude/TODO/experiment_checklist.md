# Experiment Checklist

Status: `[x]` done, `[-]` in progress, `[ ]` not started

---

## Infrastructure (Pre-Training)

- [x] Phase 1: Project skeleton, dirs, requirements.txt
- [x] Phase 2: Benchmark module (W5, C5, B5 + restricted() + configs)
- [x] Phase 3: Model base class, FourierMLP, VESDE, sdeint
- [x] Phase 4: AS + ASBS sampler implementations + configs
- [x] Phase 5: train.py, experiment config templates, eval utils
- [x] Phase 6: checkpoint_loader, trajectory_sampler (merged into Phase 5)

---

## Goal 1: Stability Test (P1)

### Stage 1: Pre-training on p_S (500 epochs per subset)
30 subsets x 3 benchmarks x 2 samplers = 180 pre-train runs

| Benchmark | Sampler | |S|=1 (5) | |S|=2 (10) | |S|=3 (10) | |S|=4 (5) |
|-----------|---------|----------|-----------|-----------|----------|
| W5        | AS      | [x] 5/5  | [x] 10/10 | [x] 10/10 | [x] 5/5  |
| W5        | ASBS    | [x] 5/5  | [x] 10/10 | [x] 10/10 | [x] 5/5  |
| C5        | AS      | [x] 5/5  | [x] 10/10 | [x] 10/10 | [x] 5/5  |
| C5        | ASBS    | [x] 5/5  | [x] 10/10 | [x] 10/10 | [x] 5/5  |
| B5        | AS      | [x] 5/5  | [x] 10/10 | [x] 10/10 | [x] 5/5  |
| B5        | ASBS    | [x] 5/5  | [x] 10/10 | [x] 10/10 | [x] 5/5  |

**All 180/180 Stage 1 pretrain runs complete (2026-04-18 17:40 KST)**

### Stage 2: Stability runs on full target (200 epochs, 5 seeds)
30 subsets x 3 benchmarks x 2 samplers x 5 seeds = 900 stability runs

| Benchmark | Sampler | Status |
|-----------|---------|--------|
| W5        | AS      | [x]    |
| W5        | ASBS    | [x]    |
| C5        | AS      | [x]    |
| C5        | ASBS    | [x]    |
| B5        | AS      | [x]    |
| B5        | ASBS    | [x]    |

**All 900/900 Stage 2 stability runs complete (2026-04-18 18:34 KST)**

### Stage 3: Ablations
- [x] 1.1 Initialization noise (eta = 0, 0.01, 0.05, 0.1) — 2700 runs (2026-04-18)
- [x] 1.2 Mode separation (rho_sep = 3, 4, 5, 7) — 240 pretrain + 1200 stability (2026-04-19 00:53 KST)
- [x] 1.3 Cross-sampler consistency — covered by running both AS + ASBS

---

## Goal 2: Predictivity Ranking (P2)
*No new training — re-analysis of Goal 1 data*

- [x] Ranked subset table per (benchmark, sampler) — 6 CSV tables
- [x] Predictivity scatter (L_S* vs p_stab) — 6 figures
- [x] Threshold detection (classification accuracy sweep) — 6 figures
- [x] Spearman rank correlation — computed per combo
- [x] Per-|S| stratification — 6 figures
- [x] Cross-sampler comparison table — 1 CSV + 1 figure
- [x] evaluation/goal2_analysis.py — complete (2026-04-19)

---

## Goal 3: Hessian Spectral Check (P3, P4)
55 checkpoints (both samplers) from Goal 1

- [x] evaluation/utils/hessian_ops.py (HVP, Lanczos)
- [x] Stage 1: Extreme eigenvalues (Lanczos T=50) — 55 checkpoints, 110 ritz files (2026-04-18)
- [x] Stage 2: Eigenvector analysis (revival vs surviving classification) — 551-row CSV + figure (2026-04-18)
- [x] Stage 3: Revival-subspace projection — 55 checkpoints, all revival_dim=0 (2026-04-19 03:00 KST)
- [x] Stage 4: Five-term decomposition (P1, total_vHv, remainder) — 55 checkpoints + bar chart (2026-04-19 03:00 KST)
- [x] evaluation/goal3_stage1-4.py — complete

---

## Goal 4: L5 Consequence Test (P5)
Same 55 checkpoints as Goal 3

- [x] Stage 1: 200 random perturbation directions per checkpoint — 11,000 total (2026-04-19 05:40 KST)
- [x] Stage 2: Per-perturbation P1 and delta_alpha measurement — 0 revival directions found
- [x] Stage 3: Scatter + fit — N/A (no revival directions at eps=0.05)
- [x] evaluation/goal4_analysis.py — complete (2026-04-19 05:40 KST)

---

## Goal 5: Step-Size Escape (P6)
13 stable checkpoints (both samplers, 3 benchmarks)

- [x] Stage 1: Step-size sweep (0.5x to 6.0x eta_0, 8 values) — 520 configs
- [x] Stage 2: Continuation training (300 epochs, 5 seeds per pair) — 520/520 runs, 0 failures (2026-04-19 12:08 KST)
- [x] Stage 3: Escape classification — 520 rows in escape_results.csv
- [x] Stage 4: Critical step-size vs theoretical comparison — 1/13 checkpoints with eta* (ratio=0.25)
- [ ] Ablations: optimizer (SGD/momentum/Adam), schedule, batch size
- [x] evaluation/goal5_analysis.py — complete (2026-04-19 12:08 KST)

---

## Evaluation & Reporting

- [x] evaluation/goal1_analysis.py (stability matrix, MWD trajectories) — complete (2026-04-19 12:15 KST)
- [x] evaluation/goal2_analysis.py (predictivity ranking)
- [x] evaluation/goal3_stage1-4.py (Hessian spectral)
- [x] evaluation/goal4_analysis.py (L5 consequence)
- [x] evaluation/goal5_analysis.py (step-size escape)
- [x] evaluation/RESULTS.md — complete (2026-04-19 12:20 KST)

---

## Compute Budget Summary (per sampler)

| Goal | Epochs | Status |
|------|--------|--------|
| 1 (pretrain) | 45,000 (30 subsets x 500 x 3 bench) | [x] |
| 1 (stability) | 90,000 (30 x 200 x 5 seeds x 3 bench) | [x] |
| 2 | 0 (analysis only) | [x] |
| 3 | ~cheap (55 spectral analyses) | [x] |
| 4 | moderate (55 ckpts x 200 dirs x MC) | [x] |
| 5 | 156,000 (13 ckpts x 8 eta x 5 seeds x 300) | [x] |
| **Total** | **~243,000 training epochs + analysis** | |

x2 samplers (AS + ASBS) = ~486,000 total training epochs
