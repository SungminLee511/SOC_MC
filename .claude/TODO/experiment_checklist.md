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
| W5        | AS      | [ ]      | [ ]       | [ ]       | [ ]      |
| W5        | ASBS    | [ ]      | [ ]       | [ ]       | [ ]      |
| C5        | AS      | [ ]      | [ ]       | [ ]       | [ ]      |
| C5        | ASBS    | [ ]      | [ ]       | [ ]       | [ ]      |
| B5        | AS      | [ ]      | [ ]       | [ ]       | [ ]      |
| B5        | ASBS    | [ ]      | [ ]       | [ ]       | [ ]      |

### Stage 2: Stability runs on full target (200 epochs, 5 seeds)
30 subsets x 3 benchmarks x 2 samplers x 5 seeds = 900 stability runs

| Benchmark | Sampler | Status |
|-----------|---------|--------|
| W5        | AS      | [ ]    |
| W5        | ASBS    | [ ]    |
| C5        | AS      | [ ]    |
| C5        | ASBS    | [ ]    |
| B5        | AS      | [ ]    |
| B5        | ASBS    | [ ]    |

### Stage 3: Ablations
- [ ] 1.1 Initialization noise (eta = 0, 0.01, 0.05, 0.1)
- [ ] 1.2 Mode separation (rho_sep = 3, 4, 5, 7)
- [ ] 1.3 Cross-sampler consistency

---

## Goal 2: Predictivity Ranking (P2)
*No new training — re-analysis of Goal 1 data*

- [ ] Ranked subset table per (benchmark, sampler)
- [ ] Predictivity scatter (L_S* vs p_stab)
- [ ] Threshold detection (classification accuracy sweep)
- [ ] Spearman rank correlation
- [ ] Per-|S| stratification
- [ ] Cross-sampler comparison table
- [ ] evaluation/goal2_analysis.py

---

## Goal 3: Hessian Spectral Check (P3, P4)
~30 checkpoints per sampler from Goal 1

- [ ] evaluation/utils/hessian_ops.py (HVP, Lanczos)
- [ ] Stage 1: Extreme eigenvalues (Lanczos T=50)
- [ ] Stage 2: Eigenvector analysis (revival vs surviving classification)
- [ ] Stage 3: Revival-subspace projection
- [ ] Stage 4: Five-term decomposition (P1, P2, I1, I2, I3)
- [ ] evaluation/goal3_analysis.py

---

## Goal 4: L5 Consequence Test (P5)
Same ~30 checkpoints as Goal 3

- [ ] Stage 1: 200 random perturbation directions per checkpoint
- [ ] Stage 2: Per-perturbation P1 and delta_alpha measurement
- [ ] Stage 3: Scatter + fit (c_u^emp vs c_u^theory)
- [ ] evaluation/goal4_analysis.py

---

## Goal 5: Step-Size Escape (P6)
9 stable checkpoints per sampler (3 per benchmark)

- [ ] Stage 1: Step-size sweep (0.5x to 6.0x eta_0, 8 values)
- [ ] Stage 2: Continuation training (300 epochs, 5 seeds per pair)
- [ ] Stage 3: Escape classification
- [ ] Stage 4: Critical step-size vs theoretical comparison
- [ ] Ablations: optimizer (SGD/momentum/Adam), schedule, batch size
- [ ] evaluation/goal5_analysis.py

---

## Evaluation & Reporting

- [ ] evaluation/goal1_analysis.py (stability matrix, MWD trajectories)
- [ ] evaluation/goal2_analysis.py (predictivity ranking)
- [ ] evaluation/goal3_analysis.py (Hessian spectral)
- [ ] evaluation/goal4_analysis.py (L5 consequence)
- [ ] evaluation/goal5_analysis.py (step-size escape)
- [ ] evaluation/RESULTS.md

---

## Compute Budget Summary (per sampler)

| Goal | Epochs | Status |
|------|--------|--------|
| 1 (pretrain) | 45,000 (30 subsets x 500 x 3 bench) | [ ] |
| 1 (stability) | 90,000 (30 x 200 x 5 seeds x 3 bench) | [ ] |
| 2 | 0 (analysis only) | [ ] |
| 3 | ~cheap (30 spectral analyses) | [ ] |
| 4 | moderate (30 ckpts x 200 dirs x MC) | [ ] |
| 5 | 108,000 (9 ckpts x 8 eta x 5 seeds x 300) | [ ] |
| **Total** | **~243,000 training epochs + analysis** | |

x2 samplers (AS + ASBS) = ~486,000 total training epochs
