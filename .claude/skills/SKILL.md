# SOC_MC — Loss-Landscape Theory Verification for Category (a) SOC Samplers

## Purpose
Verify theoretical predictions (P1-P6) about mode collapse in category (a) SOC samplers via 5 experimental goals on 2D GMM benchmarks (W5, C5, B5).

## File Tree

```
SOC_MC/
├── model/                    # Architectures + training
│   ├── __init__.py
│   ├── base.py               # CategoryASampler ABC (Trajectories, loss, sample)
│   ├── networks.py           # FourierMLP + TimeEmbed (controller network)
│   ├── sde.py                # VESDE + ControlledSDE + sdeint (Euler-Maruyama)
│   ├── configs/              # YAML model configs
├── benchmark/                # Energy functions + configs
│   ├── __init__.py
│   ├── base.py               # EnergyFunction ABC + RestrictedEnergyFunction + _gmm_energy()
│   ├── w5.py                 # W5Energy — geometric decay weights
│   ├── c5.py                 # C5Energy — heterogeneous covariances
│   ├── b5.py                 # B5Energy — cluster+isolated barrier layout
│   ├── configs/              # YAML benchmark configs
│   │   ├── w5.yaml
│   │   ├── c5.yaml
│   │   └── b5.yaml
├── result/                   # Checkpoints, logs, experiment configs
│   ├── experiments/          # One YAML per experiment
│   ├── checkpoints/          # Model checkpoints (subdirs)
│   └── logs/                 # Training CSV logs
├── evaluation/               # Metric/figure scripts
│   ├── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   ├── figures/              # Generated figures
│   └── tables/               # Generated tables
├── requirements.txt
└── README.md
```

## Dependencies
- torch>=2.0.0, numpy, pyyaml, scipy, matplotlib

## Key Docs
- `.claude/TODO/experimental_plan.md` -- 5-goal experimental program
- `.claude/TODO/codebase_implementation_guide.md` -- code structure spec
- `.claude/TODO/Category_a_soc_samplers.md` -- AS/ASBS algorithm details

## Porting Source
AS and ASBS implementations ported from `/home/sky/SML/Stein_ASBS/adjoint_samplers/`.

## Benchmarks
| Name | K | Axis | Key Feature |
|------|---|------|-------------|
| W5 | 5 | Weight imbalance | Geometric decay weights r=0.5 |
| C5 | 5 | Covariance heterogeneity | Wide/tight/anisotropic modes |
| B5 | 5 | Barrier heterogeneity | Cluster(1-3) + isolated(4,5) |

## Benchmark API (benchmark/base.py)
- `EnergyFunction(config)` — ABC, takes YAML config dict
- `.energy(x)` → (B,) — E(x) = -log p(x) + const
- `.gradient(x)` → (B, D) — ∇E via autograd
- `.score(x)` → (B, D) — -∇E
- `.restricted(subset_S)` → new EnergyFunction for p_S (0-indexed mode list)
- `.mode_assignment(x)` → (B,) int — nearest-center assignment
- `.get_ref_samples(n)` → (n, D) — ancestral sampling
- `.K`, `.dim`, `.mode_weights`, `.mode_centers` — properties
- `_gmm_energy()` — shared log-sum-exp GMM energy (handles diagonal + full cov)
- `RestrictedEnergyFunction` — wraps parent with renormalized subset weights

## Model API (model/)
- `CategoryASampler(config)` — ABC base, holds controller + VESDE + ControlledSDE
- `.sample_trajectories(batch_size)` → Trajectories (states list, timesteps, terminal)
- `.sample(batch_size)` → (B, D) terminal points only
- `.compute_target(trajs, energy_fn)` — abstract, subclass-specific
- `.loss(trajs, energy_fn)` → scalar L2 residual loss
- `FourierMLP(dim, hidden_dims, channels, activation, time_embed_dim)` — controller net
  - Forward: `(t, x) -> (B, dim)`, additive time+space embedding
- `VESDE(sigma_min, sigma_max)` — VE-SDE, g(t) = σ_min*(σ_max/σ_min)^(1-t)*sqrt(2*log(ratio))
- `ControlledSDE(ref_sde, controller)` — drift = b(t,x) + g²*u_θ(t,x)
- `sdeint(sde, state0, timesteps, return_all)` — Euler-Maruyama integrator

## Conventions
- Experiment config naming: `{goal}_{sampler}_{benchmark}_{subset}_{stage}_seed{n}.yaml`
- Subset encoding: S12 = {1,2}, Sfull = {1,...,K}
- CSV log format: epoch,event_type,loss,alpha_1,...,alpha_5,mwd,notes
- Seeds: torch + numpy + random seeded identically per run
- Eval seed: 12345 (fixed, independent of training seed)
