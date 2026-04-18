# SOC_MC — Loss-Landscape Theory Verification for Category (a) SOC Samplers

## Purpose
Verify theoretical predictions (P1-P6) about mode collapse in category (a) SOC samplers via 5 experimental goals on 2D GMM benchmarks (W5, C5, B5).

## File Tree

```
SOC_MC/
├── model/                    # Architectures + training
│   ├── __init__.py
│   ├── configs/              # YAML model configs
├── benchmark/                # Energy functions + configs
│   ├── __init__.py
│   ├── configs/              # YAML benchmark configs
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

## Conventions
- Experiment config naming: `{goal}_{sampler}_{benchmark}_{subset}_{stage}_seed{n}.yaml`
- Subset encoding: S12 = {1,2}, Sfull = {1,...,K}
- CSV log format: epoch,event_type,loss,alpha_1,...,alpha_5,mwd,notes
- Seeds: torch + numpy + random seeded identically per run
- Eval seed: 12345 (fixed, independent of training seed)
