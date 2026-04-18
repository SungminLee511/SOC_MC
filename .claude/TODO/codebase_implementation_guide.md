# Codebase Implementation Guide

This document specifies the codebase structure for implementing the experimental program described in `experimental_plan.md` and `category_a_samplers.md`. It is intended to be read together with those two documents by an implementer (human or Claude Code). This document focuses exclusively on *how the code is organized* — the *what* is in the experimental plan, the *which algorithms* is in the samplers markdown.

-----

## Top-Level Folder Structure

```
project_root/
├── model/           # Architectures + shared trainer + model configs
├── benchmark/       # Energy functions, gradients, benchmark configs
├── result/          # Checkpoints, logs, experiment configs
└── evaluation/      # Metric/figure scripts + results tracking markdown
```

Four folders. Each has a single well-defined responsibility; no cross-folder logic spillover.

-----

## `model/` — Architectures and Training

### Contents

```
model/
├── base.py              # CategoryASampler abstract base class
├── adjoint_sampling.py  # AS implementation
├── asbs.py              # ASBS implementation
├── train.py             # Single shared training script
└── configs/             # YAML model configs
    ├── as_default.yaml
    └── asbs_default.yaml
```

### Base Class Interface

All category (a) samplers inherit from a common base class defined in `base.py`. The base class exposes:

```python
class CategoryASampler:
    def __init__(self, config: dict): ...

    def sample_trajectories(self, batch_size: int) -> Trajectories:
        """Generate trajectories from current policy Q_theta."""
        ...

    def compute_target(self, trajectories: Trajectories, energy_fn) -> Tensor:
        """Compute u*_target along trajectories using current parameters."""
        ...

    def loss(self, trajectories: Trajectories, energy_fn) -> Tensor:
        """Compute L2 residual loss between controller and target."""
        ...

    def parameters(self) -> Iterable[Tensor]:
        """Return trainable parameters."""
        ...

    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict): ...
```

Each specific sampler (AS, ASBS) implements these methods using its particular target formulation. The training script does not need to know which sampler it is working with — it only calls the base-class interface.

### Shared Training Script

`train.py` is the single entry point for all training runs. Its signature:

```bash
python model/train.py --exp_config <path_to_experiment_config>
```

The experiment config (loaded from `result/experiments/`) specifies:

- Which sampler to use (loads model config).
- Which benchmark to use (loads benchmark config).
- Which subset $S$ to target (or null for full target).
- Number of training epochs.
- Random seed.
- Output directory for checkpoints and logs.
- Stage: `"pretrain"`, `"stability"`, or `"stepsize"`.

The training loop is the same for all samplers and stages: sample trajectories → compute loss → backprop → next epoch. No mid-training evaluation; checkpoints are saved at intervals specified in the experiment config.

### Model Config Files

Model YAMLs specify sampler-specific hyperparameters: network depth, width, activation, time embedding type, optimizer, learning rate, batch size, diffusion schedule parameters. These are kept minimal and inherit from sampler reference implementations where possible.

Example `as_default.yaml`:

```yaml
sampler: adjoint_sampling
controller:
  hidden_dims: [128, 128, 128, 128]
  activation: silu
  time_embed_dim: 64
diffusion:
  sigma_min: 0.01
  sigma_max: 3.0
  schedule: geometric
optimizer:
  name: adam
  lr: 1.0e-4
training:
  batch_size: 256
```

-----

## `benchmark/` — Targets

### Contents

```
benchmark/
├── base.py              # EnergyFunction abstract base class
├── w5.py                # W5 weight-heterogeneity benchmark
├── c5.py                # C5 covariance-heterogeneity benchmark
├── b5.py                # B5 barrier-heterogeneity benchmark
└── configs/             # YAML benchmark configs
    ├── w5.yaml
    ├── c5.yaml
    └── b5.yaml
```

### Base Class Interface

```python
class EnergyFunction:
    def __init__(self, config: dict): ...

    def energy(self, x: Tensor) -> Tensor:
        """Compute E(x) for x of shape (batch, dim)."""
        ...

    def gradient(self, x: Tensor) -> Tensor:
        """Compute nabla E(x)."""
        ...

    def restricted(self, subset_S: list[int]) -> 'EnergyFunction':
        """Return a new EnergyFunction corresponding to p_S."""
        ...

    def mode_assignment(self, x: Tensor) -> Tensor:
        """Return mode index k in {1, ..., K} for each point in x
        based on basin membership."""
        ...

    @property
    def K(self) -> int: ...
    @property
    def mode_weights(self) -> Tensor: ...  # (K,) tensor of w_k
```

`restricted(subset_S)` is the central construction enabling pre-training on $p_S$. It returns a new `EnergyFunction` object representing $-\log p_S$ with correctly-renormalized weights.

`mode_assignment(x)` is used downstream by evaluation scripts to estimate $\alpha_k^{(\theta)}$ from sampled terminal points.

### Benchmark Config Files

Benchmark YAMLs specify the target’s structural parameters: mode positions, covariances, weights, any structural parameters like pentagon radius or barrier positions.

Example `w5.yaml`:

```yaml
benchmark: W5
K: 5
dim: 2
mode_positions:
  type: regular_pentagon
  radius: 8.0
mode_covariance:
  type: isotropic
  sigma: 0.8
mode_weights:
  type: geometric_decay
  r: 0.5
```

-----

## `result/` — Checkpoints, Logs, Experiment Configs

### Contents

```
result/
├── experiments/         # One YAML per experiment (all 5 goals)
│   ├── goal1_as_w5_S12_pretrain_seed0.yaml
│   ├── goal1_as_w5_S12_stability_seed0.yaml
│   └── ...
├── checkpoints/         # Model checkpoints, one subdir per experiment
│   ├── goal1_as_w5_S12_pretrain_seed0/
│   │   ├── epoch_500.pt      # final checkpoint
│   │   └── config.yaml        # copy of experiment config for reference
│   └── ...
└── logs/                # Training logs, one CSV per experiment
    ├── goal1_as_w5_S12_pretrain_seed0.csv
    └── ...
```

### Experiment Config Naming Convention

Filename format: `{goal}_{sampler}_{benchmark}_{subset}_{stage}_seed{n}.yaml`

Where:

- `{goal}` ∈ `{goal1, goal2, goal3, goal4, goal5}`.
- `{sampler}` ∈ `{as, asbs, ...}`.
- `{benchmark}` ∈ `{w5, c5, b5}`.
- `{subset}` is a compact subset encoding: `S12` = ${1,2}$, `S245` = ${2,4,5}$, `Sfull` = full set ${1,…,K}$.
- `{stage}` ∈ `{pretrain, stability, stepsize, random}`.
- `{n}` is the seed.

Example: `goal1_as_w5_S12_pretrain_seed0.yaml` specifies pre-training Adjoint Sampling on the W5 benchmark restricted to ${1, 2}$, seed 0.

### Experiment Config Structure

```yaml
# Example: goal1_as_w5_S12_stability_seed0.yaml
goal: 1
stage: stability
seed: 0
n_epochs: 200

model:
  config: model/configs/as_default.yaml

benchmark:
  config: benchmark/configs/w5.yaml

subset:
  S: [1, 2]                      # the surviving subset
  full_target: false             # true for random-init runs

init:
  from_checkpoint: result/checkpoints/goal1_as_w5_S12_pretrain_seed0/epoch_500.pt
  noise_level: 0.0

logging:
  checkpoint_every: 40           # epochs
  eval_every: 20                 # epochs; triggers mode weight estimation
  n_eval_samples: 10000

output:
  checkpoint_dir: result/checkpoints/goal1_as_w5_S12_stability_seed0/
  log_file: result/logs/goal1_as_w5_S12_stability_seed0.csv
```

### Log Format

Plain CSV, one row per logged event:

```
epoch,event_type,loss,alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,mwd,notes
0,train,0.532,,,,,,,start
1,train,0.487,,,,,,,
...
20,eval,,0.498,0.497,0.003,0.001,0.001,0.003,
...
```

Event types:

- `train`: per-epoch training loss.
- `eval`: evaluation pass (mode weights + MWD from 10k-sample estimate).
- `checkpoint`: checkpoint saved (no metrics, just record).

-----

## `evaluation/` — Metrics, Figures, Results Tracking

### Contents

```
evaluation/
├── goal1_analysis.py        # Stability matrix, MWD trajectories, etc.
├── goal2_analysis.py        # Predictivity ranking, Spearman, thresholds
├── goal3_analysis.py        # Hessian spectral check via Lanczos
├── goal4_analysis.py        # L5 consequence test (P1 vs delta-alpha)
├── goal5_analysis.py        # Step-size escape curves
├── utils/                   # Shared utilities
│   ├── checkpoint_loader.py
│   ├── trajectory_sampler.py
│   ├── hessian_ops.py       # HVP, Lanczos
│   └── figure_templates.py
├── figures/                 # Generated figures (png/pdf)
├── tables/                  # Generated tables (markdown/csv)
└── RESULTS.md               # Living results document
```

### Goal Analysis Scripts

Each `goalN_analysis.py` is a self-contained script that:

1. Reads all checkpoints and logs from `result/` matching its goal pattern.
1. Computes the goal’s primary and secondary metrics as specified in `experimental_plan.md`.
1. Produces figures into `evaluation/figures/` and tables into `evaluation/tables/`.
1. Appends a summary section to `evaluation/RESULTS.md`.

Scripts are independent — running goal 3 doesn’t require goal 2 to have been run. Missing inputs produce clear error messages naming which experiments are needed.

### Results Tracking Markdown

`RESULTS.md` is a living document structured as:

```
# Experimental Results

## Goal 1: Stability Test
### Setup
[which samplers, benchmarks, subsets were run, with date]
### Findings
[Summary of stability matrix; reference to Table 1, Figure 1, etc.]
### Interpretation
[What this tells us about P1]

## Goal 2: Predictivity Ranking
...
```

Each goal’s section is appended (or overwritten on re-run) by that goal’s analysis script. Figures and tables are referenced by filename; readers open them separately.

-----

## Interface Conventions

### Checkpoint Loading

Checkpoints are loaded by matching the experiment config ID:

```python
from evaluation.utils import load_checkpoint

ckpt, config = load_checkpoint('goal1_as_w5_S12_pretrain_seed0', epoch='final')
# ckpt is a state_dict; config is the experiment YAML parsed into a dict
```

### Trajectory Sampling for Evaluation

Evaluation scripts that need to sample from a trained checkpoint use:

```python
from evaluation.utils import sample_trajectories

trajs = sample_trajectories(ckpt, config, n_samples=10000)
# Returns a tensor of terminal points for mode-weight estimation
```

This wraps the sampler’s own `sample_trajectories` method, ensuring consistency between training and evaluation.

### Hessian Operations (Goal 3)

```python
from evaluation.utils.hessian_ops import hvp, lanczos_extreme

H_v = hvp(ckpt, config, direction=v, batch_size=1024, n_avg=8)
# Returns Hessian-vector product for direction v

eigenvalues = lanczos_extreme(ckpt, config, n_iter=50, which='both')
# Returns dict with 'min' and 'max' eigenvalue estimates
```

-----

## Random Seed Conventions

Every experiment config has an explicit `seed` field. The training script seeds:

- PyTorch (`torch.manual_seed`)
- NumPy (`np.random.seed`)
- Python random (`random.seed`)

All with the same seed value at the start of each run. This ensures full reproducibility given the same config file.

Evaluation scripts use a separate fixed seed (e.g., 12345) for any sampling they do, independent of training seeds. This ensures evaluation results are reproducible even when re-run against the same checkpoints.

-----

## Running a Single Experiment

Minimum workflow:

```bash
# 1. Create experiment config
cp templates/goal1_template.yaml result/experiments/goal1_as_w5_S12_stability_seed0.yaml
# (edit as needed)

# 2. Run training
python model/train.py --exp_config result/experiments/goal1_as_w5_S12_stability_seed0.yaml

# 3. After many experiments complete, run analysis
python evaluation/goal1_analysis.py
# Populates evaluation/figures/, evaluation/tables/, and evaluation/RESULTS.md
```

For batch experiments, a shell loop over the config files in `result/experiments/` suffices.

-----

## What This Guide Does Not Specify

Deliberately left for the implementer:

- **Specific deep-learning framework** (PyTorch assumed implicitly but could be JAX; pick one and use it throughout).
- **Dependency management** (requirements.txt, conda env, etc.).
- **Logging tool choice** (CSV is recommended for simplicity; tensorboard/wandb/mlflow are acceptable if reproducibility is preserved).
- **Parallel/distributed training** (2D experiments are small enough that single-GPU is sufficient).
- **Specific PyTorch-vs-JAX architecture details for AS/ASBS** (see `category_a_samplers.md` for algorithmic specification; the specific reference implementation is left to the implementer).

The goal is a functional, reproducible codebase that can execute the full experimental program described in `experimental_plan.md` using the samplers described in `category_a_samplers.md`, with results accumulating in `evaluation/RESULTS.md`.