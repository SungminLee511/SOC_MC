#!/usr/bin/env python3
"""Generate 2700 YAML configs for Goal 1 Stage 3 Ablation 1.1 (initialization noise)."""

import os
from itertools import combinations

OUT_DIR = "/home/sky/SML/SOC_MC/result/experiments/"
os.makedirs(OUT_DIR, exist_ok=True)

noise_levels = [(0.01, "001"), (0.05, "005"), (0.1, "01")]
benchmarks = ["w5", "c5", "b5"]
samplers = ["as", "asbs"]
seeds = [0, 1, 2, 3, 4]
K = 5

# Build all 30 subsets for K=5
subsets = []
for size in range(1, K):
    for combo in combinations(range(K), size):
        name = "S" + "".join(str(c) for c in combo)
        flow = list(combo)
        subsets.append((name, flow))

TEMPLATE = """\
# Goal 1 Ablation 1.1: Init noise eta={noise} — {sampler_upper} on {bench_upper}, S={s_list}
goal: 1
stage: stability
seed: {seed}
n_epochs: 200

model:
  config: model/configs/{sampler}_default.yaml

benchmark:
  config: benchmark/configs/{bench}.yaml

subset:
  S: {s_list_flow}
  full_target: true

init:
  from_checkpoint: result/checkpoints/goal1_{sampler}_{bench}_{subset_name}_pretrain_seed0/epoch_500.pt
  noise_level: {noise}

logging:
  checkpoint_every: 40
  eval_every: 20
  n_eval_samples: 10000

output:
  checkpoint_dir: result/checkpoints/goal1_{sampler}_{bench}_{subset_name}_noise{noise_tag}_seed{seed}/
  log_file: result/logs/goal1_{sampler}_{bench}_{subset_name}_noise{noise_tag}_seed{seed}.csv
"""

count = 0
for noise, noise_tag in noise_levels:
    for subset_name, flow in subsets:
        for bench in benchmarks:
            for sampler in samplers:
                for seed in seeds:
                    fname = f"goal1_{sampler}_{bench}_{subset_name}_noise{noise_tag}_seed{seed}.yaml"
                    content = TEMPLATE.format(
                        noise=noise,
                        sampler_upper=sampler.upper(),
                        bench_upper=bench.upper(),
                        s_list=str(flow),
                        seed=seed,
                        sampler=sampler,
                        bench=bench,
                        s_list_flow=str(flow),
                        subset_name=subset_name,
                        noise_tag=noise_tag,
                    )
                    with open(os.path.join(OUT_DIR, fname), "w") as f:
                        f.write(content)
                    count += 1

print(f"Generated {count} files.")
