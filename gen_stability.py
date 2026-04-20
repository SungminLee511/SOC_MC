#!/usr/bin/env python3
import os, itertools

OUT = "/home/sky/SML/SOC_MC/result/experiments"
os.makedirs(OUT, exist_ok=True)

benchmarks = ["w5", "c5", "b5"]
samplers = ["as", "asbs"]
seeds = [0, 1, 2, 3, 4]

subsets = {}
modes = list(range(5))
from itertools import combinations
for k in range(1, 5):
    for combo in combinations(modes, k):
        name = "S" + "".join(str(c) for c in combo)
        subsets[name] = list(combo)

created = 0
skipped = 0

for sampler in samplers:
    for bench in benchmarks:
        for subset_name, s_list in subsets.items():
            for seed in seeds:
                fname = f"goal1_{sampler}_{bench}_{subset_name}_stability_seed{seed}.yaml"
                fpath = os.path.join(OUT, fname)
                if os.path.exists(fpath):
                    skipped += 1
                    continue

                sampler_upper = sampler.upper()
                bench_upper = bench.upper()
                s_flow = "[" + ", ".join(str(x) for x in s_list) + "]"

                content = f"""# Goal 1 Stage 2: Stability test — {sampler_upper} on {bench_upper}, init from S={s_flow} pretrain
goal: 1
stage: stability
seed: {seed}
n_epochs: 200

model:
  config: model/configs/{sampler}_default.yaml

benchmark:
  config: benchmark/configs/{bench}.yaml

subset:
  S: {s_flow}
  full_target: true

init:
  from_checkpoint: result/checkpoints/goal1_{sampler}_{bench}_{subset_name}_pretrain_seed0/epoch_500.pt
  noise_level: 0.0

logging:
  checkpoint_every: 40
  eval_every: 20
  n_eval_samples: 10000

output:
  checkpoint_dir: result/checkpoints/goal1_{sampler}_{bench}_{subset_name}_stability_seed{seed}/
  log_file: result/logs/goal1_{sampler}_{bench}_{subset_name}_stability_seed{seed}.csv
"""
                with open(fpath, "w") as f:
                    f.write(content)
                created += 1

print(f"Created: {created}, Skipped: {skipped}, Total expected: 900")
