#!/bin/bash
# Launch batch of experiments: 10 per GPU, 20 total
# Usage: bash launch_batch.sh

cd /home/sky/SML/SOC_MC
CONDA_ENV="adjoint_samplers"
LOG_DIR="result/nohup_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  # GPU 0: experiments 0-9
  "result/experiments/goal1_as_w5_S0_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S1_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S2_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S3_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S4_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S01_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S02_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S03_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S04_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S12_pretrain_seed0.yaml"
  # GPU 1: experiments 10-19
  "result/experiments/goal1_as_w5_S13_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S14_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S23_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S24_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S34_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S012_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S013_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S014_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S023_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S024_pretrain_seed0.yaml"
)

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name=$(basename "$cfg" .yaml)

  if [ $i -lt 10 ]; then
    GPU=0
  else
    GPU=1
  fi

  echo "Launching [$i] GPU=$GPU $name"
  CUDA_VISIBLE_DEVICES=$GPU nohup conda run -n $CONDA_ENV python model/train.py --exp_config "$cfg" \
    > "$LOG_DIR/${name}.log" 2>&1 &
done

echo ""
echo "Launched ${#CONFIGS[@]} experiments. Check: ps aux | grep train.py"
echo "Logs: $LOG_DIR/"
