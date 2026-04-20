#!/bin/bash
# Batch 2: 10 remaining W5/AS + 10 first W5/ASBS, 10 per GPU
cd /home/sky/SML/SOC_MC
CONDA_ENV="adjoint_samplers"
LOG_DIR="result/nohup_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  # GPU 0: W5/AS remaining (5x |S|=3 + 5x |S|=4)
  "result/experiments/goal1_as_w5_S034_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S123_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S124_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S134_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S234_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S0123_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S0124_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S0134_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S0234_pretrain_seed0.yaml"
  "result/experiments/goal1_as_w5_S1234_pretrain_seed0.yaml"
  # GPU 1: W5/ASBS first 10 (5x |S|=1 + 5x |S|=2)
  "result/experiments/goal1_asbs_w5_S0_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S1_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S2_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S3_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S4_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S01_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S02_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S03_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S04_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S12_pretrain_seed0.yaml"
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
