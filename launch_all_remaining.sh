#!/bin/bash
# Sequential batch launcher: 7 batches of 20, waits for each batch to finish
cd /home/sky/SML/SOC_MC
CONDA_ENV="adjoint_samplers"
LOG_DIR="result/nohup_logs"
mkdir -p "$LOG_DIR"

launch_batch() {
  local batch_name="$1"
  shift
  local configs=("$@")
  local half=$(( ${#configs[@]} / 2 ))

  echo ""
  echo "=========================================="
  echo "BATCH: $batch_name (${#configs[@]} experiments)"
  echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
  echo "=========================================="

  local pids=()
  for i in "${!configs[@]}"; do
    cfg="${configs[$i]}"
    name=$(basename "$cfg" .yaml)
    if [ $i -lt $half ]; then GPU=0; else GPU=1; fi
    echo "  [$i] GPU=$GPU $name"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV python model/train.py --exp_config "$cfg" \
      > "$LOG_DIR/${name}.log" 2>&1 &
    pids+=($!)
  done

  echo "  Waiting for ${#pids[@]} processes..."
  for pid in "${pids[@]}"; do
    wait $pid
  done

  echo "  Batch $batch_name DONE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
}

# ============================================================
# Batch 3: W5/ASBS remaining 20 (|S|=2 rest + |S|=3 + |S|=4)
# ============================================================
BATCH3=(
  # GPU 0
  "result/experiments/goal1_asbs_w5_S13_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S14_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S23_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S24_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S34_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S012_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S013_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S014_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S023_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S024_pretrain_seed0.yaml"
  # GPU 1
  "result/experiments/goal1_asbs_w5_S034_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S123_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S124_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S134_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S234_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S0123_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S0124_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S0134_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S0234_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_w5_S1234_pretrain_seed0.yaml"
)

# ============================================================
# Batch 4: C5/AS first 20 (|S|=1 + |S|=2 + |S|=3 first 5)
# ============================================================
BATCH4=(
  # GPU 0
  "result/experiments/goal1_as_c5_S0_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S1_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S2_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S3_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S4_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S01_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S02_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S03_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S04_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S12_pretrain_seed0.yaml"
  # GPU 1
  "result/experiments/goal1_as_c5_S13_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S14_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S23_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S24_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S34_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S012_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S013_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S014_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S023_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S024_pretrain_seed0.yaml"
)

# ============================================================
# Batch 5: C5/AS remaining 10 + C5/ASBS first 10
# ============================================================
BATCH5=(
  # GPU 0: C5/AS remaining
  "result/experiments/goal1_as_c5_S034_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S123_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S124_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S134_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S234_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S0123_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S0124_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S0134_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S0234_pretrain_seed0.yaml"
  "result/experiments/goal1_as_c5_S1234_pretrain_seed0.yaml"
  # GPU 1: C5/ASBS first 10
  "result/experiments/goal1_asbs_c5_S0_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S1_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S2_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S3_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S4_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S01_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S02_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S03_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S04_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S12_pretrain_seed0.yaml"
)

# ============================================================
# Batch 6: C5/ASBS remaining 20
# ============================================================
BATCH6=(
  # GPU 0
  "result/experiments/goal1_asbs_c5_S13_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S14_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S23_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S24_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S34_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S012_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S013_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S014_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S023_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S024_pretrain_seed0.yaml"
  # GPU 1
  "result/experiments/goal1_asbs_c5_S034_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S123_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S124_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S134_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S234_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S0123_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S0124_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S0134_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S0234_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_c5_S1234_pretrain_seed0.yaml"
)

# ============================================================
# Batch 7: B5/AS first 20
# ============================================================
BATCH7=(
  # GPU 0
  "result/experiments/goal1_as_b5_S0_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S1_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S2_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S3_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S4_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S01_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S02_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S03_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S04_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S12_pretrain_seed0.yaml"
  # GPU 1
  "result/experiments/goal1_as_b5_S13_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S14_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S23_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S24_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S34_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S012_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S013_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S014_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S023_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S024_pretrain_seed0.yaml"
)

# ============================================================
# Batch 8: B5/AS remaining 10 + B5/ASBS first 10
# ============================================================
BATCH8=(
  # GPU 0: B5/AS remaining
  "result/experiments/goal1_as_b5_S034_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S123_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S124_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S134_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S234_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S0123_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S0124_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S0134_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S0234_pretrain_seed0.yaml"
  "result/experiments/goal1_as_b5_S1234_pretrain_seed0.yaml"
  # GPU 1: B5/ASBS first 10
  "result/experiments/goal1_asbs_b5_S0_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S1_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S2_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S3_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S4_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S01_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S02_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S03_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S04_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S12_pretrain_seed0.yaml"
)

# ============================================================
# Batch 9: B5/ASBS remaining 20
# ============================================================
BATCH9=(
  # GPU 0
  "result/experiments/goal1_asbs_b5_S13_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S14_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S23_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S24_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S34_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S012_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S013_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S014_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S023_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S024_pretrain_seed0.yaml"
  # GPU 1
  "result/experiments/goal1_asbs_b5_S034_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S123_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S124_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S134_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S234_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S0123_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S0124_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S0134_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S0234_pretrain_seed0.yaml"
  "result/experiments/goal1_asbs_b5_S1234_pretrain_seed0.yaml"
)

# ============================================================
# Run all batches sequentially
# ============================================================
echo "Starting all remaining batches: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Total: 7 batches x 20 = 140 experiments"
echo ""

launch_batch "3_W5_ASBS_remaining" "${BATCH3[@]}"
launch_batch "4_C5_AS_first20" "${BATCH4[@]}"
launch_batch "5_C5_AS_rest_ASBS_first" "${BATCH5[@]}"
launch_batch "6_C5_ASBS_remaining" "${BATCH6[@]}"
launch_batch "7_B5_AS_first20" "${BATCH7[@]}"
launch_batch "8_B5_AS_rest_ASBS_first" "${BATCH8[@]}"
launch_batch "9_B5_ASBS_remaining" "${BATCH9[@]}"

echo ""
echo "=========================================="
echo "ALL BATCHES COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="
