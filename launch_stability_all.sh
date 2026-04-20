#!/bin/bash
# Sequential launcher: Goal 1 Stage 2 stability runs
# 900 experiments in 45 batches of 20, 10 per GPU
cd /home/sky/SML/SOC_MC
CONDA_ENV="adjoint_samplers"
LOG_DIR="result/nohup_logs"
mkdir -p "$LOG_DIR"

# Collect all stability configs sorted
mapfile -t ALL_CONFIGS < <(ls result/experiments/goal1_*_stability_seed*.yaml | sort)

TOTAL=${#ALL_CONFIGS[@]}
BATCH_SIZE=20
HALF=$((BATCH_SIZE / 2))
NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Starting Goal 1 Stage 2 stability runs: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "Total: $TOTAL experiments in $NUM_BATCHES batches of $BATCH_SIZE"
echo ""

for (( b=0; b<NUM_BATCHES; b++ )); do
  START=$((b * BATCH_SIZE))
  END=$((START + BATCH_SIZE))
  if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
  COUNT=$((END - START))

  echo "=========================================="
  echo "BATCH $((b+1))/$NUM_BATCHES ($COUNT experiments, indices $START-$((END-1)))"
  echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
  echo "=========================================="

  PIDS=()
  for (( i=START; i<END; i++ )); do
    cfg="${ALL_CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    IDX=$((i - START))
    if [ $IDX -lt $HALF ]; then GPU=0; else GPU=1; fi
    echo "  [$IDX] GPU=$GPU $name"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV python model/train.py --exp_config "$cfg" \
      > "$LOG_DIR/${name}.log" 2>&1 &
    PIDS+=($!)
  done

  echo "  Waiting for $COUNT processes..."
  for pid in "${PIDS[@]}"; do
    wait $pid
  done
  echo "  Batch $((b+1)) DONE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
  echo ""
done

echo "=========================================="
echo "ALL STABILITY RUNS COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="
