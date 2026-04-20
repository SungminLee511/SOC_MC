#!/bin/bash
# Retry: Ablation 1.2 (mode separation) pretrain + stability
cd /home/sky/SML/SOC_MC
CONDA_ENV="adjoint_samplers"
LOG_DIR="result/nohup_logs"

run_batches() {
  local phase_name="$1"
  shift
  local configs=("$@")
  local TOTAL=${#configs[@]}
  local BATCH_SIZE=20
  local NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))

  echo ""
  echo "PHASE: $phase_name — $TOTAL runs, $NUM_BATCHES batches"
  echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"

  for (( b=0; b<NUM_BATCHES; b++ )); do
    START=$((b * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
    COUNT=$((END - START))
    LOCAL_HALF=$(( COUNT / 2 ))

    echo "BATCH $((b+1))/$NUM_BATCHES ($COUNT runs) — $(date -u -d '+9 hours' '+%H:%M:%S KST')"

    PIDS=()
    for (( i=START; i<END; i++ )); do
      cfg="${configs[$i]}"
      name=$(basename "$cfg" .yaml)
      IDX=$((i - START))
      if [ $IDX -lt $LOCAL_HALF ]; then GPU=0; else GPU=1; fi
      CUDA_VISIBLE_DEVICES=$GPU conda run -n $CONDA_ENV python model/train.py --exp_config "$cfg" \
        > "$LOG_DIR/${name}.log" 2>&1 &
      PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do wait $pid; done
    echo "  done $(date -u -d '+9 hours' '+%H:%M:%S KST')"
  done
  echo "PHASE $phase_name COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
}

echo "=== Ablation 1.2 Retry ==="
echo "Start: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"

mapfile -t RHO_PRETRAIN < <(ls result/experiments/goal1_*_w5rho*_pretrain_seed0.yaml | sort)
run_batches "1.2_Pretrain" "${RHO_PRETRAIN[@]}"

mapfile -t RHO_STABILITY < <(ls result/experiments/goal1_*_w5rho*_stability_seed*.yaml | sort)
run_batches "1.2_Stability" "${RHO_STABILITY[@]}"

echo ""
echo "=== ABLATION 1.2 COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST') ==="
