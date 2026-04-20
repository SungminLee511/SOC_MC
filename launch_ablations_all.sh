#!/bin/bash
# Master launcher: Goal 1 Stage 3 Ablations
# Phase 1: Ablation 1.1 (init noise) — 2700 stability runs, 135 batches
# Phase 2: Ablation 1.2 pretrain (mode sep) — 240 pretrain runs, 12 batches
# Phase 3: Ablation 1.2 stability (mode sep) — 1200 stability runs, 60 batches
# Total: 4140 runs, 207 batches
cd /home/sky/SML/SOC_MC
CONDA_ENV="adjoint_samplers"
LOG_DIR="result/nohup_logs"
mkdir -p "$LOG_DIR"

run_batches() {
  local phase_name="$1"
  shift
  local configs=("$@")
  local TOTAL=${#configs[@]}
  local BATCH_SIZE=20
  local HALF=$((BATCH_SIZE / 2))
  local NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))

  echo ""
  echo "############################################"
  echo "PHASE: $phase_name"
  echo "Total: $TOTAL experiments in $NUM_BATCHES batches"
  echo "Started: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
  echo "############################################"
  echo ""

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

    for pid in "${PIDS[@]}"; do
      wait $pid
    done
    echo "  done $(date -u -d '+9 hours' '+%H:%M:%S KST')"
  done

  echo ""
  echo "PHASE $phase_name COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
}

echo "=========================================="
echo "GOAL 1 STAGE 3: ALL ABLATIONS"
echo "Start: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="

# Phase 1: Ablation 1.1 — init noise (2700 runs)
mapfile -t NOISE_CONFIGS < <(ls result/experiments/goal1_*_noise*_seed*.yaml | sort)
run_batches "Ablation_1.1_InitNoise" "${NOISE_CONFIGS[@]}"

# Phase 2: Ablation 1.2 — mode separation pretrain (240 runs)
mapfile -t RHO_PRETRAIN < <(ls result/experiments/goal1_*_w5rho*_pretrain_seed0.yaml | sort)
run_batches "Ablation_1.2_ModeSep_Pretrain" "${RHO_PRETRAIN[@]}"

# Phase 3: Ablation 1.2 — mode separation stability (1200 runs)
mapfile -t RHO_STABILITY < <(ls result/experiments/goal1_*_w5rho*_stability_seed*.yaml | sort)
run_batches "Ablation_1.2_ModeSep_Stability" "${RHO_STABILITY[@]}"

echo ""
echo "=========================================="
echo "ALL ABLATIONS COMPLETE: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
echo "=========================================="
