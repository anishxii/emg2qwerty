#!/bin/bash
# Sequential job queue for ablation runs on a single GPU.
# Usage: bash scripts/run_queue.sh
# Each run logs to logs/ablations/<name>.log

set -euo pipefail

CD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CD_DIR"

LOG_DIR="$CD_DIR/logs/ablations"
mkdir -p "$LOG_DIR"

BASE="user=single_user trainer.max_epochs=40 trainer.accelerator=gpu trainer.devices=1"

# Format: "run_name:::hydra_overrides"
RUNS=(
  # --- Baseline (shared reference for all axes) ---
  "baseline:::model=cnn_bilstm_ctc"

  # --- Axis 1: Channel count ---
  "ch8:::model=cnn_bilstm_ctc_8ch transforms=log_spectrogram_8ch"
  "ch4:::model=cnn_bilstm_ctc_4ch transforms=log_spectrogram_4ch"

  # --- Axis 2: Training data size ---
  "data50pct:::model=cnn_bilstm_ctc user=single_user_50pct"
  "data25pct:::model=cnn_bilstm_ctc user=single_user_25pct"

  # --- Axis 3: Sampling rate ---
  "sr1khz:::model=cnn_bilstm_ctc transforms=log_spectrogram_1khz"
  "sr500hz:::model=cnn_bilstm_ctc transforms=log_spectrogram_500hz"
)

TOTAL=${#RUNS[@]}
echo "Starting queue: $TOTAL runs"
echo "Logs: $LOG_DIR"
echo "========================================"

for i in "${!RUNS[@]}"; do
  entry="${RUNS[$i]}"
  name="${entry%%:::*}"
  args="${entry##*:::}"

  echo ""
  echo "[$(date '+%H:%M:%S')] Run $((i+1))/$TOTAL: $name"
  echo "  Args: $BASE $args"

  WANDB_RUN_NAME="ablation-$name" python3.10 -m emg2qwerty.train \
    $BASE $args \
    > "$LOG_DIR/${name}.log" 2>&1 \
    && echo "[$(date '+%H:%M:%S')] Done: $name" \
    || echo "[$(date '+%H:%M:%S')] FAILED: $name (see $LOG_DIR/${name}.log)"
done

echo ""
echo "========================================"
echo "Queue complete at $(date '+%H:%M:%S')"
