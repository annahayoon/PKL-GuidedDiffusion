#!/usr/bin/env bash
set -euo pipefail

# Orchestrate baseline comparisons using tmux.
# This script launches evaluation runs comparing PKL, L2, and Anscombe guidance
# across DDIM step counts, using a single trained diffusion checkpoint.
# It also supports RCAN if configured via Hydra (baselines.rcan_checkpoint).

# Usage:
#   bash scripts/run_all_experiments_tmux.sh
# Environment overrides (optional):
#   PROJECT_ROOT, CHECKPOINT, INPUT_DIR, GT_DIR, OUTPUT_ROOT, SESSION, DDIM_STEPS

PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$(dirname "$0")"/.. && pwd)"}
cd "$PROJECT_ROOT"

# Paths
if [ -z "${CHECKPOINT:-}" ]; then
  CANDIDATE1="$PROJECT_ROOT/checkpoints/real_run1/best_trainer.pt"
  CANDIDATE2="$PROJECT_ROOT/checkpoints/best_model.pt"
  if [ -f "$CANDIDATE1" ]; then
    CHECKPOINT="$CANDIDATE1"
  else
    CHECKPOINT="$CANDIDATE2"
  fi
fi
INPUT_DIR=${INPUT_DIR:-"$PROJECT_ROOT/data/real_microscopy/real_pairs/val/wf"}
GT_DIR=${GT_DIR:-"$PROJECT_ROOT/data/real_microscopy/real_pairs/val/2p"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PROJECT_ROOT/outputs/baseline_comparison_pkl"}

# DDIM step sweep (space-separated)
DDIM_STEPS=${DDIM_STEPS:-"25 50 100"}

# Tmux session name
SESSION=${SESSION:-"pkl_baselines"}

mkdir -p "$OUTPUT_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux not found. Please install tmux or run commands manually." >&2
  exit 1
fi

# Helper to start a tmux window for a job
# $1: window name
# $2: command
start_job() {
  local name="$1"
  local cmd="$2"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$name" "$cmd" >/dev/null
  else
    tmux new-window -t "$SESSION" -n "$name" "$cmd" >/dev/null
  fi
  echo "Launched [$SESSION:$name]"
}

# Build commands
# We run evaluate.py which computes metrics and comparisons for l2/anscombe/pkl
# in a single pass, controlled by inference.ddim_steps.

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$OUTPUT_ROOT/logs_$RUN_TS"
mkdir -p "$LOG_DIR"

# Optional: set Hydra runtime outputs to the OUTPUT_ROOT to keep things tidy
HYDRA_BASE="hydra.run.dir=$OUTPUT_ROOT hydra.output_subdir=null"

for steps in $DDIM_STEPS; do
  RUN_NAME="eval_steps_${steps}"
  OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
  mkdir -p "$OUT_DIR"
  CMD="python -u scripts/evaluate.py \
    $HYDRA_BASE \
    experiment.name=$RUN_NAME \
    inference.checkpoint_path=$CHECKPOINT \
    inference.input_dir=$INPUT_DIR \
    +inference.gt_dir=$GT_DIR \
    inference.output_dir=$OUT_DIR \
    inference.ddim_steps=$steps \
    inference.eta=0.0 \
    experiment.device=cuda \
    training.use_conditioning=true \
    guidance.type=pkl"
  # Note: evaluate.py internally evaluates l2, anscombe, and pkl using the same checkpoint/config.
  # If you have an RCAN checkpoint, add: baselines.rcan_checkpoint=/abs/path/to/rcan.ckpt
  LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
  start_job "$RUN_NAME" "$CMD | tee $LOG_FILE"
done

# Optionally, add per-method inference runs to save full-resolution reconstructions (optional).
# Uncomment the block below if you want separate saved reconstructions by method and steps.
: <<'OPTIONAL_INFERENCE'
METHODS=(pkl l2 anscombe)
for steps in $DDIM_STEPS; do
  for method in "${METHODS[@]}"; do
    RUN_NAME="infer_${method}_steps_${steps}"
    OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
    mkdir -p "$OUT_DIR"
    CMD="python -u scripts/inference.py \
      $HYDRA_BASE \
      experiment.name=$RUN_NAME \
      inference.checkpoint_path=$CHECKPOINT \
      inference.input_dir=$INPUT_DIR \
      inference.output_dir=$OUT_DIR \
      inference.ddim_steps=$steps \
      inference.eta=0.0 \
      experiment.device=cuda \
      training.use_conditioning=true \
      guidance.type=$method"
    LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
    start_job "$RUN_NAME" "$CMD | tee $LOG_FILE"
  done
done
OPTIONAL_INFERENCE

# Final note
echo "All jobs launched in tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
