#!/usr/bin/env bash
# Launch cross-domain diffusion training in tmux with TensorBoard.

SESSION_NAME=${1:-pkl_multi_$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${2:-multidomain-ddpm-tmux}
PROJ_DIR="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising"
VENV_BIN="$PROJ_DIR/.venv/bin"
LOG_DIR="$PROJ_DIR/logs/$SESSION_NAME"
mkdir -p "$LOG_DIR"

# Kill existing session with same name
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

# Build commands
TRAIN_CMD="cd $PROJ_DIR && PYTHONUNBUFFERED=1 $VENV_BIN/python -u $PROJ_DIR/scripts/train_multidomain.py \
  --data-root $PROJ_DIR/data \
  --image-size 128 \
  --batch-size 8 \
  --num-workers 64 \
  --max-epochs 100 \
  --lr 1e-4 \
  --timesteps 1000 \
  --use-ema \
  --wandb \
  --project pkl-diffusion-multidomain \
  --name $RUN_NAME \
  --device cuda | tee $LOG_DIR/train.log"

TB_CMD="cd $PROJ_DIR && $VENV_BIN/tensorboard --logdir $PROJ_DIR/logs --host 0.0.0.0 --port 6006 | tee $LOG_DIR/tensorboard.log"

# Create tmux session and run training in pane 0
TMUX_SHELL=${SHELL:-/bin/bash}
tmux new-session -d -s "$SESSION_NAME" -c "$PROJ_DIR" "$TMUX_SHELL -lc '$TRAIN_CMD'"

# Create pane 1 and run tensorboard
sleep 1
tmux split-window -h -t "$SESSION_NAME":0 -c "$PROJ_DIR" "$TMUX_SHELL -lc '$TB_CMD'"

echo "Started tmux session: $SESSION_NAME"
echo "- Training log: $LOG_DIR/train.log"
echo "- TensorBoard: http://<host>:6006"
