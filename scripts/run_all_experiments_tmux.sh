#!/usr/bin/env bash

set -euo pipefail

# Orchestrate preprocessing → splits → training → inference → evaluation in tmux
# Logs are saved under logs/run_{timestamp}/*.log

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="real_run_${TS}"
LOG_DIR="$PROJECT_ROOT/logs/${RUN_NAME}"
OUT_DIR="$PROJECT_ROOT/outputs/${RUN_NAME}"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints/${RUN_NAME}"
DATA_DIR_REAL="$PROJECT_ROOT/data/real_microscopy"
INFER_OUT_DIR="$OUT_DIR/inference"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$CHECKPOINTS_DIR" "$DATA_DIR_REAL" "$INFER_OUT_DIR"

# Default paths for real data (override via env vars or CLI)
WF_PATH_DEFAULT="$PROJECT_ROOT/data_wf_tp/wf.tif"
TP_PATH_DEFAULT="$PROJECT_ROOT/data_wf_tp/tp_reg.tif"
BEADS_DIR_DEFAULT="$PROJECT_ROOT/data_wf_tp/beads"

WF_PATH="${WF_PATH:-$WF_PATH_DEFAULT}"
TP_PATH="${TP_PATH:-$TP_PATH_DEFAULT}"
BEADS_DIR="${BEADS_DIR:-$BEADS_DIR_DEFAULT}"

# Metric thresholds (override via env vars)
THRESH_PSNR_MIN="${THRESH_PSNR_MIN:-24.0}"
THRESH_SSIM_MIN="${THRESH_SSIM_MIN:-0.65}"

# Training overrides
MAX_EPOCHS="${MAX_EPOCHS:-50}"
# Leave these unset for autotune unless user provides them
: "${DEVICE:=auto}"
: "${BATCH_SIZE:=}"
: "${NUM_WORKERS:=}"

SESSION="pkl_dg_${TS}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/orchestrator.log"; }

ensure_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux not found. Please install tmux." >&2
    exit 1
  fi
}

ensure_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found." >&2
    exit 1
  fi
}

ensure_tmux
ensure_python

export PROJECT_ROOT
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.8"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Detect hardware to tune defaults
CORES=$(nproc --all 2>/dev/null || echo 8)
MEM_AVAIL_GB=$(free -g | awk '/Mem:/ {print $7}' 2>/dev/null || echo 32)
GPU_INFO=$(bash -lc "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits" 2>/dev/null || true)
GPU_COUNT=$(echo "$GPU_INFO" | grep -c . || echo 0)
GPU_NAME=$(echo "$GPU_INFO" | head -n1 | awk -F, '{gsub(/^ +| +$/,"",$1); print $1}')
GPU_MEM_TOTAL=$(echo "$GPU_INFO" | head -n1 | awk -F, '{gsub(/^ +| +$/,"",$2); print $2}')
GPU_MEM_FREE=$(echo "$GPU_INFO" | head -n1 | awk -F, '{gsub(/^ +| +$/,"",$3); print $3}')

# Decide device
if [ "${DEVICE:-auto}" = "auto" ]; then
  if [ "$GPU_COUNT" -gt 0 ] && [ "${GPU_MEM_FREE:-0}" -gt 2000 ]; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

# Optional knobs to avoid OOM and shorten eval (autotuned if not provided)
AUTO_SCALE_BATCH="${AUTO_SCALE_BATCH:-1}"
EVAL_SUBSET="${EVAL_SUBSET:-auto}"
EVAL_TIF_DIR="$OUT_DIR/eval_tif"

# Autotune NUM_WORKERS if not set explicitly
if [ -z "${NUM_WORKERS_SET:-}" ] && [ -z "${NUM_WORKERS+x}" ]; then
  # Favor 8 by default; scale up to 16 on large-CPU hosts
  if [ "$CORES" -ge 48 ]; then
    NUM_WORKERS=16
  else
    NUM_WORKERS=8
  fi
fi

# Autotune initial BATCH_SIZE if not set explicitly
if [ -z "${BATCH_SIZE_SET:-}" ] && [ -z "${BATCH_SIZE+x}" ]; then
  if [ "$DEVICE" = "cuda" ]; then
    # Conservative mapping for 256x256 UNet-small
    if [ "${GPU_MEM_FREE:-0}" -ge 36000 ]; then
      BATCH_SIZE=16
    elif [ "${GPU_MEM_FREE:-0}" -ge 24000 ]; then
      BATCH_SIZE=8
    elif [ "${GPU_MEM_FREE:-0}" -ge 12000 ]; then
      BATCH_SIZE=4
    else
      BATCH_SIZE=2
    fi
  else
    BATCH_SIZE=2
  fi
fi

# Determine test set size to possibly cap evaluation subset
TEST_WF_DIR="$DATA_DIR_REAL/splits/test/wf"
if [ "$EVAL_SUBSET" = "auto" ]; then
  if [ -d "$TEST_WF_DIR" ]; then
    TEST_COUNT=$(find "$TEST_WF_DIR" -type f -name "*.png" | wc -l | awk '{print $1}')
    if [ "$TEST_COUNT" -gt 0 ] && [ "$TEST_COUNT" -gt 1500 ]; then
      EVAL_SUBSET=1000
    else
      EVAL_SUBSET=0
    fi
  else
    EVAL_SUBSET=0
  fi
fi

log "Auto-config: CORES=$CORES, MEM_AVAILABLE_GB=$MEM_AVAIL_GB, GPU=$GPU_NAME total=${GPU_MEM_TOTAL:-0}MiB free=${GPU_MEM_FREE:-0}MiB"
log "Auto-config: DEVICE=$DEVICE, BATCH_SIZE=$BATCH_SIZE, NUM_WORKERS=$NUM_WORKERS, EVAL_SUBSET=$EVAL_SUBSET"

# Build commands
CMD_PREPARE="python3 scripts/process_real_data.py \
  --wf-path \"$WF_PATH\" \
  --tp-path \"$TP_PATH\" \
  --beads-dir \"$BEADS_DIR\" \
  --output-dir \"$DATA_DIR_REAL\" \
  --create-splits"

CMD_SPLITS="python3 scripts/create_frame_based_splits.py \
  --data-dir \"$DATA_DIR_REAL\""

CMD_TRAIN="python3 scripts/train_real_data.py \
  wandb.mode=disabled \
  experiment.device=$DEVICE \
  training.max_epochs=$MAX_EPOCHS \
  training.batch_size=$BATCH_SIZE \
  training.num_workers=$NUM_WORKERS \
  paths.checkpoints=\"$CHECKPOINTS_DIR\" \
  paths.data=\"$PROJECT_ROOT/data\" \
  paths.outputs=\"$OUT_DIR\" \
  paths.logs=\"$PROJECT_ROOT/logs\""

# Use latest checkpoint produced by training for inference/eval
CHECKPOINT_PATH="$CHECKPOINTS_DIR/final_model.pt"

CMD_INFER="python3 scripts/inference.py \
  experiment.device=$DEVICE \
  inference.checkpoint_path=\"$CHECKPOINT_PATH\" \
  inference.input_dir=\"$EVAL_TIF_DIR/wf\" \
  inference.output_dir=\"$INFER_OUT_DIR\""

# Evaluation will operate on TIF-converted copies of test WF/2P (to match evaluator expectations)
CMD_EVAL="python3 scripts/evaluate.py \
  experiment.device=$DEVICE \
  inference.checkpoint_path=\"$CHECKPOINT_PATH\" \
  inference.input_dir=\"$EVAL_TIF_DIR/wf\" \
  inference.gt_dir=\"$EVAL_TIF_DIR/2p\""

# Helper: common TIF conversion for test WF/2P (subset-aware)
cat >"$LOG_DIR/prepare_tif.py" <<'PY'
import sys, os, glob
from PIL import Image
import numpy as np
import tifffile

if len(sys.argv) < 6:
    print("usage: prepare_tif.py <src_wf_png> <src_2p_png> <out_wf_tif> <out_2p_tif> <max_n>")
    sys.exit(1)

src_wf, src_2p, out_wf, out_2p, max_n = sys.argv[1:6]
max_n = int(max_n)
os.makedirs(out_wf, exist_ok=True)
os.makedirs(out_2p, exist_ok=True)

wf_files = sorted(glob.glob(os.path.join(src_wf, '*.png')))
if max_n > 0:
    wf_files = wf_files[:max_n]

for wf_path in wf_files:
    bn = os.path.basename(wf_path)
    stem, _ = os.path.splitext(bn)
    gt_path = os.path.join(src_2p, bn)
    if not os.path.exists(gt_path):
        continue
    arr_wf = np.array(Image.open(wf_path)).astype(np.float32)
    arr_2p = np.array(Image.open(gt_path)).astype(np.float32)
    tifffile.imwrite(os.path.join(out_wf, f'{stem}.tif'), arr_wf)
    tifffile.imwrite(os.path.join(out_2p, f'{stem}.tif'), arr_2p)
PY

# Helper: training with automatic batch-size downscaling on OOM
cat >"$LOG_DIR/03_train_autoscale.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="$1"; DEVICE="$2"; MAX_EPOCHS="$3"; NUM_WORKERS="$4"; CHECKPOINTS_DIR="$5"; OUT_DIR="$6"; PROJECT_ROOT="$7"; INITIAL_BS="$8"; shift 8
TRY_LIST=("$INITIAL_BS")
for v in "$@"; do TRY_LIST+=("$v"); done

success=0
for bs in "${TRY_LIST[@]}"; do
  echo "== Training with batch_size=${bs} ==" | tee -a "$LOG_DIR/03_train.log"
  set +e
  python3 scripts/train_real_data.py \
    wandb.mode=disabled \
    experiment.device="$DEVICE" \
    training.max_epochs="$MAX_EPOCHS" \
    training.batch_size="$bs" \
    training.num_workers="$NUM_WORKERS" \
    paths.checkpoints="$CHECKPOINTS_DIR" \
    paths.data="$PROJECT_ROOT/data" \
    paths.outputs="$OUT_DIR" \
    paths.logs="$PROJECT_ROOT/logs" \
    2>&1 | tee -a "$LOG_DIR/03_train.log"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then success=1; break; fi
  if grep -qi "out of memory" "$LOG_DIR/03_train.log"; then
    echo "OOM encountered at batch_size=${bs}, trying smaller..." | tee -a "$LOG_DIR/03_train.log"
    continue
  else
    echo "Training failed (non-OOM). See log." | tee -a "$LOG_DIR/03_train.log"
    exit $rc
  fi
done

if [ "$success" -ne 1 ]; then
  echo "All batch sizes failed. Aborting." | tee -a "$LOG_DIR/03_train.log"
  exit 2
fi
EOF
chmod +x "$LOG_DIR/03_train_autoscale.sh"

# Create helper for preprocess + splits with skip logic
cat >"$LOG_DIR/01_preprocess_or_skip.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR_REAL="$1"
LOG_DIR="$2"
CMD_PREPARE="$3"
CMD_SPLITS="$4"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

pairs_exist=0
splits_exist=0
if compgen -G "$DATA_DIR_REAL/real_pairs/wf/*.png" > /dev/null && compgen -G "$DATA_DIR_REAL/real_pairs/2p/*.png" > /dev/null; then
  pairs_exist=1
fi
if compgen -G "$DATA_DIR_REAL/splits/train/wf/*.png" > /dev/null; then
  splits_exist=1
fi

if [ "$FORCE_PREPROCESS" = "1" ]; then
  echo "[preprocess] FORCE_PREPROCESS=1 → running full preprocessing and splits" | tee "$LOG_DIR/01_preprocess.log"
  eval "$CMD_PREPARE" 2>&1 | tee -a "$LOG_DIR/01_preprocess.log"
  echo "[splits] creating frame-based splits" | tee "$LOG_DIR/02_splits.log"
  eval "$CMD_SPLITS" 2>&1 | tee -a "$LOG_DIR/02_splits.log"
  exit 0
fi

if [ "$splits_exist" = "1" ]; then
  echo "[skip] Detected existing splits under $DATA_DIR_REAL/splits → skipping preprocessing and splitting" | tee "$LOG_DIR/01_preprocess.log"
  exit 0
fi

if [ "$pairs_exist" = "1" ]; then
  echo "[preprocess] Pairs already exist → skipping preprocessing" | tee "$LOG_DIR/01_preprocess.log"
  echo "[splits] creating frame-based splits" | tee "$LOG_DIR/02_splits.log"
  eval "$CMD_SPLITS" 2>&1 | tee -a "$LOG_DIR/02_splits.log"
  exit 0
fi

echo "[preprocess] No pairs/splits found → running full preprocessing and splits" | tee "$LOG_DIR/01_preprocess.log"
eval "$CMD_PREPARE" 2>&1 | tee -a "$LOG_DIR/01_preprocess.log"
echo "[splits] creating frame-based splits" | tee "$LOG_DIR/02_splits.log"
eval "$CMD_SPLITS" 2>&1 | tee -a "$LOG_DIR/02_splits.log"
EOS
chmod +x "$LOG_DIR/01_preprocess_or_skip.sh"

# Create tmux session and panes
tmux new-session -d -s "$SESSION" -n run

# Pane 1: Preprocess + splits
tmux send-keys -t "$SESSION":run.0 "set -euo pipefail" C-m
tmux send-keys -t "$SESSION":run.0 "cd \"$PROJECT_ROOT\"" C-m
tmux send-keys -t "$SESSION":run.0 "bash \"$LOG_DIR/01_preprocess_or_skip.sh\" \"$DATA_DIR_REAL\" \"$LOG_DIR\" \"$CMD_PREPARE\" \"$CMD_SPLITS\"" C-m
tmux send-keys -t "$SESSION":run.0 "echo '== Preprocessing/splits stage finished (skip logic applied) =='" C-m

# Pane 2: Training (waits until Pane 1 completes by checking splits exist)
tmux split-window -v -t "$SESSION":run
tmux send-keys -t "$SESSION":run.1 "set -euo pipefail" C-m
tmux send-keys -t "$SESSION":run.1 "cd \"$PROJECT_ROOT\"" C-m
tmux send-keys -t "$SESSION":run.1 "echo '== Waiting for splits to exist =='" C-m
tmux send-keys -t "$SESSION":run.1 "while [ ! -d \"$DATA_DIR_REAL/splits/train/wf\" ]; do sleep 5; done" C-m
tmux send-keys -t "$SESSION":run.1 "echo '== Training real-data model ==' | tee \"$LOG_DIR/03_train.log\"" C-m
if [ "$AUTO_SCALE_BATCH" -eq 1 ]; then
  tmux send-keys -t "$SESSION":run.1 "bash \"$LOG_DIR/03_train_autoscale.sh\" \"$LOG_DIR\" \"$DEVICE\" \"$MAX_EPOCHS\" \"$NUM_WORKERS\" \"$CHECKPOINTS_DIR\" \"$OUT_DIR\" \"$PROJECT_ROOT\" \"$BATCH_SIZE\" 8 4 2 1" C-m
else
  tmux send-keys -t "$SESSION":run.1 "$CMD_TRAIN 2>&1 | tee -a \"$LOG_DIR/03_train.log\"" C-m
fi
tmux send-keys -t "$SESSION":run.1 "echo '== Training complete =='" C-m

# Pane 3: Inference (waits for checkpoint)
tmux split-window -h -t "$SESSION":run
tmux send-keys -t "$SESSION":run.2 "set -euo pipefail" C-m
tmux send-keys -t "$SESSION":run.2 "cd \"$PROJECT_ROOT\"" C-m
tmux send-keys -t "$SESSION":run.2 "echo '== Waiting for checkpoint =='" C-m
tmux send-keys -t "$SESSION":run.2 "while [ ! -f \"$CHECKPOINT_PATH\" ]; do sleep 10; done" C-m
tmux send-keys -t "$SESSION":run.2 "echo '== Preparing TIF inputs for inference/evaluation ==' | tee \"$LOG_DIR/04_infer.log\"" C-m
tmux send-keys -t "$SESSION":run.2 "python3 \"$LOG_DIR/prepare_tif.py\" \"$DATA_DIR_REAL/splits/test/wf\" \"$DATA_DIR_REAL/splits/test/2p\" \"$EVAL_TIF_DIR/wf\" \"$EVAL_TIF_DIR/2p\" \"$EVAL_SUBSET\" 2>&1 | tee -a \"$LOG_DIR/04_infer.log\"" C-m
tmux send-keys -t "$SESSION":run.2 "echo '== Inference on test WF (TIF) ==' | tee -a \"$LOG_DIR/04_infer.log\"" C-m
tmux send-keys -t "$SESSION":run.2 "$CMD_INFER 2>&1 | tee -a \"$LOG_DIR/04_infer.log\"" C-m
tmux send-keys -t "$SESSION":run.2 "echo '== Inference complete =='" C-m
tmux send-keys -t "$SESSION":run.2 "touch \"$LOG_DIR/.inference_done\"" C-m

# Pane 4: Evaluation + thresholds (waits for checkpoint)
tmux split-window -v -t "$SESSION":run.2
tmux send-keys -t "$SESSION":run.3 "set -euo pipefail" C-m
tmux send-keys -t "$SESSION":run.3 "cd \"$PROJECT_ROOT\"" C-m
tmux send-keys -t "$SESSION":run.3 "echo '== Waiting for checkpoint and inference to finish =='" C-m
tmux send-keys -t "$SESSION":run.3 "while [ ! -f \"$CHECKPOINT_PATH\" ]; do sleep 10; done" C-m
tmux send-keys -t "$SESSION":run.3 "while [ ! -f \"$LOG_DIR/.inference_done\" ]; do sleep 5; done" C-m
tmux send-keys -t "$SESSION":run.3 "echo '== Evaluating model ==' | tee \"$LOG_DIR/05_eval.log\"" C-m
# No-op if TIF already prepared in Pane 3
tmux send-keys -t "$SESSION":run.3 "python3 \"$LOG_DIR/prepare_tif.py\" \"$DATA_DIR_REAL/splits/test/wf\" \"$DATA_DIR_REAL/splits/test/2p\" \"$EVAL_TIF_DIR/wf\" \"$EVAL_TIF_DIR/2p\" \"$EVAL_SUBSET\" 2>&1 | tee -a \"$LOG_DIR/05_eval.log\"" C-m
tmux send-keys -t "$SESSION":run.3 "$CMD_EVAL 2>&1 | tee -a \"$LOG_DIR/05_eval.log\"" C-m

# Append accuracy gate
cat >"$LOG_DIR/06_threshold_check.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
LOG_FILE="$1"
PSNR_MIN="$2"
SSIM_MIN="$3"

# Parse the summary lines printed by scripts/evaluate.py, e.g.,
# pkl {'psnr': 28.1234, 'ssim': 0.8123, ...}

best_psnr=$(grep -E "^pkl\s+\{" "$LOG_FILE" | tail -n1 | sed -E "s/.*'psnr': ([0-9.]+).*/\1/")
best_ssim=$(grep -E "^pkl\s+\{" "$LOG_FILE" | tail -n1 | sed -E "s/.*'ssim': ([0-9.]+).*/\1/")

echo "Parsed pkl metrics: PSNR=${best_psnr:-NA}, SSIM=${best_ssim:-NA}"

fail=0
awk -v a="${best_psnr:-0}" -v b="$PSNR_MIN" 'BEGIN{exit !(a<b)}'
if [ $? -eq 0 ]; then
  echo "FAIL: PSNR ${best_psnr:-0} < threshold ${PSNR_MIN}"
  fail=1
fi

awk -v a="${best_ssim:-0}" -v b="$SSIM_MIN" 'BEGIN{exit !(a<b)}'
if [ $? -eq 0 ]; then
  echo "FAIL: SSIM ${best_ssim:-0} < threshold ${SSIM_MIN}"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  exit 2
fi

echo "Thresholds satisfied."
EOF
chmod +x "$LOG_DIR/06_threshold_check.sh"

tmux send-keys -t "$SESSION":run.3 "bash \"$LOG_DIR/06_threshold_check.sh\" \"$LOG_DIR/05_eval.log\" $THRESH_PSNR_MIN $THRESH_SSIM_MIN | tee \"$LOG_DIR/06_thresholds.log\"" C-m
tmux send-keys -t "$SESSION":run.3 "echo '== Evaluation complete =='; tmux display-message 'Evaluation and thresholds complete'" C-m

log "Session: $SESSION"
log "Logs: $LOG_DIR"
log "Attach with: tmux attach -t $SESSION"

echo "Launched tmux workflow."


