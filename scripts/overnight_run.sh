#!/usr/bin/env bash
set -euo pipefail

# Log file
LOG_DIR=/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Starting overnight run at $(date)"

# Activate venv
VENV=/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/.venv
if [ -f "$VENV/bin/activate" ]; then
  . "$VENV/bin/activate"
  echo "[INFO] Activated venv: $VENV"
else
  echo "[WARN] No venv found at $VENV; continuing with system Python"
fi

# User-configurable paths
PROJ=/home/jilab/anna_OS_ML/PKL-DiffusionDenoising
WF_TILES_DIR="$PROJ/data/real_microscopy/splits/test/wf"
RCAN_CKPT="${RCAN_CKPT:-}"  # Allow env override; if empty, skip RCAN

export PROJ WF_TILES_DIR RCAN_CKPT

# Ensure project root is current working dir and on PYTHONPATH
cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# 1) Diffusion (conditioned)
RUN_NAME_DIFF_COND="diff_cond_$(date +%Y-%m-%d_%H-%M-%S)"
python "$PROJ/scripts/train_real_data.py" \
  experiment.name="$RUN_NAME_DIFF_COND" \
  training.use_conditioning=true \
  experiment.mixed_precision=false \
  training.learning_rate=5e-5 \
  training.gradient_clip=1.0 \
  training.supervised_x0_weight=0.05 \
  paths.checkpoints="$PROJ/checkpoints" \
  wandb.mode=disabled || { echo "[ERROR] Conditioned diffusion training failed"; exit 1; }

python "$PROJ/scripts/inference.py" \
  experiment.name="$RUN_NAME_DIFF_COND" \
  training.use_conditioning=true \
  inference.checkpoint_path="$PROJ/checkpoints/best_model.pt" \
  inference.input_dir="$WF_TILES_DIR" \
  inference.output_dir="$PROJ/outputs/tmp_eval_png/$RUN_NAME_DIFF_COND" \
  guidance.type=pkl guidance.lambda_base=0.1 \
  guidance.schedule.T_threshold=800 guidance.schedule.epsilon_lambda=1e-3 \
  inference.ddim_steps=20 inference.eta=0.0 \
  experiment.device=cuda || { echo "[ERROR] Conditioned diffusion inference failed"; exit 1; }

# 2) Diffusion (unconditioned)
RUN_NAME_DIFF_UNCOND="diff_uncond_$(date +%Y-%m-%d_%H-%M-%S)"
python "$PROJ/scripts/train_real_data.py" \
  experiment.name="$RUN_NAME_DIFF_UNCOND" \
  training.use_conditioning=false \
  model.in_channels=1 \
  experiment.mixed_precision=false \
  training.learning_rate=5e-5 \
  training.gradient_clip=1.0 \
  training.supervised_x0_weight=0.05 \
  paths.checkpoints="$PROJ/checkpoints" \
  wandb.mode=disabled || { echo "[ERROR] Unconditioned diffusion training failed"; exit 1; }

python "$PROJ/scripts/inference.py" \
  experiment.name="$RUN_NAME_DIFF_UNCOND" \
  training.use_conditioning=false \
  inference.checkpoint_path="$PROJ/checkpoints/best_model.pt" \
  inference.input_dir="$WF_TILES_DIR" \
  inference.output_dir="$PROJ/outputs/tmp_eval_png/$RUN_NAME_DIFF_UNCOND" \
  guidance.type=pkl guidance.lambda_base=0.1 \
  guidance.schedule.T_threshold=800 guidance.schedule.epsilon_lambda=1e-3 \
  inference.ddim_steps=20 inference.eta=0.0 \
  experiment.device=cuda || { echo "[ERROR] Unconditioned diffusion inference failed"; exit 1; }

# 3) Supervised U-Net baseline
RUN_NAME_SUP="sup_unet_$(date +%Y-%m-%d_%H-%M-%S)"
python - <<'PY' || { echo "[ERROR] SupUNet training failed"; exit 1; }
import os
from pkl_dg.baselines import train_supervised_unet
PROJ=os.environ["PROJ"]; RUN_NAME=os.environ["RUN_NAME_SUP"]
cfg={"run_name":RUN_NAME,"device":"cuda",
     "paths":{"checkpoints":f"{PROJ}/checkpoints","outputs":f"{PROJ}/outputs"},
     "data":{"data_dir":f"{PROJ}/data/real_microscopy","image_size":256,"min_intensity":0,"max_intensity":255},
     "training":{"batch_size":8,"num_workers":4,"max_epochs":60,"learning_rate":2e-4},
     "model_channels":64,"l1_weight":1.0}
train_supervised_unet(cfg)
PY

python - <<'PY' || { echo "[ERROR] SupUNet inference failed"; exit 1; }
import os
from pathlib import Path
from pkl_dg.baselines import infer_supervised_unet
PROJ=os.environ["PROJ"]; RUN_NAME=os.environ["RUN_NAME_SUP"]
cfg={"run_name":RUN_NAME,"device":"cuda",
     "paths":{"checkpoints":f"{PROJ}/checkpoints","outputs":f"{PROJ}/outputs"},
     "data":{"data_dir":f"{PROJ}/data/real_microscopy","image_size":256,"min_intensity":0,"max_intensity":255},
     "model_channels":64}
ckpt=Path(f"{PROJ}/checkpoints/{RUN_NAME}_sup_unet.pt")
infer_supervised_unet(cfg, ckpt, Path(os.environ["WF_TILES_DIR"]), Path(f"{PROJ}/outputs/tmp_eval_png/{RUN_NAME}"))
PY

# 4) RCAN baseline (optional)
if [ -n "${RCAN_CKPT}" ] && [ -f "${RCAN_CKPT}" ]; then
  RUN_NAME_RCAN="rcan_$(date +%Y-%m-%d_%H-%M-%S)"
  python "$PROJ/scripts/infer_rcan.py" \
    --checkpoint "$RCAN_CKPT" \
    --input-dir "$WF_TILES_DIR" \
    --output-dir "$PROJ/outputs/rcan_tiles/$RUN_NAME_RCAN" \
    --outputs-root "$PROJ/outputs" \
    --run-name "$RUN_NAME_RCAN" \
    --device cuda \
    --normalize --min-intensity 0 --max-intensity 255 || echo "[WARN] RCAN inference skipped/failed"
else
  echo "[INFO] Skipping RCAN (no checkpoint provided)"
fi

# 5) RL baseline
RUN_NAME_RL="rl_$(date +%Y-%m-%d_%H-%M-%S)"
python "$PROJ/scripts/infer_rl.py" \
  --input-dir "$WF_TILES_DIR" \
  --output-dir "$PROJ/outputs/rl_tiles/$RUN_NAME_RL" \
  --outputs-root "$PROJ/outputs" \
  --run-name "$RUN_NAME_RL" \
  --iters 30 || echo "[WARN] RL inference failed"

echo "[DONE] All runs completed at $(date)"


