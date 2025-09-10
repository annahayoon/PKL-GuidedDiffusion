# 🚀 Cloud Server Setup Instructions for PKL-Diffusion

## Overview
This guide will help you transfer and run PKL-Diffusion training on cloud GPU servers using the **minimal transfer approach** for cost optimization.

## 📋 Pre-Flight Checklist

### ✅ Data Integrity Check
```bash
# Check data structure
ls -la data/real_microscopy/splits/
# Should show: train/, val/, test/ directories

# Verify data counts
find data/real_microscopy/splits/train -name "*.png" | wc -l
find data/real_microscopy/splits/val -name "*.png" | wc -l
find data/real_microscopy/splits/test -name "*.png" | wc -l
```

### ✅ Local Training Test
```bash
# Test that training script works locally
python scripts/train_real_data.py training.max_epochs=1 wandb.mode=disabled
```

### ✅ Dependencies Check
```bash
# Verify all required packages are installed
pip check
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🌐 Cloud Server Setup

### Step 1: Choose Cloud Provider

#### Option A: Thunder Compute (Recommended - Best Value)
- **H100**: $1.47/hour
- **A100**: $0.78/hour
- **Setup**: Create account → Launch GPU instance

#### Option B: Hyperstack
- **H100**: $1.90/hour
- **A100**: $1.35/hour
- **Setup**: Sign up → Create project → Launch instance

#### Option C: AWS/GCP/Azure
- **H100**: $3-5/hour
- **A100**: $2-3/hour
- **Setup**: More complex but enterprise-grade

### Step 2: Launch GPU Instance

#### Recommended Instance Specs:
```yaml
GPU: H100 (80GB) or A100 (80GB)
CPU: 8+ cores
RAM: 32GB+
Storage: 100GB+ SSD
OS: Ubuntu 20.04/22.04
```

#### Instance Launch Commands:
```bash
# For Thunder Compute
# 1. Go to dashboard
# 2. Click "Launch Instance"
# 3. Select H100 or A100
# 4. Choose Ubuntu 22.04
# 5. Set storage to 100GB
# 6. Launch instance
```

### Step 3: Connect to Instance

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Or if using password auth
ssh ubuntu@your-instance-ip
```

---

## 📦 Code Transfer (Minimal Approach)

### Method 1: Git Repository (Recommended)

#### On Local Machine:
```bash
# Initialize git if not already done
cd /home/jilab/anna_OS_ML/PKL-DiffusionDenoising
git init
git add pkl_dg/ scripts/ configs/ requirements.txt setup.py README.md
git commit -m "Initial commit for cloud training"

# Create .gitignore
cat > .gitignore << EOF
checkpoints/
logs/
outputs/
wandb/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.DS_Store
*.log
EOF

# Push to GitHub/GitLab
git remote add origin https://github.com/yourusername/pkl-diffusion.git
git push -u origin main
```

#### On Cloud Server:
```bash
# Clone repository
git clone https://github.com/yourusername/pkl-diffusion.git
cd pkl-diffusion
```

### Method 2: Direct Transfer (Alternative)

#### On Local Machine:
```bash
# Create minimal package
tar -czf pkl-diffusion-minimal.tar.gz \
  pkl_dg/ \
  scripts/ \
  configs/ \
  requirements.txt \
  setup.py \
  README.md

# Transfer to cloud server
scp pkl-diffusion-minimal.tar.gz ubuntu@your-instance-ip:~/
```

#### On Cloud Server:
```bash
# Extract package
tar -xzf pkl-diffusion-minimal.tar.gz
cd pkl-diffusion-minimal
```

---

## 📊 Data Transfer

### Transfer Real Microscopy Data

#### On Local Machine:
```bash
# Create data archive
tar -czf real_microscopy_data.tar.gz data/real_microscopy/

# Transfer data (this will take time - ~720MB)
scp real_microscopy_data.tar.gz ubuntu@your-instance-ip:~/
```

#### On Cloud Server:
```bash
# Extract data
tar -xzf real_microscopy_data.tar.gz
mkdir -p pkl-diffusion/data
mv data/real_microscopy pkl-diffusion/data/
```

---

## 🛠️ Environment Setup on Cloud Server

### Step 1: System Updates
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget htop tmux
```

### Step 2: Python Environment
```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Create virtual environment
cd pkl-diffusion
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Step 4: Verify Installation
```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"

# Test package import
python -c "from pkl_dg.models.diffusion import DDPMTrainer; print('Package imported successfully')"
```

---

## 🚀 Training Commands

### Fresh Training (Recommended)
```bash
# Start fresh training with optimal settings
tmux new-session -d -s training_session "python scripts/train_real_data.py \
  training.max_epochs=800 \
  training.num_workers=12 \
  training.batch_size=4 \
  wandb.mode=online \
  wandb.project=pkl-diffusion-cloud \
  experiment.name=cloud_training_\$(date +%Y%m%d_%H%M%S)"
```

### Monitor Training
```bash
# Attach to training session
tmux attach-session -t training_session

# Or monitor from outside
tmux capture-pane -t training_session -p | tail -10
```

### Resume Training (If you transfer checkpoints)
```bash
# Resume from specific checkpoint
tmux new-session -d -s resume_training "python scripts/resume_training_real.py \
  +resume_checkpoint=checkpoints/real_run1/epoch_0299_trainer.pt \
  +resume_start_epoch=299 \
  training.max_epochs=800 \
  training.num_workers=12 \
  wandb.mode=online"
```

---

## 📈 Monitoring & Logging

### TensorBoard Setup
```bash
# Start TensorBoard in background
tmux new-session -d -s tensorboard "tensorboard --logdir=logs --host=0.0.0.0 --port=6006"

# Access TensorBoard
# Open browser to: http://your-instance-ip:6006
```

### Weights & Biases Setup
```bash
# Login to W&B
wandb login

# Your training will automatically log to W&B
# Check: https://wandb.ai/your-username/pkl-diffusion-cloud
```

---

## 💾 Checkpoint Management

### Save Checkpoints
```bash
# Checkpoints are automatically saved to:
ls -la checkpoints/

# Best model
ls -la checkpoints/best_model.pt

# Final model
ls -la checkpoints/final_model.pt
```

### Download Checkpoints
```bash
# From local machine, download trained model
scp ubuntu@your-instance-ip:~/pkl-diffusion/checkpoints/best_model.pt ./
scp ubuntu@your-instance-ip:~/pkl-diffusion/checkpoints/final_model.pt ./
```

---

## 🧪 Evaluation

### Run Evaluation
```bash
# Evaluate on test set
python scripts/evaluate.py \
  inference.checkpoint_path=checkpoints/best_model.pt \
  inference.input_dir=data/real_microscopy/splits/test/wf \
  inference.output_dir=outputs/evaluation \
  wandb.mode=online
```

### Download Results
```bash
# Download evaluation results
scp -r ubuntu@your-instance-ip:~/pkl-diffusion/outputs/evaluation ./
```

---

## 💰 Cost Optimization Tips

### 1. Use Spot Instances
- **Savings**: 50-70% cost reduction
- **Risk**: Can be interrupted
- **Mitigation**: Save checkpoints frequently

### 2. Monitor Usage
```bash
# Check GPU utilization
nvidia-smi

# Check instance uptime
uptime

# Monitor costs in cloud dashboard
```

### 3. Auto-shutdown Script
```bash
# Create auto-shutdown script
cat > auto_shutdown.sh << 'EOF'
#!/bin/bash
# Shutdown if no training process for 30 minutes
while true; do
    if ! pgrep -f "python scripts/train" > /dev/null; then
        echo "No training process found, shutting down in 30 minutes..."
        sleep 1800
        if ! pgrep -f "python scripts/train" > /dev/null; then
            sudo shutdown -h now
        fi
    fi
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x auto_shutdown.sh
# Run in background: nohup ./auto_shutdown.sh &
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train_real_data.py training.batch_size=2

# Reduce num_workers
python scripts/train_real_data.py training.num_workers=4
```

#### 2. Import Errors
```bash
# Reinstall package
pip uninstall pkl-diffusion-denoising
pip install -e .
```

#### 3. Data Loading Issues
```bash
# Check data paths
ls -la data/real_microscopy/splits/
python -c "from pkl_dg.data.real_pairs import RealPairsDataset; print('Data loading works')"
```

#### 4. Connection Issues
```bash
# Keep connection alive
ssh -o ServerAliveInterval=60 ubuntu@your-instance-ip

# Use tmux for persistent sessions
tmux new-session -d -s main
```

---

## 📞 Support

### Emergency Commands
```bash
# Kill all training processes
pkill -f "python scripts/train"

# Check system resources
htop
nvidia-smi
df -h

# View logs
tail -f logs/training.log
```

### Backup Strategy
```bash
# Create backup of important files
tar -czf backup_$(date +%Y%m%d).tar.gz checkpoints/ logs/ outputs/
```

---

## 🎯 Expected Results

### Training Time Estimates:
- **H100**: ~12-15 hours for 500 epochs
- **A100**: ~1.5-2 days for 500 epochs
- **Cost**: $18-45 total

### Performance Targets:
- **Training Loss**: < 0.01
- **Validation Loss**: < 0.02
- **PSNR**: > 30 dB
- **SSIM**: > 0.9

---

## ✅ Completion Checklist

- [ ] Cloud instance launched
- [ ] Code transferred
- [ ] Data transferred
- [ ] Environment setup
- [ ] Training started
- [ ] Monitoring active
- [ ] Checkpoints saved
- [ ] Evaluation completed
- [ ] Results downloaded
- [ ] Instance terminated

---

**Happy Training! 🚀**
