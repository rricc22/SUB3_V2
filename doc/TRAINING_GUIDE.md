# SUB3_V2 Training Guide

Complete guide for training the heart rate prediction model on your desktop GPU.

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- 16GB+ RAM
- ~2GB disk space for preprocessed data

### Software
- Python 3.11+
- CUDA 11.8+ (for GPU support)
- conda or pip

---

## Setup Instructions

### 1. Transfer Preprocessed Data to Desktop

You need to transfer the following files from this laptop to your desktop:

```bash
# Files to transfer (total ~1.6GB):
DATA/processed/train.pt          # 814MB
DATA/processed/val.pt            # 182MB
DATA/processed/test.pt           # 153MB
DATA/processed/metadata.json     # 728 bytes
DATA/processed/scaler_params.json # 891 bytes
```

**Option A: USB Drive**
```bash
# On laptop
cp -r DATA/processed /path/to/usb/SUB3_V2_data/

# On desktop
cp -r /path/to/usb/SUB3_V2_data/processed /path/to/SUB3_V2/DATA/
```

**Option B: Network Transfer (rsync)**
```bash
# From laptop to desktop (replace with your desktop IP/hostname)
rsync -avz --progress DATA/processed/ user@desktop:/path/to/SUB3_V2/DATA/processed/
```

**Option C: Cloud (Google Drive, Dropbox, etc.)**
- Upload `DATA/processed/` folder to cloud
- Download on desktop

### 2. Install Dependencies on Desktop

```bash
cd /path/to/SUB3_V2

# Option 1: Using conda (recommended)
conda create -n sub3_v2 python=3.11
conda activate sub3_v2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# Option 2: Using pip only
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. Verify GPU Setup

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
```

### 4. Login to Weights & Biases

```bash
wandb login
# Enter your API key when prompted
# Get key from: https://wandb.ai/authorize
```

---

## Training the Model

### Quick Start (Recommended Settings)

```bash
cd Model

python3 train.py \
  --data-dir ../DATA/processed \
  --hidden-size 128 \
  --num-layers 2 \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.001 \
  --loss-fn mse \
  --patience 10 \
  --checkpoint-dir ../checkpoints \
  --wandb-project heart-rate-prediction \
  --run-name "baseline_v2"
```

**Expected training time:** 20-30 minutes on RTX 3080

### Monitor Training

1. **Terminal**: Progress bars show real-time loss/MAE
2. **W&B Dashboard**: Open browser to https://wandb.ai and view:
   - Training curves
   - Validation metrics
   - System metrics (GPU usage, etc.)

### Training Arguments Explained

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-size` | 128 | LSTM hidden units (64/128/256) |
| `--num-layers` | 2 | Number of LSTM layers |
| `--dropout` | 0.3 | Dropout rate (0.2-0.4) |
| `--epochs` | 100 | Max training epochs |
| `--batch-size` | 16 | Batch size (adjust for GPU) |
| `--lr` | 0.001 | Learning rate |
| `--loss-fn` | mse | Loss function (mse/mae) |
| `--patience` | 10 | Early stopping patience |
| `--bidirectional` | False | Use bidirectional LSTM |

### Hyperparameter Tuning

Try these configurations:

```bash
# Configuration 1: Larger model
python3 train.py --hidden-size 256 --run-name "large_model"

# Configuration 2: More layers
python3 train.py --num-layers 3 --run-name "deep_model"

# Configuration 3: Bidirectional
python3 train.py --bidirectional --run-name "bidirectional"

# Configuration 4: Lower learning rate
python3 train.py --lr 0.0005 --run-name "lr_0.0005"

# Configuration 5: MAE loss
python3 train.py --loss-fn mae --run-name "mae_loss"
```

---

## Evaluation

After training, evaluate the best model on the test set:

```bash
python3 evaluate.py \
  --checkpoint ../checkpoints/best_model.pt \
  --data-dir ../DATA/processed \
  --output-dir ../results
```

**Outputs:**
- `results/test_results.json` - Metrics (MAE, RMSE)
- `results/sample_predictions.png` - Sample predictions
- `results/error_distribution.png` - Error histograms
- `results/per_sample_mae.png` - Per-sample MAE
- `results/scatter_plot.png` - Predicted vs actual

---

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA out of memory errors:

```bash
# Reduce batch size
python3 train.py --batch-size 8

# Or even smaller
python3 train.py --batch-size 4
```

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Force CPU training (slow but works)
python3 train.py --cpu
```

### W&B Connection Issues

```bash
# Use offline mode
python3 train.py --no-wandb

# Logs will be saved locally but not synced to cloud
```

### Slow Training

- **Check GPU usage**: `nvidia-smi` should show high utilization
- **Reduce workers**: `--num-workers 2` or `--num-workers 0`
- **Check data location**: Ensure data is on fast SSD, not HDD

---

## Expected Results

Based on V1 baseline and V2 improvements:

| Metric | V1 Baseline | V2 Target | Notes |
|--------|-------------|-----------|-------|
| Test MAE | 13.88 BPM | < 10 BPM | Primary metric |
| Test RMSE | ~18 BPM | < 14 BPM | Secondary metric |
| R² | 0.188 | > 0.35 | Explained variance |
| Training time | ~20 min | ~20-30 min | On RTX 3080 |

**V2 Improvements:**
1. ✅ 11 features vs 3 (temporal features)
2. ✅ Masked loss (no padding pollution)
3. ✅ Cleaner data (3-stage pipeline)
4. ✅ User-based stratified split

---

## Advanced Usage

### Resume Training from Checkpoint

```python
# Modify train.py to add:
# --resume flag to load checkpoint and continue training
```

### Multi-GPU Training

If you have multiple GPUs:

```python
# Add DataParallel wrapper in train.py:
model = nn.DataParallel(model)
```

### Custom Experiments

Create experiment config files:

```yaml
# configs/experiment_1.yaml
hidden_size: 256
num_layers: 3
dropout: 0.4
lr: 0.0005
```

---

## Next Steps

1. **Train baseline model** with recommended settings
2. **Compare with V1** results (13.88 BPM MAE)
3. **Hyperparameter tuning** to optimize performance
4. **Ablation study** to understand feature importance
5. **Write final report** documenting V2 improvements

---

## Quick Reference

```bash
# Training
python3 Model/train.py --data-dir DATA/processed --wandb-project heart-rate-prediction

# Evaluation
python3 Model/evaluate.py --checkpoint checkpoints/best_model.pt --data-dir DATA/processed

# Check GPU
nvidia-smi

# Monitor training
# Open browser → https://wandb.ai → Select project

# View logs
tail -f logs/training.log  # If you add file logging
```

---

## Support

For issues:
1. Check this guide
2. Review error messages
3. Check W&B dashboard for training curves
4. Verify GPU is being used (`nvidia-smi`)

**Good luck with training! 🚀**
