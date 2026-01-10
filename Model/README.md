# Model Directory

## Overview

LSTM-based heart rate prediction with feature engineering and masking.

**Key improvements in V2**:
- Input size: 3 → 11 features
- Hidden size: 64 → 128 units
- Dropout: 0.2 → 0.3
- Masked loss (ignores padded regions)
- Learning rate: 0.001 → 0.0005

## Scripts

### `lstm.py`

Model definition with 11-feature input.

**Architecture**:
```python
HeartRateLSTM_v2(
    input_size=11,        # 11 features
    hidden_size=128,      # Increased from 64
    num_layers=2,
    dropout=0.3,          # Increased from 0.2
    bidirectional=False
)
```

**Parameters**: ~180K (vs ~50K in V1)

**Forward pass**:
```python
predictions = model(features)  # [batch, seq_len, 1]
```

### `train.py`

Training loop with masking and early stopping.

**Usage**:
```bash
# Basic training
python3 Model/train.py --model lstm --epochs 100 --batch_size 16

# With custom config
python3 Model/train.py \
    --hidden_size 128 \
    --dropout 0.3 \
    --lr 0.0005 \
    --use_masking \
    --patience 10
```

**Arguments**:
- `--model`: Model type (lstm, gru)
- `--hidden_size`: LSTM hidden dimension (default: 128)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.3)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.0005)
- `--epochs`: Max epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)
- `--use_masking`: Enable masked loss (default: True)

**Output**:
- `checkpoints/best_model.pt` - Best validation model
- `results/training_curves.png` - Loss/MAE plots
- `logs/training_YYYYMMDD_HHMMSS.log` - Training logs

### `loss.py`

Masked MSE loss function.

**Usage**:
```python
from loss import MaskedMSELoss

criterion = MaskedMSELoss()
loss = criterion(predictions, targets, mask)
```

**Implementation**:
```python
class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, mask):
        loss = (pred - target) ** 2  # [batch, seq, 1]
        masked_loss = loss * mask    # Zero out padding
        return masked_loss.sum() / mask.sum()
```

### `evaluate.py`

Model evaluation on test set.

**Usage**:
```bash
python3 Model/evaluate.py --checkpoint checkpoints/best_model.pt
```

**Output**:
- MAE, RMSE, R² metrics
- Per-sample predictions
- Visualization plots

## Training Configuration

**Recommended V2 settings**:

```python
TRAINING_CONFIG = {
    # Model
    'input_size': 11,
    'hidden_size': 128,      # Increased
    'num_layers': 2,
    'dropout': 0.3,          # Increased
    'bidirectional': False,
    
    # Training
    'batch_size': 16,        # Optimal from V1
    'learning_rate': 0.0005, # Halved
    'epochs': 100,
    'patience': 10,
    'weight_decay': 1e-4,
    'gradient_clip': 1.0,
    
    # Loss
    'use_masking': True,     # NEW: Required
    
    # Optimizer
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
}
```

## Model Comparison

### V1 Baseline Results

| Model | MAE (BPM) | R² | Params |
|-------|-----------|-----|--------|
| Finetuned Stage 1 | 8.94 | 0.279 | ~50K |
| LSTM Baseline | 13.88 | 0.188 | ~50K |
| GRU | 14.23 | 0.156 | ~40K |

### V2 Targets

| Metric | V1 Baseline | V2 Target | Improvement |
|--------|-------------|-----------|-------------|
| MAE (base) | 13.88 BPM | < 10 BPM | -28% |
| R² | 0.188 | > 0.35 | +86% |
| Training time | ~20 min | < 30 min | Similar |

## Evaluation Metrics

### Primary Metrics

1. **MAE (Mean Absolute Error)**
```python
mae = torch.abs(predictions - targets).mean()  # In BPM
```
Target: < 10 BPM

2. **RMSE (Root Mean Squared Error)**
```python
rmse = torch.sqrt(((predictions - targets) ** 2).mean())
```
Penalizes large errors

3. **R² (Coefficient of Determination)**
```python
r2 = 1 - (SS_residual / SS_total)
```
Target: > 0.35

### Secondary Metrics

4. **Per-timestep MAE** - Detect prediction drift over workout
5. **Per-user MAE** - Check generalization across users
6. **Zone accuracy** - % timesteps in correct HR zone

## Training Process

### 1. Load Data

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load preprocessed tensors
train_data = torch.load('DATA/processed/train.pt')

# Create dataset
dataset = TensorDataset(
    train_data['features'],
    train_data['heart_rate'],
    train_data['mask']
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 2. Initialize Model

```python
from lstm import HeartRateLSTM_v2
from loss import MaskedMSELoss

model = HeartRateLSTM_v2(
    input_size=11,
    hidden_size=128,
    num_layers=2,
    dropout=0.3
).to(device)

criterion = MaskedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### 3. Training Loop

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for features, heart_rate, mask in dataloader:
        features = features.to(device)
        heart_rate = heart_rate.to(device)
        mask = mask.to(device)
        
        # Forward
        predictions = model(features)
        loss = criterion(predictions, heart_rate, mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    val_loss, val_mae = validate(model, val_dataloader, criterion)
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### 4. Evaluation

```python
# Load best model
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()

# Test set evaluation
test_loss, test_mae, test_rmse, test_r2 = evaluate(
    model, test_dataloader, criterion
)

print(f"Test MAE: {test_mae:.2f} BPM")
print(f"Test RMSE: {test_rmse:.2f} BPM")
print(f"Test R²: {test_r2:.3f}")
```

## Ablation Study

Test contribution of each feature:

```bash
# Baseline (3 features)
python3 Model/train.py --features speed,altitude,gender

# + Lag features
python3 Model/train.py --features speed,altitude,gender,speed_lag_2,altitude_lag_30

# + Derivatives
python3 Model/train.py --features speed,altitude,gender,speed_derivative

# + Rolling stats
python3 Model/train.py --features speed,altitude,gender,rolling_speed_10

# All features (11)
python3 Model/train.py --features all
```

## Debugging

### Check Gradients

```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")  # Should be <1.0
```

### Visualize Predictions

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(heart_rate[0, :, 0].cpu(), label='Actual', alpha=0.7)
plt.plot(predictions[0, :, 0].detach().cpu(), label='Predicted', alpha=0.7)
plt.axvline(original_length[0], color='r', linestyle='--', label='Padding starts')
plt.legend()
plt.savefig('results/sample_prediction.png')
```

### Validate Masking

```python
# Check mask correctness
original_len = original_lengths[0].item()
assert torch.all(mask[0, :original_len] == 1.0)
assert torch.all(mask[0, original_len:] == 0.0)
print("Masking validated ✓")
```

## Performance Optimization

### GPU Utilization

```python
# Check GPU usage
nvidia-smi

# Expected: ~80% GPU utilization during training
```

### DataLoader Tuning

```python
# Faster dataloading
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,      # Parallel workers
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

## Checkpointing

**Saved checkpoint format**:

```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_loss': val_loss,
    'val_mae': val_mae,
    'config': {
        'input_size': 11,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
    }
}
```

**Load checkpoint**:

```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Val MAE: {checkpoint['val_mae']:.2f} BPM")
```
