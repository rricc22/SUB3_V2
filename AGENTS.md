# Agent Guidelines - SUB3_V2

## Project Context

**Goal**: Predict heart rate time-series from speed/altitude during running workouts  
**Dataset**: Endomondo HR (974 valid running workouts after quality filtering)  
**Target Metric**: MAE < 10 BPM (base model), < 7 BPM (finetuned)

---

## Quick Commands

```bash
# Data quality validation
streamlit run EDA/quality_annotation_app.py          # Manual annotation interface
python3 EDA/analyze_quality.py                       # Quality statistics

# Preprocessing
python3 Preprocessing/prepare_sequences.py           # Full pipeline with feature engineering
python3 Preprocessing/quality_filters.py --validate  # Run quality checks on all data

# Model training
python3 Model/train.py --model lstm --epochs 100 --batch_size 16 --use_masking
python3 Model/train.py --hidden_size 128 --dropout 0.3  # V2 recommended config

# Evaluation
python3 Model/evaluate.py --checkpoint checkpoints/best_model.pt
python3 Model/compare_with_v1.py  # Compare V2 vs V1 baselines
```

---

## Code Style (PEP 8)

- **Naming**: `snake_case` functions/vars, `CamelCase` classes, ALL_CAPS constants
- **Imports**: stdlib → third-party (numpy, pandas, torch, sklearn) → local; blank lines between groups
- **Docstrings**: Triple quotes `"""` at module/function level with Args/Returns sections
- **Type hints**: Use for function signatures (Optional, List, Dict)
- **Error handling**: try-except with informative messages; log errors, don't silently fail
- **File headers**: Shebang `#!/usr/bin/env python3` + module docstring

---

## V2 Specific Guidelines

### Feature Engineering

When adding new features:

1. **Document physiological rationale** (why this feature?)
2. **Test correlation** before implementing (EDA first)
3. **Handle edge cases** (first/last timesteps for lag features)
4. **Normalize consistently** (same scaler for base + lag features)

Example:
```python
def create_lag_features(speed, lag=2):
    """
    Create lagged speed features.
    
    Rationale: HR responds to speed with ~2 timestep delay (from EDA).
    
    Args:
        speed: [seq_len] array
        lag: Number of timesteps to shift
    
    Returns:
        speed_lag: [seq_len] array (first `lag` values = speed[0])
    """
    speed_lag = np.roll(speed, lag)
    speed_lag[:lag] = speed[0]  # Fill beginning with first value
    return speed_lag
```

### Masking

Always use masks when computing loss/metrics on padded sequences:

```python
# Correct (masked)
loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum()

# Incorrect (unmasked - includes padding pollution)
loss = ((predictions - targets) ** 2).mean()
```

### Data Quality

Before preprocessing, validate data quality:

```python
# Check quality
is_valid, score, issues = validate_workout_quality(workout)

if not is_valid:
    logger.warning(f"Workout {id} failed quality check: {issues}")
    continue  # Skip low-quality data
```

---

## Project Structure Navigation

### Input Features (11 total)

**Base** (3):
- `speed` [km/h]: Running speed from GPS
- `altitude` [m]: Elevation from GPS/barometer
- `gender` [binary]: 1=male, 0=female

**Lag** (3):
- `speed_lag_2`: Speed 2 timesteps ago (physiological HR delay)
- `speed_lag_5`: Speed 5 timesteps ago (sustained effort)
- `altitude_lag_30`: Altitude 30 timesteps ago (elevation HR lag)

**Derivatives** (2):
- `speed_derivative`: Acceleration/deceleration (speed[t] - speed[t-1])
- `altitude_derivative`: Elevation change rate (altitude[t] - altitude[t-1])

**Rolling** (2):
- `rolling_speed_10`: Moving average (10 timesteps, ~1 min)
- `rolling_speed_30`: Moving average (30 timesteps, ~3 min)

**Cumulative** (1):
- `cumulative_elevation`: Total elevation gain (sum of positive altitude changes)

### Output

- `heart_rate` [BPM]: Time-series target (unnormalized for interpretable loss)

### Data Format

**Preprocessed tensors**:
```python
{
    'features': [N, 500, 11],         # All features concatenated
    'heart_rate': [N, 500, 1],        # Target
    'mask': [N, 500, 1],              # 1=valid, 0=padded (NEW in V2)
    'gender': [N, 1],                 # Metadata
    'userId': [N, 1],                 # For embeddings
    'original_lengths': [N, 1],       # For reference
}
```

---

## Model Architecture (V2)

### LSTM Configuration

```python
HeartRateLSTM_v2(
    input_size=11,        # 11 features (was 3 in V1)
    hidden_size=128,      # Increased from 64
    num_layers=2,         # Same as V1
    dropout=0.3,          # Increased from 0.2
    bidirectional=False   # Same as V1
)
```

**Parameters**: ~180K (vs ~50K in V1)

### Training Configuration

```python
TRAINING_CONFIG = {
    'batch_size': 16,        # Optimal from V1
    'learning_rate': 0.0005, # Halved due to more params
    'epochs': 100,
    'patience': 10,          # Early stopping
    'weight_decay': 1e-4,    # L2 regularization
    'gradient_clip': 1.0,    # Prevent exploding gradients
    'use_masking': True,     # NEW: Ignore padded regions
}
```

---

## Baseline Metrics (V1 Reference)

| Model | MAE (BPM) | Status |
|-------|-----------|--------|
| Finetuned Stage 1 | **8.94** | Best (V1) |
| Finetuned Stage 2 | 10.15 | Excellent |
| LSTM Baseline | 13.88 | Good |
| GRU | 14.23 | Good |
| LSTM + Embeddings | 15.79 | OK |
| Lag-Llama | 38.08 | Poor |

**V2 Target**: MAE < 10 BPM (base model, no finetuning)

---

## Common Pitfalls

### 1. Forgetting to Mask Padded Regions

```python
# ❌ WRONG
loss = criterion(predictions, heart_rate)

# ✅ CORRECT
loss = criterion(predictions, heart_rate, mask)
```

### 2. Normalizing Features Inconsistently

```python
# ❌ WRONG (speed normalized, speed_lag_2 not normalized)
speed_norm = scaler.transform(speed)
speed_lag_2_raw = np.roll(speed, 2)  # Uses raw speed!

# ✅ CORRECT (create lag features AFTER normalization)
speed_norm = scaler.transform(speed)
speed_lag_2_norm = np.roll(speed_norm, 2)
```

### 3. Leaking Test Data into Normalization

```python
# ❌ WRONG (fit on all data)
scaler = StandardScaler().fit(np.concatenate([train, val, test]))

# ✅ CORRECT (fit on train only)
scaler = StandardScaler().fit(train)
train_norm = scaler.transform(train)
val_norm = scaler.transform(val)
test_norm = scaler.transform(test)
```

### 4. Not Handling Edge Cases in Lag Features

```python
# ❌ WRONG (first 2 values are garbage from rolling)
speed_lag_2 = np.roll(speed, 2)

# ✅ CORRECT (fill beginning with first value)
speed_lag_2 = np.roll(speed, 2)
speed_lag_2[:2] = speed[0]
```

---

## Testing Strategy

### Unit Tests

```bash
pytest Preprocessing/test_feature_engineering.py  # Feature creation
pytest Model/test_loss.py                         # Masked loss correctness
pytest Preprocessing/test_quality_filters.py      # Validation functions
```

### Integration Tests

```bash
pytest tests/test_preprocessing_pipeline.py  # End-to-end preprocessing
pytest tests/test_training_loop.py           # Training convergence
```

### Validation Tests

```bash
python3 Model/ablation_study.py  # Remove features one-by-one
python3 Model/validate_on_synthetic.py  # Known patterns
```

---

## Documentation Requirements

When implementing new features:

1. **Module docstring** - Purpose, inputs, outputs
2. **Function docstrings** - Args, Returns, Raises
3. **Inline comments** - For non-obvious logic
4. **Update CHANGELOG.md** - Record changes
5. **Update README.md** - If user-facing

Example:
```python
#!/usr/bin/env python3
"""
Feature engineering module for heart rate prediction.

Creates temporal features (lag, derivatives, rolling stats) from raw
speed and altitude sequences.

Author: OpenCode
Date: 2025-01-10
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

def engineer_features(
    speed: np.ndarray,
    altitude: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Create 8 engineered features from raw speed and altitude.
    
    Features:
        - Lag: speed_lag_2, speed_lag_5, altitude_lag_30
        - Derivatives: speed_derivative, altitude_derivative
        - Rolling: rolling_speed_10, rolling_speed_30
        - Cumulative: cumulative_elevation
    
    Args:
        speed: [seq_len] array of speed in km/h
        altitude: [seq_len] array of altitude in meters
    
    Returns:
        Dictionary with 8 feature arrays, each [seq_len]
    
    Raises:
        ValueError: If speed and altitude have different lengths
    
    Example:
        >>> speed = np.array([10, 11, 12, 11, 10])
        >>> altitude = np.array([100, 105, 110, 108, 105])
        >>> features = engineer_features(speed, altitude)
        >>> features.keys()
        dict_keys(['speed_lag_2', 'speed_lag_5', ...])
    """
    if len(speed) != len(altitude):
        raise ValueError("Speed and altitude must have same length")
    
    # Implementation...
```

---

## Debugging Tips

### Visualize Predictions

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(heart_rate.numpy(), label='Actual HR', alpha=0.7)
plt.plot(predictions.detach().numpy(), label='Predicted HR', alpha=0.7)
plt.axvline(original_length, color='r', linestyle='--', label='Original End')
plt.legend()
plt.savefig('results/debug_prediction.png')
```

### Check Gradients

```python
# Gradient norm (should be <1.0 with clipping)
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

### Validate Masking

```python
# Check that padded regions have mask=0
assert torch.all(mask[:, original_length:] == 0)
assert torch.all(mask[:, :original_length] == 1)
```

---

## Performance Targets

| Metric | V1 Baseline | V2 Target | How to Measure |
|--------|-------------|-----------|----------------|
| MAE | 13.88 BPM | < 10 BPM | `torch.abs(pred - target).mean()` |
| R² | 0.188 | > 0.35 | `1 - SS_res / SS_tot` |
| Training time | ~20 min | < 30 min | Wall clock (100 epochs) |
| Correlation | 0.25 | > 0.40 | `np.corrcoef(speed, hr)[0,1]` |

---

## Git Workflow

```bash
# Feature branch
git checkout -b feature/lag-features

# Commit with descriptive message
git commit -m "feat: Add lag features (speed_lag_2, altitude_lag_30)

- Implemented in Preprocessing/feature_engineering.py
- Added unit tests
- EDA shows correlation improvement: 0.25 -> 0.32
"

# Push and create PR
git push origin feature/lag-features
```

---

## Contact & Support

- **Project Lead**: Riccardo
- **V1 Reference**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Documentation**: `docs/` folder (PRD, Architecture, Data Quality)

---

**Last Updated**: 2025-01-10
