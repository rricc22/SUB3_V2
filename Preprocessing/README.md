# Preprocessing Pipeline

## Overview

Transforms raw Endomondo JSON data into PyTorch tensors ready for training.

**Key improvements in V2**:
- Quality validation (manual + automated)
- Feature engineering (3 → 11 features)
- Masking (validity masks for padded regions)
- Stratified user splitting (balanced fitness levels)

## Pipeline Steps

```
1. Load raw JSON
2. Quality validation
3. Filter valid workouts
4. Feature engineering
5. Pad/truncate to 500 timesteps
6. Generate masks
7. Normalize features
8. User-based splitting (stratified)
9. Convert to PyTorch tensors
10. Save to disk
```

## Scripts

### `prepare_sequences.py`

Main preprocessing pipeline.

**Usage**:
```bash
python3 Preprocessing/prepare_sequences.py
```

**Output**: 
- `DATA/processed/train.pt`
- `DATA/processed/val.pt`
- `DATA/processed/test.pt`
- `DATA/processed/metadata.json`
- `DATA/processed/scaler_params.json`

**Configuration** (top of file):
```python
SEQUENCE_LENGTH = 500       # Pad/truncate to this length
MAX_SAMPLES = None          # None = load all workouts
RANDOM_SEED = 42            # Reproducibility
MIN_QUALITY_SCORE = 70      # NEW: Quality threshold (0-100)
```

### `feature_engineering.py`

Creates engineered features from raw speed/altitude.

**Functions**:
- `create_lag_features()` - Lag features (2, 5, 30 timesteps)
- `create_derivative_features()` - Speed/altitude rate of change
- `create_rolling_features()` - Moving averages (10, 30 timesteps)
- `create_cumulative_features()` - Total elevation gain

**Usage**:
```python
from feature_engineering import engineer_features

features = engineer_features(speed, altitude)
# Returns dict with 8 engineered feature arrays
```

### `quality_filters.py`

Data quality validation functions.

**Functions**:
- `validate_hr_sensor()` - Detect spikes, flatlines, dropouts
- `validate_gps_speed()` - Detect impossible speeds, GPS noise
- `validate_altitude()` - Detect altitude jumps
- `validate_timestamps()` - Check sampling regularity
- `validate_activity_type()` - Verify workout labels

**Usage**:
```python
from quality_filters import validate_workout_quality

is_valid, score, issues = validate_workout_quality(workout)

if not is_valid:
    print(f"Workout failed: {issues}")
```

## Feature Engineering Details

### Lag Features

**Rationale**: HR responds to exercise intensity with a physiological delay.

```python
# Speed lag 2 (HR responds ~2 timesteps after speed change)
speed_lag_2 = np.roll(speed, 2)
speed_lag_2[:2] = speed[0]  # Fill first 2 values

# Altitude lag 30 (HR responds ~30 timesteps after elevation change)
altitude_lag_30 = np.roll(altitude, 30)
altitude_lag_30[:30] = altitude[0]
```

### Derivative Features

**Rationale**: Acceleration/deceleration directly affects HR response.

```python
# Speed derivative (acceleration)
speed_derivative = np.diff(speed, prepend=speed[0])

# Altitude derivative (elevation change rate)
altitude_derivative = np.diff(altitude, prepend=altitude[0])
```

### Rolling Statistics

**Rationale**: Smooth noise, capture sustained effort.

```python
# Rolling mean (10 timesteps ≈ 1 minute)
rolling_speed_10 = pd.Series(speed).rolling(10, min_periods=1).mean()

# Rolling mean (30 timesteps ≈ 3 minutes)
rolling_speed_30 = pd.Series(speed).rolling(30, min_periods=1).mean()
```

### Cumulative Features

**Rationale**: Total elevation gain affects fatigue/HR.

```python
# Cumulative elevation gain (only positive changes)
elevation_gain = np.maximum(altitude_derivative, 0)
cumulative_elevation = np.cumsum(elevation_gain)
```

## Masking Strategy

**Problem**: 43% of sequences are padded (original length < 500).

**Solution**: Generate validity masks during padding.

```python
def pad_or_truncate_with_mask(sequence, target_length=500):
    current_length = len(sequence)
    mask = np.zeros(target_length, dtype=np.float32)
    
    if current_length >= target_length:
        # Truncate
        padded = sequence[:target_length]
        mask[:] = 1.0  # All valid
    else:
        # Pad with last value
        padding = np.full(target_length - current_length, sequence[-1])
        padded = np.concatenate([sequence, padding])
        mask[:current_length] = 1.0  # Only real data valid
    
    return padded, mask
```

**Usage in training**:
```python
# Loss ignores padded regions
loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum()
```

## Normalization

**Strategy**: Fit StandardScaler on training data only, apply to all splits.

```python
# Fit on training data
speed_scaler = StandardScaler().fit(train_speed)
altitude_scaler = StandardScaler().fit(train_altitude)

# Apply to all splits
train_speed_norm = speed_scaler.transform(train_speed)
val_speed_norm = speed_scaler.transform(val_speed)
test_speed_norm = speed_scaler.transform(test_speed)

# HR kept unnormalized (interpretable loss in BPM)
```

**Important**: Create lag features AFTER normalization.

```python
# ✅ CORRECT
speed_norm = scaler.transform(speed)
speed_lag_2_norm = np.roll(speed_norm, 2)

# ❌ WRONG (lag uses raw, feature uses normalized)
speed_lag_2_raw = np.roll(speed, 2)
speed_norm = scaler.transform(speed)
```

## User-Based Splitting (Stratified)

**Goal**: Prevent data leakage + balance fitness levels across splits.

**Strategy**:
1. Compute avg HR per user (fitness proxy)
2. Stratify by fitness quartile (Q1=fit, Q4=unfit)
3. Split users (not workouts) ensuring each split has mix

```python
def split_by_user_stratified(workouts):
    # Compute user fitness
    user_avg_hr = {}
    for w in workouts:
        if w['userId'] not in user_avg_hr:
            user_avg_hr[w['userId']] = []
        user_avg_hr[w['userId']].append(np.mean(w['heart_rate']))
    
    user_fitness = {uid: np.mean(hrs) for uid, hrs in user_avg_hr.items()}
    fitness_quartile = pd.qcut(list(user_fitness.values()), 4, labels=[0,1,2,3])
    
    # Stratified split
    train_val_users, test_users = train_test_split(
        users, test_size=0.15, stratify=fitness_quartile
    )
    ...
```

## Quality Filtering

**Thresholds** (from manual annotation):

| Check | Threshold | Action |
|-------|-----------|--------|
| Overall quality score | ≥ 70 | Use in training |
| HR spikes | < 3 spikes | Flag if more |
| HR flatlines | < 10 sec | Reject if longer |
| Impossible speeds | 0 occurrences | Hard fail |
| Altitude jumps | < 2 jumps | Flag if more |
| Sampling gaps | < 5 gaps >30s | Flag if more |

## Verification

After preprocessing, verify data quality:

```bash
# Check tensor shapes
python3 -c "
import torch
data = torch.load('DATA/processed/train.pt')
print('Features:', data['features'].shape)
print('HR:', data['heart_rate'].shape)
print('Mask:', data['mask'].shape)
"

# Expected output:
# Features: torch.Size([N, 500, 11])
# HR: torch.Size([N, 500, 1])
# Mask: torch.Size([N, 500, 1])
```

## Common Issues

### Issue 1: Features not normalized consistently

**Symptom**: Speed and speed_lag_2 have different scales.

**Fix**: Create lag features AFTER normalization.

### Issue 2: Mask not applied in loss

**Symptom**: Model learns to predict on padded regions.

**Fix**: Use MaskedMSELoss or apply mask manually.

### Issue 3: Data leakage in splits

**Symptom**: Same user in train and test.

**Fix**: Use user-based splitting (already implemented).

## Performance

Expected preprocessing time:

| Dataset Size | Time |
|--------------|------|
| 974 workouts | ~5 min |
| 10K workouts | ~30 min |
| 253K workouts | ~3 hours |

Memory usage: ~4 GB for 253K workouts.
