# DATA Directory

## Structure

```
DATA/
├── raw/
│   └── endomondoHR.json         # Raw Endomondo dataset (symlink to V1)
├── quality_check/
│   ├── annotations.csv          # Manual quality annotations (to be created)
│   └── quality_report.md        # Quality analysis summary
└── processed/
    ├── train.pt                 # Training tensors [N, 500, 11]
    ├── val.pt                   # Validation tensors
    ├── test.pt                  # Test tensors
    ├── metadata.json            # Dataset statistics
    └── scaler_params.json       # Normalization parameters
```

## Data Flow

```
Raw JSON (253K workouts)
    ↓
Quality Validation (manual + automated)
    ↓
Feature Engineering (3 → 11 features)
    ↓
Preprocessing (pad, mask, normalize, split)
    ↓
PyTorch Tensors (processed/)
```

## Raw Data Format

Each line in `endomondoHR.json`:

```json
{
  "id": "123456",
  "userId": "789",
  "sport": "run",
  "gender": "male",
  "speed": [10.2, 10.5, 10.3, ...],
  "altitude": [100, 102, 105, ...],
  "heart_rate": [145, 148, 150, ...],
  "timestamp": [1234567890, 1234567891, ...]
}
```

## Processed Data Format

Each `.pt` file contains a dictionary:

```python
{
    'features': torch.FloatTensor,    # [N, 500, 11]
    'heart_rate': torch.FloatTensor,  # [N, 500, 1]
    'mask': torch.FloatTensor,        # [N, 500, 1] (1=valid, 0=padded)
    'gender': torch.FloatTensor,      # [N, 1]
    'userId': torch.LongTensor,       # [N, 1]
    'original_lengths': torch.LongTensor  # [N, 1]
}
```

## Features (11 total)

**Base** (3):
1. `speed` - Running speed (km/h)
2. `altitude` - Elevation (meters)
3. `gender` - Binary (1=male, 0=female)

**Engineered** (8):
4. `speed_lag_2` - Speed 2 timesteps ago
5. `speed_lag_5` - Speed 5 timesteps ago
6. `altitude_lag_30` - Altitude 30 timesteps ago
7. `speed_derivative` - Acceleration
8. `altitude_derivative` - Elevation change rate
9. `rolling_speed_10` - Moving avg (10 timesteps)
10. `rolling_speed_30` - Moving avg (30 timesteps)
11. `cumulative_elevation` - Total elevation gain

## Usage

```python
import torch

# Load preprocessed data
train_data = torch.load('DATA/processed/train.pt')

features = train_data['features']      # [N, 500, 11]
heart_rate = train_data['heart_rate']  # [N, 500, 1]
mask = train_data['mask']              # [N, 500, 1]

print(f"Training samples: {features.shape[0]}")
print(f"Feature dimensions: {features.shape}")
```

## Statistics

Expected after quality filtering:

| Split | Workouts | Users | % Total |
|-------|----------|-------|---------|
| Train | ~680 | ~70% users | 70% |
| Val | ~145 | ~15% users | 15% |
| Test | ~145 | ~15% users | 15% |
| **Total** | **~970** | **~100% users** | **100%** |

Note: Exact numbers depend on quality filtering results.
