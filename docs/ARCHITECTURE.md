# Technical Architecture
## SUB3_V2 Heart Rate Prediction System

**Version**: 2.0  
**Last Updated**: 2025-01-10

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          RAW DATA                                │
│                   Endomondo HR JSON (253K)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  QUALITY VALIDATION  │  ← Manual annotation
          │  - HR sensor check   │
          │  - GPS accuracy      │
          │  - Sampling gaps     │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  FEATURE ENGINEERING │
          │  - Lag features      │
          │  - Derivatives       │
          │  - Rolling stats     │
          │  - Cumulative        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   PREPROCESSING      │
          │  - Pad/truncate      │
          │  - Generate mask     │
          │  - Normalize         │
          │  - User split        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   PyTorch Tensors    │
          │  [N, 500, 11]        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │    LSTM MODEL        │
          │  - 2 layers, 128d    │
          │  - Dropout 0.3       │
          │  - Masked loss       │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │    PREDICTIONS       │
          │  HR [N, 500, 1]      │
          └──────────────────────┘
```

---

## Data Pipeline Architecture

### 1. Input Layer

**Source**: `DATA/raw/endomondoHR.json`

**Format**: JSON Lines (one workout per line)
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

### 2. Quality Validation Layer (NEW in V2)

**Module**: `Preprocessing/quality_filters.py`

**Checks**:
```python
class QualityValidator:
    def validate_hr_sensor(self, hr_array):
        """Detect HR sensor malfunctions"""
        # Check 1: No sudden spikes (>30 BPM in 1 step)
        hr_diff = np.abs(np.diff(hr_array))
        if np.any(hr_diff > 30):
            return False, "HR spike detected"
        
        # Check 2: No prolonged flatlines (>10 identical values)
        flatline = (hr_diff == 0)
        max_flatline = np.max(np.convolve(flatline, np.ones(10)))
        if max_flatline >= 10:
            return False, "HR flatline detected"
        
        return True, "OK"
    
    def validate_gps(self, speed_array):
        """Detect GPS errors"""
        # No sustained speeds >25 km/h for running
        if np.any(speed_array > 25):
            return False, "Impossible speed"
        return True, "OK"
```

### 3. Feature Engineering Layer (NEW in V2)

**Module**: `Preprocessing/feature_engineering.py`

**Input**: Raw sequences (speed, altitude)  
**Output**: 11-feature tensor

```python
class FeatureEngineer:
    def __init__(self):
        self.features = [
            'speed', 'altitude', 'gender',              # Base (3)
            'speed_lag_2', 'speed_lag_5',               # Lag (2)
            'altitude_lag_30',                          # Lag (1)
            'speed_derivative', 'altitude_derivative',  # Derivatives (2)
            'rolling_speed_10', 'rolling_speed_30',     # Rolling (2)
            'cumulative_elevation'                      # Cumulative (1)
        ]
    
    def engineer_features(self, speed, altitude):
        """Transform raw -> engineered features"""
        features = {}
        
        # Lag features (physiological delay)
        features['speed_lag_2'] = np.roll(speed, 2)
        features['speed_lag_2'][:2] = speed[0]
        
        features['speed_lag_5'] = np.roll(speed, 5)
        features['speed_lag_5'][:5] = speed[0]
        
        features['altitude_lag_30'] = np.roll(altitude, 30)
        features['altitude_lag_30'][:30] = altitude[0]
        
        # Derivatives (rate of change)
        features['speed_derivative'] = np.diff(speed, prepend=speed[0])
        features['altitude_derivative'] = np.diff(altitude, prepend=altitude[0])
        
        # Rolling statistics (smoothing)
        features['rolling_speed_10'] = (
            pd.Series(speed).rolling(10, min_periods=1).mean().values
        )
        features['rolling_speed_30'] = (
            pd.Series(speed).rolling(30, min_periods=1).mean().values
        )
        
        # Cumulative features
        elevation_gain = np.maximum(features['altitude_derivative'], 0)
        features['cumulative_elevation'] = np.cumsum(elevation_gain)
        
        return features
```

### 4. Preprocessing Layer

**Module**: `Preprocessing/prepare_sequences.py`

**Key Operations**:

1. **Padding with Masking**
```python
def pad_or_truncate_with_mask(sequence, target_length=500):
    """
    Pad/truncate to 500 timesteps + generate validity mask.
    
    Returns:
        padded: [500] array
        mask: [500] binary (1=valid, 0=padded)
    """
    current_length = len(sequence)
    mask = np.zeros(target_length, dtype=np.float32)
    
    if current_length >= target_length:
        padded = sequence[:target_length]
        mask[:] = 1.0  # All valid
    else:
        # Pad with last value (mask will ignore it in loss)
        padding = np.full(target_length - current_length, sequence[-1])
        padded = np.concatenate([sequence, padding])
        mask[:current_length] = 1.0  # Only real data valid
    
    return padded, mask
```

2. **Stratified User Splitting**
```python
def split_by_user_stratified(workouts):
    """
    Split users by fitness level to balance distributions.
    
    Strategy:
    - Compute avg HR per user (fitness proxy)
    - Stratify by quartile (Q1=fit, Q4=unfit)
    - Ensure each split has mix of fitness levels
    """
    user_avg_hr = compute_user_avg_hr(workouts)
    fitness_quartile = pd.qcut(user_avg_hr, 4, labels=[0,1,2,3])
    
    train_val_users, test_users = train_test_split(
        users, 
        test_size=0.15, 
        stratify=fitness_quartile,
        random_state=42
    )
    return train_val_users, test_users
```

3. **Normalization**
```python
# Fit on training data only
speed_scaler = StandardScaler().fit(train_speed)
altitude_scaler = StandardScaler().fit(train_altitude)

# Apply to all splits
train_speed_norm = speed_scaler.transform(train_speed)
val_speed_norm = speed_scaler.transform(val_speed)

# HR kept unnormalized for interpretable loss (BPM)
```

**Output Tensor Format**:
```python
{
    'features': torch.FloatTensor,    # [N, 500, 11]
    'heart_rate': torch.FloatTensor,  # [N, 500, 1]
    'mask': torch.FloatTensor,        # [N, 500, 1]  ← NEW
    'gender': torch.FloatTensor,      # [N, 1]
    'userId': torch.LongTensor,       # [N, 1]
    'original_lengths': torch.LongTensor  # [N, 1]
}
```

---

## Model Architecture

### LSTM with Masking (V2)

```python
class HeartRateLSTM_v2(nn.Module):
    """
    LSTM model for HR prediction with engineered features.
    
    Changes from V1:
    - Input size: 3 → 11 (engineered features)
    - Hidden size: 64 → 128 (handle more features)
    - Dropout: 0.2 → 0.3 (prevent overfitting)
    - Forward pass: Simplified (features pre-concatenated)
    """
    
    def __init__(
        self,
        input_size=11,        # 11 features (was 3)
        hidden_size=128,      # Increased from 64
        num_layers=2,
        dropout=0.3,          # Increased from 0.2
        bidirectional=False
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, 1)
    
    def forward(self, features):
        """
        Args:
            features: [batch, seq_len, 11] - All features concatenated
        
        Returns:
            predictions: [batch, seq_len, 1] - HR predictions
        """
        # LSTM forward
        lstm_out, _ = self.lstm(features)  # [batch, seq_len, hidden]
        
        # Dropout + FC
        out = self.dropout(lstm_out)
        predictions = self.fc(out)  # [batch, seq_len, 1]
        
        return predictions
```

**Parameters**: ~180K (vs ~50K in V1, due to 11 inputs)

---

## Loss Function

### Masked MSE Loss (NEW in V2)

```python
class MaskedMSELoss(nn.Module):
    """
    MSE loss that ignores padded regions.
    
    Critical for V2: 43% of sequences are padded.
    Without masking, model learns to predict on artificial data.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: [batch, seq_len, 1]
            targets: [batch, seq_len, 1]
            mask: [batch, seq_len, 1] (1=valid, 0=padded)
        
        Returns:
            loss: Scalar (averaged over valid timesteps only)
        """
        # Compute MSE per timestep
        loss = self.mse(predictions, targets)  # [batch, seq_len, 1]
        
        # Zero out padded regions
        masked_loss = loss * mask
        
        # Average over valid timesteps only
        return masked_loss.sum() / mask.sum()
```

**Impact**: Focuses learning on real data, ignores 43% padding pollution.

---

## Training Pipeline

### Configuration

```python
TRAINING_CONFIG = {
    # Model
    'input_size': 11,        # 11 features
    'hidden_size': 128,      # Increased from 64
    'num_layers': 2,
    'dropout': 0.3,          # Increased from 0.2
    'bidirectional': False,
    
    # Training
    'batch_size': 16,        # Optimal from V1 experiments
    'learning_rate': 0.0005, # Halved due to more params
    'epochs': 100,
    'patience': 10,          # Early stopping
    'weight_decay': 1e-4,    # L2 regularization
    'gradient_clip': 1.0,    # Prevent exploding gradients
    
    # Loss
    'use_masking': True,     # NEW: Ignore padded regions
    
    # Optimizer
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
}
```

### Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_mae = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)      # [B, 500, 11]
        heart_rate = batch['heart_rate'].to(device)  # [B, 500, 1]
        mask = batch['mask'].to(device)              # [B, 500, 1]
        
        # Forward pass
        predictions = model(features)
        
        # Masked loss
        loss = criterion(predictions, heart_rate, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics (also masked)
        mae = torch.abs(predictions - heart_rate) * mask
        mae = mae.sum() / mask.sum()
        
        total_loss += loss.item()
        total_mae += mae.item()
    
    return total_loss / len(dataloader), total_mae / len(dataloader)
```

---

## Evaluation Metrics

### Primary Metrics

1. **MAE (Mean Absolute Error)**
```python
mae = |predicted - actual|.mean()  # In BPM
```
Target: < 10 BPM

2. **RMSE (Root Mean Squared Error)**
```python
rmse = sqrt(((predicted - actual) ** 2).mean())
```
Penalizes large errors

3. **R² (Coefficient of Determination)**
```python
r2 = 1 - (SS_residual / SS_total)
```
Target: > 0.35

### Secondary Metrics

4. **Per-timestep MAE** (visualize prediction quality over workout)
5. **Per-user MAE** (detect generalization issues)
6. **Zone accuracy** (% timesteps in correct HR zone)

---

## Deployment Architecture (Future)

```
┌──────────────┐
│  Mobile App  │
└──────┬───────┘
       │ HTTPS
       ▼
┌──────────────┐
│   API Server │  Flask/FastAPI
│   (GPU)      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ PyTorch Model│  LSTM checkpoint
│  (ONNX)      │  Optimized for inference
└──────────────┘
```

**Inference**:
- Input: Speed/altitude array (50-500 timesteps)
- Processing: Feature engineering → normalization → model forward
- Output: HR predictions + confidence
- Latency: < 100ms

---

## Performance Optimizations

### Current (Training)
- DataLoader workers: 4
- Pin memory: True
- GPU utilization: ~80%
- Training time: ~20 min (100 epochs, 974 samples)

### Future (Inference)
- ONNX export for 2x speedup
- Quantization (FP32 → FP16) for mobile
- Batch inference for historical analysis

---

## Testing Strategy

### Unit Tests
- Feature engineering (lag, derivatives, rolling)
- Padding + masking logic
- Loss computation correctness

### Integration Tests
- End-to-end preprocessing pipeline
- Training loop convergence
- Checkpoint save/load

### Validation Tests
- Ablation study (remove features one-by-one)
- Cross-user validation (leave-one-out)
- Synthetic data (known HR patterns)

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| Deep Learning | PyTorch | 2.0+ |
| Data Processing | NumPy, Pandas | Latest |
| Visualization | Matplotlib, Plotly | Latest |
| Experiment Tracking | TensorBoard | Latest |
| Version Control | Git | - |

---

## File Organization

```
Preprocessing/
├── prepare_sequences.py      # Main pipeline
├── quality_filters.py         # Data validation
├── feature_engineering.py     # Feature creation
└── README.md

Model/
├── lstm.py                    # Model definition
├── train.py                   # Training loop
├── loss.py                    # MaskedMSELoss
├── evaluate.py                # Metrics computation
└── README.md
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-10
