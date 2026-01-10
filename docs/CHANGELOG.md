# Changelog
## SUB3_V2 Heart Rate Prediction

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-01-10

### Added - Data Quality & Feature Engineering

#### Data Quality Validation (NEW)
- **Manual annotation workflow** for 50-100 workouts
- Quality validation functions:
  - `validate_hr_sensor()`: Detect spikes, flatlines, dropouts
  - `validate_gps_speed()`: Detect impossible speeds, GPS noise
  - `validate_altitude()`: Detect altitude jumps, barometric errors
  - `validate_timestamps()`: Check sampling regularity
  - `validate_activity_type()`: Verify workout labels
- Streamlit annotation app (`EDA/quality_annotation_app.py`)
- Quality scoring system (0-100 scale)
- Automated quality filtering in preprocessing

#### Feature Engineering (NEW)
- **8 new engineered features** (total: 11 vs 3 in V1):
  - Lag features: `speed_lag_2`, `speed_lag_5`, `altitude_lag_30`
  - Derivatives: `speed_derivative`, `altitude_derivative`
  - Rolling statistics: `rolling_speed_10`, `rolling_speed_30`
  - Cumulative: `cumulative_elevation_gain`
- Feature engineering module (`Preprocessing/feature_engineering.py`)
- Lag-based on EDA temporal correlation analysis (speed lags by 2 timesteps)

#### Masked Loss Function (NEW)
- `MaskedMSELoss` class to ignore padded regions
- Mask generation during preprocessing
- Mask validation in training loop
- Prevents learning on 43% of artificial padding data

#### Documentation (BMAD Method)
- Product Requirements Document (`docs/PRD.md`)
- Technical Architecture (`docs/ARCHITECTURE.md`)
- Data Quality Criteria (`docs/DATA_QUALITY.md`)
- Agent Guidelines (`AGENTS.md`)
- Project README with baseline metrics table

### Changed - Preprocessing & Model

#### Preprocessing Pipeline
- **Padding with masking**: Generate validity masks (1=valid, 0=padded)
- **Stratified user splitting**: Balance fitness levels across splits
- **Feature normalization**: Consistent scaling for base + lag features
- Output tensor format: `features` [N, 500, 11] instead of separate arrays
- Added `mask` tensor: [N, 500, 1] (NEW)

#### Model Architecture
- **Input size**: 3 → 11 features
- **Hidden size**: 64 → 128 units (handle more features)
- **Dropout**: 0.2 → 0.3 (prevent overfitting)
- **Parameters**: ~50K → ~180K
- Simplified forward pass (features pre-concatenated)

#### Training Configuration
- **Learning rate**: 0.001 → 0.0005 (more parameters)
- **Use masking**: False → True (ignore padded regions)
- **Batch size**: 16 (kept same, optimal from V1)

### Improved - Performance Targets

| Metric | V1 Baseline | V2 Target |
|--------|-------------|-----------|
| MAE (base) | 13.88 BPM | < 10 BPM |
| MAE (finetuned) | 8.94 BPM | < 7 BPM |
| R² | 0.188 | > 0.35 |
| Speed→HR correlation | 0.25 | > 0.40 |

### Fixed - Known Issues from V1

- **Weak correlation** → Lag features capture physiological delay
- **Padding pollution** → Masking ignores artificial data
- **Distribution mismatch** → Stratified splitting balances splits
- **No temporal features** → Derivatives and rolling stats added
- **Loss on padding** → MaskedMSELoss excludes padded timesteps

---

## [1.0.0] - 2024-12-17 (V1 Baseline)

### Summary - Best Results

| Model | MAE (BPM) | Status |
|-------|-----------|--------|
| **Finetuned Stage 1** | **8.94** | Best ⭐ |
| Finetuned Stage 2 | 10.15 | Excellent |
| LSTM Baseline | 13.88 | Good |
| GRU | 14.23 | Good |
| LSTM + Embeddings | 15.79 | OK |

### Implemented
- Basic preprocessing (`prepare_sequences_v2.py`)
  - Pad/truncate to 500 timesteps
  - Normalize speed/altitude (Z-score)
  - User-based splitting (70/15/15)
- Model architectures:
  - LSTM (2 layers, 64 units)
  - GRU (4 layers, bidirectional)
  - LSTM with user embeddings
  - Lag-Llama (transformer)
  - PatchTST (transformer)
- Finetuning pipeline:
  - Stage 1: Freeze layer 0, train layer 1 + FC
  - Stage 2: Unfreeze all layers
  - Apple Watch personal data (196 samples)
- Evaluation metrics: MAE, RMSE, R²

### Issues Identified
1. Weak feature-target correlation (0.25)
2. High padding ratio (43% of sequences)
3. Distribution mismatch in test set
4. No temporal features (lag, derivatives)
5. Loss computed on padded regions
6. No data quality validation

---

## Development Roadmap

### Phase 0: Data Quality (Week 1)
- [ ] Manual annotation of 100 workouts
- [ ] Establish quality thresholds
- [ ] Implement automated quality filters
- [ ] Generate quality report

### Phase 1: Preprocessing (Week 2)
- [ ] Implement feature engineering
- [ ] Generate masked tensors
- [ ] Validate feature correlations
- [ ] Document preprocessing pipeline

### Phase 2: Model Training (Week 3)
- [ ] Train LSTM with 11 features + masking
- [ ] Hyperparameter search
- [ ] Ablation study (feature importance)
- [ ] Compare V1 vs V2 baselines

### Phase 3: Evaluation (Week 4)
- [ ] Test set evaluation
- [ ] Cross-user validation
- [ ] Visual inspection (sample predictions)
- [ ] Final performance report

### Future Enhancements (V3)
- Multi-step ahead prediction (forecast HR 10 steps ahead)
- Uncertainty quantification (confidence intervals)
- Attention mechanisms (learn which features matter when)
- Multi-sport support (cycling, swimming)
- Real-time streaming inference
- Mobile deployment (ONNX export)

---

## Breaking Changes

### V1 → V2 Migration

**Data format**:
```python
# V1
{
    'speed': [N, 500, 1],
    'altitude': [N, 500, 1],
    'gender': [N, 1],
    'heart_rate': [N, 500, 1]
}

# V2
{
    'features': [N, 500, 11],  # All features concatenated
    'mask': [N, 500, 1],       # NEW: Validity mask
    'heart_rate': [N, 500, 1],
    'gender': [N, 1],
    'userId': [N, 1]
}
```

**Model input**:
```python
# V1
model.forward(speed, altitude, gender, lengths)

# V2
model.forward(features)  # features = concatenated [N, 500, 11]
```

**Loss function**:
```python
# V1
loss = criterion(predictions, targets)

# V2
loss = criterion(predictions, targets, mask)  # Masking required
```

---

## Acknowledgments

- **Dataset**: Endomondo HR from FitRec project
- **V1 Development**: Riccardo + OpenCode
- **V2 Design**: Based on comprehensive EDA findings

---

**Version**: 2.0.0  
**Status**: Planning / Data Quality Phase  
**Last Updated**: 2025-01-10
