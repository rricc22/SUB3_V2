# Product Requirements Document (PRD)
## Heart Rate Prediction - Version 2.0

**Project**: SUB3_V2  
**Date**: 2025-01-10  
**Status**: Planning / Data Quality Phase

---

## Business Requirements

### Problem Statement

Athletes training for endurance events (marathons, ultra-marathons) need to understand their physiological response to exercise intensity to optimize training and avoid overtraining. Current wearables provide real-time HR monitoring, but prediction models can:

1. **Validate sensor accuracy** - Detect HR monitor malfunctions
2. **Predict HR in zones** - Plan workouts to stay in target zones
3. **Personalize training** - Understand individual HR response patterns
4. **Energy management** - Predict HR at race pace for pacing strategy

### Success Criteria

| Metric | V1 Baseline | V2 Target | Rationale |
|--------|-------------|-----------|-----------|
| **MAE (base model)** | 13.88 BPM | < 10 BPM | Clinical accuracy threshold |
| **MAE (finetuned)** | 8.94 BPM | < 7 BPM | Real-time application viable |
| **R²** | 0.188 | > 0.35 | Explain >35% of variance |
| **Feature Correlation** | 0.25 | > 0.40 | Stronger predictive signal |

### Target Users

1. **Primary**: Sub-3 marathon runners (experienced athletes)
2. **Secondary**: Endurance coaches analyzing athlete data
3. **Future**: General fitness enthusiasts with wearables

### Use Cases

#### Use Case 1: Pre-race Pacing Strategy
**Actor**: Marathon runner  
**Goal**: Predict HR at planned race pace  
**Flow**:
1. Input race pace (speed) and course profile (altitude)
2. Model predicts expected HR throughout race
3. Runner adjusts pace plan to stay in aerobic zone (<165 BPM)

**Success**: Predicted HR within 5 BPM of actual HR during race

#### Use Case 2: Training Zone Adherence
**Actor**: Coach  
**Goal**: Design interval workout that keeps athlete in Zone 2  
**Flow**:
1. Input athlete profile and workout intervals
2. Model predicts HR response to each interval
3. Adjust interval intensity to maintain zone compliance

**Success**: >90% of workout time in target zone

---

## Model Requirements

### Input Features (V2)

**Base Features** (3):
- `speed`: Running speed (km/h)
- `altitude`: Elevation (meters)
- `gender`: Binary (male/female)

**Engineered Features** (8):
- `speed_lag_2`: Speed 2 timesteps ago (physiological delay)
- `speed_lag_5`: Speed 5 timesteps ago (sustained effort)
- `altitude_lag_30`: Altitude 30 timesteps ago (elevation HR lag)
- `speed_derivative`: Acceleration/deceleration
- `altitude_derivative`: Elevation change rate
- `rolling_speed_10`: Moving average (10 timesteps)
- `rolling_speed_30`: Moving average (30 timesteps)
- `cumulative_elevation`: Total elevation gain

**Total**: 11 features (vs 3 in V1)

### Output

- **Heart rate**: Time-series prediction (BPM) for each timestep
- **Confidence intervals**: Optional (future enhancement)

### Model Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| Max inference time | < 100ms | Real-time application |
| Model size | < 10 MB | Mobile deployment |
| Min sequence length | 50 timesteps | ~5 minutes of data |
| Max sequence length | 500 timesteps | ~50 minutes (marathon segments) |

---

## Data Requirements

### Dataset: Endomondo HR

- **Source**: FitRec project (Endomondo fitness tracker)
- **Size**: 253,020 total workouts
- **After filtering**: 974 valid running workouts
- **Features**: GPS (speed, altitude), HR, timestamps, user metadata

### Data Quality Criteria (NEW in V2)

Before preprocessing, workouts must pass:

#### Automated Filters (from V1)
1. Sport type = "run"
2. Complete sequences (speed, altitude, HR, gender, userId)
3. Minimum length ≥ 50 timesteps
4. Valid HR range: 50-220 BPM
5. No NaN/Inf values

#### Manual Quality Checks (NEW)
6. **HR sensor quality**: No sudden spikes (>30 BPM change in 1 timestep)
7. **GPS accuracy**: No impossible speeds (>25 km/h sustained)
8. **Altitude consistency**: No drift (>100m sudden jump)
9. **Sampling regularity**: Timesteps <30s apart
10. **Workout validity**: Actual running (not walking/biking mislabeled)

**Target**: Annotate 50-100 workouts to establish quality baselines

### Data Splits

**Strategy**: User-based split (prevent data leakage)

| Split | % Users | % Workouts | Purpose |
|-------|---------|------------|---------|
| Train | 70% | ~680 | Model training |
| Val | 15% | ~145 | Hyperparameter tuning |
| Test | 15% | ~145 | Final evaluation |

**NEW in V2**: Stratified by user fitness level (avg HR proxy)

---

## Performance Requirements

### Accuracy Targets

| Level | MAE Range | Description | Use Case |
|-------|-----------|-------------|----------|
| **Excellent** | < 5 BPM | Medical-grade accuracy | Clinical applications |
| **Very Good** | 5-10 BPM | Real-time viable | Training guidance |
| **Good** | 10-15 BPM | Informative | Trend analysis |
| **Acceptable** | 15-20 BPM | Basic utility | Zone detection |
| **Poor** | > 20 BPM | Not useful | - |

**V2 Target**: Very Good (< 10 BPM) on base model

### Computational Performance

- **Training time**: < 30 minutes on GPU (100 epochs)
- **Inference time**: < 100ms per workout (500 timesteps)
- **Memory usage**: < 2 GB RAM during training

---

## Technical Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Weak correlation persists | High | High | Accept 10-12 BPM MAE as realistic ceiling |
| Overfitting on small dataset | Medium | Medium | Strong regularization, dropout 0.3 |
| New features increase noise | Medium | Medium | Ablation study to validate each feature |
| Padding mask implementation bugs | Low | High | Unit tests, visual validation |
| User stratification fails | Low | Medium | Fall back to random user split |

---

## Success Metrics

### Primary KPIs

1. **MAE < 10 BPM** on Endomondo test set (base model, no finetuning)
2. **R² > 0.35** (explain meaningful variance)
3. **Training stability** (no loss divergence, converges <50 epochs)

### Secondary KPIs

4. **Finetuned MAE < 7 BPM** (personal Apple Watch data)
5. **Feature correlation** (speed→HR > 0.40)
6. **Reduced test distribution mismatch** (MAE gap <2 BPM)

### Validation Criteria

- **Visual inspection**: Predictions follow trends on sample workouts
- **Physiological plausibility**: HR increases with speed, lags appropriately
- **Ablation analysis**: Each feature contributes positively
- **Cross-user generalization**: Variance in MAE <5 BPM across users

---

## Deliverables

### Phase 0: Data Quality (Week 1)
- [ ] Manual annotation of 50-100 workouts
- [ ] Quality criteria documentation
- [ ] Outlier detection rules

### Phase 1: Preprocessing (Week 2)
- [ ] `prepare_sequences.py` with feature engineering
- [ ] Preprocessed tensors (train/val/test)
- [ ] Feature correlation report

### Phase 2: Model Development (Week 3)
- [ ] LSTM model with 11-feature input
- [ ] Masked loss implementation
- [ ] Training pipeline with early stopping

### Phase 3: Evaluation (Week 4)
- [ ] Baseline comparison (V1 vs V2)
- [ ] Ablation study results
- [ ] Final model checkpoint
- [ ] Performance report

---

## Out of Scope (Future Work)

- Multi-step ahead prediction (predict HR 10 timesteps in future)
- Uncertainty quantification (confidence intervals)
- Multi-sport support (cycling, swimming)
- Real-time streaming inference
- Mobile app deployment

---

## Approval

**Product Owner**: Riccardo  
**Technical Lead**: OpenCode Agent  
**Status**: Approved for Data Quality Phase

**Next Step**: Manual data quality validation (50-100 workouts)

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-10
