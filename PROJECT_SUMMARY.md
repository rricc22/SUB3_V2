# SUB3_V2 Project Summary

**Created**: 2025-01-10  
**Status**: Documentation Complete, Ready for Implementation

---

## What is SUB3_V2?

Version 2.0 of the heart rate prediction project, focused on **data quality** and **feature engineering** to achieve better model performance than V1.

### Key Improvements

| Aspect | V1 | V2 | Impact |
|--------|----|----|--------|
| **Data Validation** | None | Manual + Automated | Remove low-quality workouts |
| **Features** | 3 | 11 | Capture temporal dynamics |
| **Masking** | No | Yes | Ignore 43% padding pollution |
| **Splitting** | Random | Stratified | Balance fitness levels |
| **Target MAE** | 13.88 BPM | < 10 BPM | -28% improvement |

---

## Project Structure

```
SUB3_V2/
├── README.md                    # Project overview + baseline metrics
├── AGENTS.md                    # Coding guidelines
├── .gitignore                   # Git ignore rules
│
├── docs/                        # BMAD Documentation
│   ├── PRD.md                   # Product Requirements
│   ├── ARCHITECTURE.md          # Technical architecture
│   ├── DATA_QUALITY.md          # Quality criteria
│   ├── CHANGELOG.md             # Version history
│   └── QUICK_START.md           # Getting started guide
│
├── DATA/
│   ├── raw/                     # Raw Endomondo data
│   ├── quality_check/           # Manual annotations
│   ├── processed/               # PyTorch tensors (to be created)
│   └── README.md
│
├── EDA/
│   ├── outputs/                 # Visualizations
│   └── README.md                # (to be created)
│
├── Preprocessing/
│   ├── prepare_sequences.py     # (to be created)
│   ├── quality_filters.py       # (to be created)
│   ├── feature_engineering.py   # (to be created)
│   └── README.md
│
├── Model/
│   ├── lstm.py                  # (to be created)
│   ├── train.py                 # (to be created)
│   ├── loss.py                  # (to be created)
│   ├── evaluate.py              # (to be created)
│   └── README.md
│
├── checkpoints/                 # Model weights (created during training)
├── results/                     # Evaluation outputs
└── logs/                        # Training logs
```

---

## Documentation Completed ✓

### BMAD Method Documents

1. **PRD.md** (Product Requirements Document)
   - Business requirements
   - Success criteria (MAE < 10 BPM)
   - Use cases
   - Model requirements (11 features)
   - Data quality criteria
   - Performance requirements
   - Deliverables

2. **ARCHITECTURE.md** (Technical Architecture)
   - System overview
   - Data pipeline (quality validation → feature engineering → preprocessing)
   - Model architecture (LSTM with 11 features)
   - Loss function (MaskedMSELoss)
   - Training pipeline
   - Evaluation metrics
   - Testing strategy

3. **DATA_QUALITY.md** (Data Quality Criteria)
   - Quality dimensions (HR sensor, GPS, altitude, temporal, activity)
   - Validation functions
   - Manual annotation workflow
   - Quality thresholds
   - Implementation plan
   - Expected outcomes

4. **CHANGELOG.md** (Version History)
   - V2.0.0 features (data quality, feature engineering, masking)
   - V1.0.0 baseline results
   - Breaking changes (V1 → V2 migration)
   - Development roadmap

### Project Documentation

5. **README.md** (Project Overview)
   - Quick reference baseline metrics table
   - V2 improvements focus
   - Project structure
   - Quick start commands
   - Expected improvements
   - Development status

6. **AGENTS.md** (Coding Guidelines)
   - Project context
   - Quick commands
   - Code style (PEP 8)
   - V2 specific guidelines (feature engineering, masking, quality)
   - Project structure navigation
   - Model architecture details
   - Baseline metrics
   - Common pitfalls
   - Testing strategy
   - Debugging tips

7. **QUICK_START.md** (Getting Started Guide)
   - Prerequisites
   - Full workflow (Phases 0-3)
   - Code examples
   - Troubleshooting
   - Next steps

### Module Documentation

8. **DATA/README.md** - Data structure and usage
9. **Preprocessing/README.md** - Preprocessing pipeline details
10. **Model/README.md** - Model training and evaluation
11. **.gitignore** - Git ignore rules

---

## Next Steps (Implementation)

### Phase 0: Data Quality (Week 1)

**Goal**: Manually validate 100 workouts to establish quality baselines.

**Tasks**:
1. Create Streamlit annotation app (`EDA/quality_annotation_app.py`)
2. Annotate 100 diverse workouts
3. Analyze annotations and establish thresholds
4. Implement automated quality filters (`Preprocessing/quality_filters.py`)

**Deliverables**:
- `DATA/quality_check/annotations.csv`
- `DATA/quality_check/quality_report.md`
- `Preprocessing/quality_filters.py`

### Phase 1: Preprocessing (Week 2)

**Goal**: Transform raw data into PyTorch tensors with engineered features.

**Tasks**:
1. Implement feature engineering (`Preprocessing/feature_engineering.py`)
2. Implement main preprocessing pipeline (`Preprocessing/prepare_sequences.py`)
3. Verify preprocessed data

**Deliverables**:
- `Preprocessing/feature_engineering.py`
- `Preprocessing/prepare_sequences.py`
- `DATA/processed/train.pt`, `val.pt`, `test.pt`

### Phase 2: Model Training (Week 3)

**Goal**: Train LSTM with 11 features and masked loss.

**Tasks**:
1. Implement model (`Model/lstm.py`)
2. Implement masked loss (`Model/loss.py`)
3. Implement training loop (`Model/train.py`)
4. Train and validate model

**Deliverables**:
- `Model/lstm.py`
- `Model/loss.py`
- `Model/train.py`
- `checkpoints/best_model.pt`

### Phase 3: Evaluation (Week 4)

**Goal**: Evaluate V2 model and compare with V1 baseline.

**Tasks**:
1. Implement evaluation script (`Model/evaluate.py`)
2. Run test set evaluation
3. Compare V1 vs V2 results
4. Generate performance report

**Deliverables**:
- `Model/evaluate.py`
- `results/test_metrics.json`
- `results/v1_vs_v2_comparison.md`

---

## Key Design Decisions

### 1. Feature Engineering (11 features vs 3)

**Rationale**: Weak correlation (0.25) in V1 is partly due to missing temporal features.

**Features**:
- **Lag** (3): Capture physiological delay (HR responds 2 timesteps after speed change)
- **Derivatives** (2): Acceleration/deceleration directly affects HR
- **Rolling** (2): Smooth noise, capture sustained effort
- **Cumulative** (1): Total elevation gain affects fatigue

**Expected impact**: Increase correlation from 0.25 → 0.40

### 2. Masking (NEW in V2)

**Problem**: 43% of sequences are padded with repeated last value.

**Solution**: Generate validity masks during preprocessing, ignore padded regions in loss.

**Impact**: Prevents learning on artificial data, cleaner gradients.

### 3. Data Quality Validation (NEW in V2)

**Problem**: V1 had no sensor quality checks (HR spikes, GPS noise, etc.).

**Solution**: Manual annotation of 100 workouts → establish thresholds → automated filtering.

**Expected impact**: Remove ~20-30% low-quality workouts, cleaner training data.

### 4. Stratified User Splitting (NEW in V2)

**Problem**: V1 test set had different statistics (higher variance).

**Solution**: Stratify users by fitness level (avg HR proxy) to balance splits.

**Impact**: Reduce distribution mismatch, better generalization.

---

## Success Criteria

### Primary Metrics

| Metric | V1 Baseline | V2 Target | Status |
|--------|-------------|-----------|--------|
| **MAE (base)** | 13.88 BPM | < 10 BPM | Not yet tested |
| **R²** | 0.188 | > 0.35 | Not yet tested |
| **Correlation** | 0.25 | > 0.40 | Not yet tested |

### Secondary Metrics

- **Finetuned MAE**: < 7 BPM (vs 8.94 BPM in V1)
- **Training time**: < 30 min (vs ~20 min in V1)
- **Visual inspection**: Predictions follow trends

---

## Technology Stack

- **Language**: Python 3.11+
- **Deep Learning**: PyTorch 2.0+
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Annotation**: Streamlit
- **Version Control**: Git

---

## References

- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Dataset**: Endomondo HR from FitRec project
- **Baseline**: LSTM Baseline (13.88 BPM MAE)
- **Best V1**: Finetuned Stage 1 (8.94 BPM MAE on Apple Watch)

---

## Documentation Status

- [x] PRD (Product Requirements Document)
- [x] Architecture (Technical design)
- [x] Data Quality (Validation criteria)
- [x] Changelog (Version history)
- [x] README (Project overview)
- [x] AGENTS.md (Coding guidelines)
- [x] Quick Start Guide
- [x] Module READMEs (Data, Preprocessing, Model)
- [ ] Implementation (to be completed in Phases 0-3)
- [ ] Final Results Report (after Phase 3)

---

**Project Lead**: Riccardo  
**Technical Documentation**: OpenCode  
**Methodology**: BMAD (Business, Model, Architecture, Development)

**Status**: Ready for Phase 0 (Data Quality Validation)
