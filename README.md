# SUB3_V2: Heart Rate Prediction from Running Data

**Version**: 2.0  
**Goal**: Predict heart rate time-series from speed and altitude during running workouts  
**Target**: Sub-3-hour marathon training optimization through personalized HR prediction

---

## Quick Reference: Baseline Metrics (V1)

| Model | Architecture | Parameters | Test MAE | Status |
|-------|-------------|------------|----------|--------|
| **Finetuned Stage 1** | LSTM + Transfer Learning | ~50K | **8.94 BPM** | **Best** ⭐ |
| Finetuned Stage 2 | LSTM + Full Fine-tuning | ~50K | 10.15 BPM | Excellent |
| LSTM Baseline | 2-layer LSTM (64 units) | ~50K | 13.88 BPM | Good |
| GRU | 4-layer Bidirectional GRU | ~40K | 14.23 BPM | Good |
| LSTM + Embeddings | LSTM with user/gender embeddings | ~65K | 15.79 BPM | OK |
| Lag-Llama | Transfer learning from pretrained | ~2M | 38.08 BPM | Poor |
| PatchTST | Patch-based Transformer | ~2M | N/A | Failed |

**V2 Target**: MAE < 10 BPM on base model (without personal finetuning)

---

## Why Version 2.0?

### Critical Problems in V1

| # | Problem | Impact | V1 Metric |
|---|---------|--------|-----------|
| 1 | **Weak feature-target correlation** | Speed→HR: only 0.21-0.25 | Limits model accuracy ceiling |
| 2 | **High padding ratio** | 43% of sequences artificially padded | Model learns noise |
| 3 | **Distribution mismatch** | Test set has different statistics | Poor generalization |
| 4 | **No temporal features** | HR responds to speed with 2-timestep lag | Missing physiological dynamics |
| 5 | **Loss on padded regions** | Gradients diluted by padding | Slower convergence |

### V2 Solutions

1. **Data Quality First** → Manual validation of raw data before preprocessing
2. **Feature Engineering** → Lag features, derivatives, rolling statistics
3. **Masking** → Ignore padded regions in loss computation
4. **Stratified Splitting** → Balance user distributions across splits
5. **Enhanced EDA** → Deep temporal analysis before implementation

---

## Project Structure

```
SUB3_V2/
├── README.md                    # This file
├── AGENTS.md                    # Agent guidelines for coding
│
├── docs/                        # BMAD Documentation
│   ├── PRD.md                   # Product Requirements Document
│   ├── ARCHITECTURE.md          # Technical architecture
│   ├── DATA_QUALITY.md          # Data quality criteria & validation
│   └── CHANGELOG.md             # Version history
│
├── DATA/
│   ├── raw/                     # Raw Endomondo data (symlink or copy)
│   ├── quality_check/           # Manual annotations, quality reports
│   │   ├── annotations.csv      # Manual quality labels
│   │   └── quality_report.md    # Summary of findings
│   └── processed/               # Final preprocessed PyTorch tensors
│       ├── train.pt
│       ├── val.pt
│       ├── test.pt
│       └── metadata.json
│
├── EDA/                         # Exploratory Data Analysis
│   ├── data_quality_check.ipynb # Interactive quality inspection
│   ├── temporal_analysis.py     # Lag correlation study
│   ├── feature_engineering_test.py  # Validate new features
│   └── outputs/                 # Plots and reports
│
├── Preprocessing/
│   ├── prepare_sequences.py     # Main preprocessing pipeline
│   ├── quality_filters.py       # Outlier detection, data cleaning
│   ├── feature_engineering.py   # Lag, derivatives, rolling stats
│   └── README.md                # Preprocessing documentation
│
├── Model/
│   ├── lstm.py                  # LSTM model (11 features)
│   ├── train.py                 # Training loop with masking
│   ├── loss.py                  # MaskedMSELoss
│   ├── evaluate.py              # Evaluation metrics
│   └── README.md                # Model documentation
│
├── checkpoints/                 # Saved model weights
├── results/                     # Evaluation outputs, plots
└── logs/                        # Training logs
```

---

## Quick Start

### 1. Data Quality Validation (Manual)

```bash
# Interactive quality check
jupyter notebook EDA/data_quality_check.ipynb

# Annotate 50-100 workouts for quality issues
# Export to DATA/quality_check/annotations.csv
```

### 2. Preprocessing with Feature Engineering

```bash
python3 Preprocessing/prepare_sequences.py
```

**New features** (11 total vs 3 in V1):
- Base: `speed`, `altitude`, `gender`
- Lag: `speed_lag_2`, `speed_lag_5`, `altitude_lag_30`
- Derivatives: `speed_derivative`, `altitude_derivative`
- Rolling: `rolling_speed_10`, `rolling_speed_30`
- Cumulative: `cumulative_elevation_gain`

### 3. Train Model

```bash
python3 Model/train.py --model lstm --epochs 100 --batch_size 16 --use_masking
```

### 4. Evaluate

```bash
python3 Model/evaluate.py --checkpoint checkpoints/best_model.pt
```

---

## Expected Improvements

| Metric | V1 Baseline | V2 Target | Improvement |
|--------|-------------|-----------|-------------|
| MAE (base model) | 13.88 BPM | < 10 BPM | -28% |
| R² | 0.188 | > 0.35 | +86% |
| Speed→HR correlation | 0.25 | > 0.40 | +60% |

---

## Development Status

- [x] Project structure created
- [x] BMAD documentation written
- [ ] Data quality validation completed
- [ ] Feature engineering implemented
- [ ] Preprocessing pipeline tested
- [ ] Model training with masking
- [ ] Baseline comparison
- [ ] V2 results published

---

## References

- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Dataset**: Endomondo HR dataset (253K workouts, filtered to 974 running)
- **Best V1 Model**: Finetuned LSTM Stage 1 (8.94 BPM MAE on Apple Watch data)

---

**Last Updated**: 2025-01-10
