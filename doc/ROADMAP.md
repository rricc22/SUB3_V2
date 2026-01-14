# SUB3_V2 Roadmap

**Last Updated**: 2026-01-14
**Current Status**: ✅ All phases complete

---

## Progress Overview

```
█████████████████████████████████ 100% COMPLETE ✅

✅ Phase 0: Data Exploration (Jan 11-13)
✅ Phase 1: Data Preprocessing (Jan 11-13)
✅ Phase 2: Tensor Preparation (Jan 13)
✅ Phase 3: Model Training (Jan 14)
✅ Phase 4: Evaluation & Viz (Jan 14)
```

---

## Timeline Summary

| Phase | Duration | Status | Results |
|-------|----------|--------|---------|
| **Phase 0: Data Exploration** | 3 days | ✅ Complete | EDA tools, gallery, indices |
| **Phase 1: Preprocessing** | 2 days | ✅ Complete | 3-stage pipeline, clean data (2.3GB) |
| **Phase 2: Tensor Prep** | 1 day | ✅ Complete | 11 features, PyTorch tensors (1.6GB) |
| **Phase 3: Training** | 1 day | ✅ Complete | Model trained, 11.90 BPM MAE |
| **Phase 4: Evaluation** | 1 day | ✅ Complete | 5 animations, anomaly analysis |

**Total Time**: ~8 days (Jan 10-14, 2026)

---

## What Was Accomplished

### Phase 0: Data Exploration ✅
- Line-based indexing for 61K+ workouts
- Interactive Streamlit app
- HTML gallery generator (46K thumbnails)
- Quality analysis tools

### Phase 1: Preprocessing ✅
- **3-stage hybrid pipeline**:
  - Stage 1: Rule-based filtering
  - Stage 2: LLM validation (HR offset detection)
  - Stage 3: Apply corrections
- GPS speed computation
- Moving average smoothing
- Output: 2.3GB clean dataset

### Phase 2: Tensor Preparation ✅
- **11 engineered features**:
  - Base (3): speed, altitude, gender
  - Lag (3): speed_lag_2, speed_lag_5, altitude_lag_30
  - Derivatives (2): speed_derivative, altitude_derivative
  - Rolling (2): rolling_speed_10, rolling_speed_30
  - Cumulative (1): cumulative_elevation_gain
- Padding/truncation with masking
- Stratified user splitting (70/15/15)
- Output: train.pt, val.pt, test.pt (1.6GB total)

### Phase 3: Model Training ✅
- LSTM architecture (11 inputs, 128 hidden, 2 layers, dropout 0.4)
- Masked MSE loss (ignores padding)
- W&B experiment tracking
- Training config: batch_size=64, lr=0.0005, patience=3
- Early stopping at epoch 6 (val MAE: 11.19 BPM)

### Phase 4: Evaluation & Visualization ✅
- **Test Results**: 11.90 BPM MAE, 14.42 BPM RMSE
- **5 Animations**:
  1. Gradual reveal (best: 2.6 BPM)
  2. Gradual reveal (worst: 42 BPM - offset error)
  3. Multi-workout comparison (4 workouts)
  4. Feature influence (speed/altitude → HR)
  5. Error heatmap (time evolution)
- Anomaly investigation (workout #4600 analysis)

---

## Performance Summary

| Metric | V1 Baseline | V2 Result | Target | Gap |
|--------|-------------|-----------|---------|-----|
| **Test MAE** | 13.88 BPM | **11.90 BPM** | < 10 BPM | -1.9 BPM |
| **Improvement** | - | **-14%** | -28% | Need -14% more |

---

## Key Findings

### Success Factors ✅
1. Feature engineering (11 features) improved modeling
2. Masked loss correctly handles 43% padding
3. 3-stage preprocessing caught HR offset errors
4. Early stopping prevented overfitting

### Challenges ⚠️
1. **Regression to mean**: Model clusters around 145-160 BPM
2. **Some offset errors remain**: Test set has residual data quality issues
3. **Quick overfitting**: Val MAE minimum at epoch 1

---

## Next Steps to Reach <10 BPM 🎯

1. **Add weight decay** (L2 regularization) - currently missing
2. **Increase batch size** to 128 (was 64)
3. **Check normalization** - HR variance may be too compressed
4. **Stratify by HR range** - ensure low/high HR in all splits
5. **Re-validate test set** - Remove remaining offset errors

---

## Repository Structure

```
SUB3_V2/
├── CLAUDE.md              # Complete guide (updated Jan 14)
├── doc/
│   ├── PROJECT_SUMMARY.md # Current status
│   ├── ROADMAP.md         # This file
│   └── TRAINING_GUIDE.md  # Desktop GPU training guide
├── DATA/
│   ├── processed/         # Train/val/test tensors (1.6GB)
│   ├── indices/           # Line-based indices
│   └── raw/               # Endomondo JSON
├── Model/                 # Training & evaluation scripts
│   ├── checkpoints/       # best_model.pt (epoch 6)
│   ├── results/           # Plots and metrics
│   └── animations/        # 5 GIF animations (7.1MB total)
├── EDA/                   # Exploration tools
└── Preprocessing/         # 3-stage pipeline
```

---

## References

- **CLAUDE.md**: Complete command reference and design patterns
- **PROJECT_SUMMARY.md**: Detailed findings and file organization
- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`

---

**Status**: ✅ Project complete, ready for v2.1 improvements
**Owner**: Riccardo
**Next**: Implement regularization improvements
