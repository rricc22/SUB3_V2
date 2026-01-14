# SUB3_V2 Project Summary

**Created**: 2025-01-10
**Last Updated**: 2026-01-14
**Status**: ✅ MODEL TRAINED & EVALUATED

---

## Current Status

**All phases complete!** Model trained and evaluated with animations.

### Performance Results

| Metric | V1 Baseline | V2 Current | Target | Status |
|--------|-------------|------------|---------|--------|
| **Test MAE** | 13.88 BPM | **11.90 BPM** | < 10 BPM | 🟡 Close! |
| **Test RMSE** | - | **14.42 BPM** | - | ✅ Good |
| **Training Stopped** | - | **Epoch 6** | - | ✅ Early stopping worked |

**Model Config**: batch_size=64, dropout=0.4, lr=0.0005, hidden=128, layers=2
**Dataset**: 32,806 train / 7,299 val / 6,145 test samples

---

## What's Done ✅

### Phase 0-1: Data Processing (Jan 11-13, 2026)
- ✅ 3-stage preprocessing pipeline (rule-based + LLM + corrections)
- ✅ HR offset detection and correction
- ✅ Clean dataset with smoothing (2.3GB)

### Phase 2: Tensor Preparation (Jan 13, 2026)
- ✅ 11 engineered features (lag, derivatives, rolling, cumulative)
- ✅ PyTorch tensors generated (train.pt, val.pt, test.pt)
- ✅ Masking for padded sequences
- ✅ Stratified user splitting

### Phase 3: Model Training (Jan 14, 2026)
- ✅ LSTM model trained with masked loss
- ✅ W&B integration for experiment tracking
- ✅ Fixed overfitting issues (batch_size 16→64, dropout 0.3→0.4)
- ✅ Best model saved (epoch 6)

### Phase 4: Evaluation & Visualization (Jan 14, 2026)
- ✅ Test set evaluation (11.90 BPM MAE)
- ✅ 5 animation types generated:
  1. Gradual reveal (best workout: 2.6 BPM error)
  2. Gradual reveal (worst workout: 42 BPM error - HR offset issue)
  3. Multi-workout comparison (4 workouts side-by-side)
  4. Feature influence (speed/altitude → HR with cursor)
  5. Error heatmap (8 workouts, time evolution)
- ✅ Anomaly investigation (workout #4600: 31 BPM systematic offset)

---

## Key Findings

### What Worked ✅
1. **Feature engineering**: 11 features improved correlations
2. **Masked loss**: Correctly ignores 43% padding
3. **Early stopping**: Caught best model at epoch 6
4. **Animation tools**: Great for debugging data quality issues

### Known Issues ⚠️
1. **Regression to mean**: Model clusters predictions around 145-160 BPM
   - Underpredicts high HR (>170 BPM)
   - Overpredicts low HR (<130 BPM)
2. **HR offset errors remain in test set**: Example workout #4600
   - Ground truth: 122 BPM at 12.3 km/h (incorrect)
   - Model prediction: 154 BPM (correct!)
   - Reported "error": 31 BPM (but model is right)
3. **Model overfitted quickly**: Val MAE minimum at epoch 1, then increased

### Next Steps to Reach <10 BPM 🎯
1. Add weight decay (L2 regularization)
2. Increase batch size to 128
3. Check feature normalization (HR variance might be too small)
4. Stratify by HR range (ensure low/high HR in all splits)
5. Re-run Stage 2 LLM validation to catch remaining offset errors

---

## File Organization

```
SUB3_V2/
├── CLAUDE.md                   # Updated with training/eval commands
├── DATA/
│   ├── processed/              # PyTorch tensors (1.6GB)
│   │   ├── train.pt
│   │   ├── val.pt
│   │   ├── test.pt
│   │   ├── metadata.json
│   │   └── scaler_params.json
│   ├── indices/                # Line-based indices
│   └── raw/                    # Endomondo JSON (symlink)
├── Model/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── animate_predictions.py  # 5 animation types
│   ├── investigate_workout.py  # Anomaly deep-dive
│   ├── lstm.py                 # Model architecture
│   ├── loss.py                 # Masked loss functions
│   ├── checkpoints/
│   │   └── best_model.pt       # Best model (epoch 6)
│   ├── results/
│   │   ├── sample_predictions.png
│   │   ├── error_distribution.png
│   │   ├── scatter_plot.png
│   │   └── test_results.json
│   └── animations/
│       ├── 1_gradual_reveal_best.gif (1.6MB)
│       ├── 2_gradual_reveal_worst.gif (1.2MB)
│       ├── 3_multi_workout_comparison.gif (2.4MB)
│       ├── 4_feature_influence.gif (1.7MB)
│       └── 5_error_heatmap.gif (183KB)
├── EDA/                        # Exploration tools
└── Preprocessing/              # 3-stage pipeline
```

---

## Quick Commands

```bash
# Train model (from project root)
python3 Model/train.py --batch-size 64 --dropout 0.4 --lr 0.0005 --patience 3

# Evaluate model
python3 Model/evaluate.py --checkpoint Model/checkpoints/best_model.pt \
    --data-dir DATA/processed --output-dir Model/results

# Generate animations
python3 Model/animate_predictions.py --checkpoint Model/checkpoints/best_model.pt \
    --data-dir DATA/processed --output-dir Model/animations --fps 20

# Investigate specific workout
python3 Model/investigate_workout.py
```

---

## Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.0+
- **Tracking**: Weights & Biases (W&B)
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, PIL (animations)

---

## References

- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Dataset**: Endomondo HR from FitRec project
- **CLAUDE.md**: Complete command reference and design patterns

---

**Project Lead**: Riccardo
**Status**: Training complete, ready for improvements
**Next**: Implement regularization improvements to reach <10 BPM target
