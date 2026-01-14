# SUB3_V2 Roadmap

**Last Updated**: 2026-01-14
**Current Status**: ✅ All phases complete + Deployed to Hugging Face

---

## Progress Overview

```
████████████████████████████████████ 100% COMPLETE ✅

✅ Phase 0: Data Exploration (Jan 11-13)
✅ Phase 1: Data Preprocessing (Jan 11-13)
✅ Phase 2: Feature Engineering (Jan 13-14)
✅ Phase 3: Model Training V2 (Jan 14)
✅ Phase 4: Evaluation & Viz (Jan 14)
✅ Phase 5: Deployment to HF (Jan 14)
```

---

## Timeline Summary

| Phase | Duration | Status | Results |
|-------|----------|--------|---------|
| **Phase 0: Data Exploration** | 3 days | ✅ Complete | EDA tools, gallery, indices |
| **Phase 1: Preprocessing** | 2 days | ✅ Complete | 3-stage pipeline, 40K workouts |
| **Phase 2: Feature Engineering** | 1 day | ✅ Complete | 14 features, tensors (1.6GB) |
| **Phase 3: Training V2** | 1 day | ✅ Complete | **7.42 BPM MAE** |
| **Phase 4: Evaluation** | 1 day | ✅ Complete | Animations, analysis |
| **Phase 5: HF Deployment** | 1 day | ✅ Complete | Model + Dataset + Demo + Collection |

**Total Time**: ~9 days (Jan 10-14, 2026)

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
- Output: 40,186 quality-filtered workouts (2.3GB)

### Phase 2: Feature Engineering ✅
- **14 engineered features** (expanded from 11):
  - Base (3): speed, altitude, gender
  - Lag (3): speed_lag_2, speed_lag_5, altitude_lag_30
  - Derivatives (2): speed_derivative, altitude_derivative
  - Rolling (2): rolling_speed_10, rolling_speed_30
  - Cumulative (1): cumulative_elevation_gain
  - **New (3)**: Temporal features for physiological modeling
- Padding/truncation with masking
- Stratified user splitting (70/15/15)
- Output: train.pt, val.pt, test.pt (1.6GB total)

### Phase 3: Model Training V2 ✅
- LSTM architecture (14 inputs, 128 hidden, 2 layers, dropout 0.3)
- Masked MSE loss (ignores padding)
- W&B experiment tracking
- Training config: batch_size=16, lr=0.0005, patience=10
- **Result: 7.42 BPM MAE** (17% improvement over V1 best)

### Phase 4: Evaluation & Visualization ✅
- **Test Results**: **7.42 BPM MAE**, 9.87 BPM RMSE
- **Animations**:
  - Category-based (recovery, steady, intensive)
  - Single workout animations for demo
  - Multi-workout comparisons
- Performance analysis by workout type

### Phase 5: Deployment to Hugging Face ✅
- **Model Repository**: V2 checkpoint + training docs + animations
- **Dataset Repository**: Parquet format (647MB) + dataset viewer
- **Interactive Demo**: Gradio app with V2 model
- **Collection**: All components grouped together
- **Documentation**: Complete usage examples

---

## Performance Summary

| Metric | V1 Baseline | V1 Best | V2 Final | Target | Status |
|--------|-------------|---------|----------|--------|--------|
| **Test MAE** | 13.88 BPM | 8.94 BPM | **7.42 BPM** | < 10 BPM | ✅ **EXCEEDED** |
| **Improvement vs V1 Best** | - | - | **17.0%** | - | ✅ Excellent |
| **Total Improvement** | - | - | **46.5%** | -28% | ✅ **+18.5%** |
| **Features** | 3 | 3 | **14** | - | ✅ +367% |
| **Dataset Size** | 974 | 974 | **40,186** | - | ✅ +4,027% |

---

## Key Findings

### Success Factors ✅
1. **Feature engineering**: 14 features captured physiological patterns
2. **Massive dataset**: 40K workouts vs 974 in V1
3. **Data quality**: 3-stage preprocessing caught offset errors
4. **Masked loss**: Properly handled 43% padding
5. **Optimal hyperparameters**: batch_size=16, dropout=0.3, hidden=128
6. **Full deployment**: Complete HuggingFace ecosystem

### V2 Achievements 🎯
1. **7.42 BPM MAE** - Exceeded target of <10 BPM
2. **17% improvement** over V1 best model
3. **46.5% total improvement** from V1 baseline
4. **Dataset viewer** working with 40K browseable workouts
5. **Interactive demo** deployed and accessible
6. **Complete collection** on Hugging Face

---

## Deployment Artifacts

### Hugging Face
- **Collection**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3
- **Model**: https://huggingface.co/rricc22/heart-rate-prediction-lstm
- **Dataset**: https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2
- **Demo**: https://huggingface.co/spaces/rricc22/heart-rate-predictor

### Local Files
- `Model/checkpoints_v2/best_model.pt` - V2 model (7.42 BPM MAE)
- `DATA/huggingface_dataset/` - Parquet files (647MB)
- `huggingface_deployment/` - Deployed Space files
- `COLLECTION_INFO.md` - Collection documentation
- `DEPLOYMENT_COMPLETE.md` - Full deployment summary

---

## Future Improvements (Optional)

Since the target was exceeded, these are optional enhancements:

1. **Personalization**: User embeddings for athlete-specific predictions
2. **Attention mechanism**: Focus on important timesteps
3. **Transformer model**: Compare against LSTM baseline
4. **Real-time inference**: Optimize for edge devices
5. **Additional modalities**: Include cadence, power data

---

## Repository Structure

```
SUB3_V2/
├── doc/
│   ├── PROJECT_SUMMARY.md # Updated with V2 results
│   ├── ROADMAP.md         # This file
│   └── TRAINING_GUIDE.md  # Desktop GPU training guide
├── COLLECTION_INFO.md     # HF collection documentation
├── DEPLOYMENT_COMPLETE.md # Full deployment summary
├── DATA/
│   ├── processed_v2/      # Train/val/test tensors (1.6GB)
│   ├── huggingface_dataset/ # Parquet files (647MB)
│   └── CONVERSION_SUMMARY.md
├── Model/                 # Training & evaluation scripts
│   ├── checkpoints_v2/    # best_model.pt (7.42 BPM MAE)
│   ├── results_v2/        # Plots and metrics
│   └── animations_v2_single/ # Demo animations
├── huggingface_deployment/ # Deployed Space files
├── Preprocessing/         # 3-stage pipeline + conversion
└── EDA/                   # Exploration tools
```

---

## References

- **Collection**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3
- **PROJECT_SUMMARY.md**: Detailed findings and V2 results
- **DEPLOYMENT_COMPLETE.md**: Full deployment details
- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`

---

**Status**: ✅ Project complete and deployed
**Owner**: Riccardo
**Achievement**: 7.42 BPM MAE (46.5% improvement from baseline)
**Date**: January 14, 2026
