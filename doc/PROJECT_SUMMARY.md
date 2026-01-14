# SUB3_V2 Project Summary

**Created**: 2025-01-10
**Last Updated**: 2026-01-14
**Status**: ✅ COMPLETE - DEPLOYED TO HUGGING FACE

---

## Current Status

**All phases complete!** Model trained, evaluated, and fully deployed to Hugging Face.

### Performance Results - V2 Final

| Metric | V1 Baseline | V1 Best | V2 Final | Improvement | Status |
|--------|-------------|---------|----------|-------------|--------|
| **Test MAE** | 13.88 BPM | 8.94 BPM | **7.42 BPM** | **17%** over V1 best | ✅ **TARGET MET!** |
| **Test RMSE** | - | - | **9.87 BPM** | - | ✅ Excellent |
| **Total Improvement** | - | - | **46.5%** | From 13.88 to 7.42 | 🎯 **SUCCESS** |

**Model Config**: batch_size=16, dropout=0.3, lr=0.0005, hidden=128, layers=2, **14 features**
**Dataset**: 28,130 train / 6,027 val / 6,029 test samples (40,186 total workouts)

---

## What's Done ✅

### Phase 0-1: Data Processing (Jan 11-13, 2026)
- ✅ 3-stage preprocessing pipeline (rule-based + LLM + corrections)
- ✅ HR offset detection and correction
- ✅ Clean dataset with smoothing (2.3GB → 40,186 workouts)
- ✅ Converted to Parquet format for Hugging Face (647 MB)

### Phase 2: Feature Engineering (Jan 13-14, 2026)
- ✅ **14 engineered features** (was 11, added 3 temporal features)
  - Base: speed, altitude, gender
  - Lag: speed_lag_2, speed_lag_5, altitude_lag_30
  - Derivatives: speed_derivative, altitude_derivative
  - Rolling: rolling_speed_10, rolling_speed_30
  - Cumulative: cumulative_elevation_gain
  - **New**: 3 temporal features (indices 11-13) for physiological modeling
- ✅ PyTorch tensors generated (train.pt, val.pt, test.pt) - 1.6GB
- ✅ Masking for padded sequences
- ✅ Stratified user splitting (70/15/15)

### Phase 3: Model Training V2 (Jan 14, 2026)
- ✅ LSTM model trained with masked loss
- ✅ W&B integration for experiment tracking
- ✅ Optimal hyperparameters: batch_size=16, dropout=0.3, hidden=128
- ✅ Best model saved: **7.42 BPM MAE** (17% improvement over V1 best)
- ✅ Model architecture: 206K parameters with 14 input features

### Phase 4: Evaluation & Visualization (Jan 14, 2026)
- ✅ Test set evaluation: **7.42 BPM MAE**, 9.87 BPM RMSE
- ✅ Animation types generated:
  - Category-based animations (recovery, steady, intensive)
  - Single workout animations for demo
  - Multi-workout comparisons
- ✅ Performance analysis by workout type

### Phase 5: Deployment to Hugging Face (Jan 14, 2026)
- ✅ **Model Repository**: [rricc22/heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm)
  - V2 model checkpoint (2.4 MB)
  - Training strategy documentation
  - 3 animated GIFs from test set
- ✅ **Dataset Repository**: [rricc22/endomondo-hr-prediction-v2](https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2)
  - Parquet format (647 MB)
  - Dataset viewer compatible
  - Train/val/test splits included
- ✅ **Interactive Demo**: [rricc22/heart-rate-predictor](https://huggingface.co/spaces/rricc22/heart-rate-predictor)
  - Gradio app with V2 model
  - Upload custom workouts
  - Sample workouts included
- ✅ **Collection**: [Heart Rate Prediction from Running Data](https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3)
  - Groups model, dataset, and demo
  - Public and accessible

---

## Key Findings

### What Worked ✅
1. **Feature engineering with 14 features**: Added 3 temporal features improved MAE from 11.90 to 7.42 BPM
2. **Masked loss**: Correctly ignores 43% padding
3. **Data quality**: 3-stage preprocessing pipeline caught HR offset errors
4. **Hugging Face deployment**: Complete ecosystem (model + dataset + demo)
5. **Parquet format**: Dataset viewer now works, users can explore data

### V2 Achievements 🎯
1. **7.42 BPM MAE** - Beat target of <10 BPM
2. **17% improvement** over V1 best model (8.94 BPM)
3. **46.5% total improvement** from V1 baseline (13.88 BPM)
4. **Full deployment** to Hugging Face with collection
5. **Dataset viewer** working with 40K browseable workouts

### Deployment Success ✅
- **Model Hub**: V2 checkpoint with training strategy
- **Dataset Hub**: Parquet format with viewer support
- **Spaces**: Interactive demo with V2 model
- **Collection**: All components grouped together
- **Documentation**: Complete usage examples and README files

---

## File Organization

```
SUB3_V2/
├── COLLECTION_INFO.md          # HuggingFace collection details
├── DEPLOYMENT_COMPLETE.md      # Full deployment summary
├── doc/
│   ├── PROJECT_SUMMARY.md      # This file
│   ├── ROADMAP.md              # Development timeline
│   └── TRAINING_GUIDE.md       # Training documentation
├── DATA/
│   ├── processed_v2/           # PyTorch tensors (1.6GB)
│   │   ├── train.pt            # 28,130 workouts
│   │   ├── val.pt              # 6,027 workouts
│   │   ├── test.pt             # 6,029 workouts
│   │   ├── metadata.json
│   │   └── scaler_params.json
│   ├── huggingface_dataset/    # Parquet files for HF (647MB)
│   │   ├── dataset.parquet     # Full dataset
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── README.md
│   └── CONVERSION_SUMMARY.md   # Dataset conversion details
├── Model/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── animate_by_category.py  # Category animations
│   ├── lstm.py                 # Model architecture (14 features)
│   ├── loss.py                 # Masked loss functions
│   ├── checkpoints_v2/
│   │   └── best_model.pt       # V2 model (7.42 BPM MAE)
│   ├── results_v2/
│   │   ├── sample_predictions.png
│   │   ├── error_distribution.png
│   │   ├── scatter_plot.png
│   │   └── test_results.json   # 7.42 BPM MAE
│   └── animations_v2_single/
│       ├── steady_workout.gif
│       ├── intervals_workout.gif
│       └── progressive_workout.gif
├── huggingface_deployment/      # HF Space files
│   ├── app.py                   # Gradio app (V2 model)
│   ├── best_model.pt            # V2 checkpoint
│   ├── README.md                # Space documentation
│   ├── HF_MODEL_CARD.md         # Model card
│   └── *.gif                    # Animated predictions
├── Preprocessing/
│   ├── convert_to_hf_format.py  # JSON → Parquet converter
│   └── clean_dataset_v2.json    # Clean dataset (2.3GB)
└── EDA/                         # Exploration tools
```

---

## Quick Commands

```bash
# Train V2 model (from project root)
python3 Model/train.py --batch-size 16 --dropout 0.3 --lr 0.0005 --patience 10

# Evaluate V2 model
python3 Model/evaluate.py --checkpoint Model/checkpoints_v2/best_model.pt \
    --data-dir DATA/processed_v2 --output-dir Model/results_v2

# Generate category animations
python3 Model/animate_by_category.py --checkpoint Model/checkpoints_v2/best_model.pt \
    --data-dir DATA/processed_v2 --output-dir Model/animations_v2_category

# Convert dataset to Parquet for HuggingFace
python3 Preprocessing/convert_to_hf_format.py

# Load dataset from HuggingFace
python3 -c "from datasets import load_dataset; ds = load_dataset('rricc22/endomondo-hr-prediction-v2'); print(ds)"
```

---

## Hugging Face Links

- **Collection**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3
- **Model**: https://huggingface.co/rricc22/heart-rate-prediction-lstm
- **Dataset**: https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2
- **Demo**: https://huggingface.co/spaces/rricc22/heart-rate-predictor

---

## Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.0+
- **Tracking**: Weights & Biases (W&B)
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, PIL (animations)
- **Deployment**: Hugging Face (Model Hub, Datasets, Spaces)
- **Data Format**: Parquet (for HF compatibility)
- **Demo**: Gradio

---

## References

- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Dataset**: Endomondo HR from FitRec project (40,186 running workouts)
- **Collection**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3
- **Documentation**: COLLECTION_INFO.md, DEPLOYMENT_COMPLETE.md

---

**Project Lead**: Riccardo
**Status**: ✅ Complete - Fully deployed to Hugging Face
**Achievement**: 7.42 BPM MAE (46.5% improvement from baseline)
**Date**: January 14, 2026
