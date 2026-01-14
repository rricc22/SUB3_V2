# Hugging Face Collection - Heart Rate Prediction

## ✅ Collection Created Successfully!

**URL**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3

### 📦 Contents

| Type | Repository | Description |
|------|------------|-------------|
| 🤖 **Model** | [heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm) | V2 LSTM achieving 7.42 BPM MAE (17% improvement) |
| 📊 **Dataset** | [endomondo-hr-prediction-v2](https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2) | 40,186 quality-filtered running workouts |
| 🎮 **Demo** | [heart-rate-predictor](https://huggingface.co/spaces/rricc22/heart-rate-predictor) | Interactive Gradio app for predictions |

### 📝 Current Description

```
Complete ML system for predicting heart rate from running speed/altitude. 
LSTM model (7.42 BPM MAE), 40K workouts dataset, interactive demo.
```

### 🎨 Optional: Enhance Collection Description

To add a more detailed description to your collection:

1. **Go to**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3
2. **Click** "Edit collection" (top right)
3. **Replace** the description with the enhanced version below

---

## 📄 Enhanced Description (Copy & Paste)

```markdown
Complete machine learning system for predicting heart rate time-series from running speed and altitude data.

## 🎯 Overview

This collection provides everything needed to predict heart rate responses during running:
- **LSTM Model** achieving 7.42 BPM Mean Absolute Error
- **40K+ running workouts** from real athletes (Endomondo dataset)
- **Interactive demo** to test predictions on your own data

## 🚀 Performance

- **7.42 BPM MAE** on 6,029 unseen test workouts
- **17% improvement** over baseline (from 8.94 to 7.42 BPM)
- **14 engineered features** capturing physiological response delays
- **206K parameters** LSTM trained on 28,130 workouts

## 📊 Dataset Highlights

**40,186 quality-filtered running workouts** from 761 athletes:
- RECOVERY runs: 15,095 (easy pace, low intensity)
- STEADY runs: 22,991 (moderate pace)
- INTENSIVE workouts: 2,100 (high intensity, intervals)

Features include:
- Heart rate, speed, altitude time-series (500 timesteps max)
- Pre-computed statistics (HR mean, speed mean, elevation gain)
- Train/validation/test splits (70/15/15)
- Smoothed with 7-point moving average

## 🎮 Try the Demo

The interactive Space lets you:
1. Upload your own running workout data (JSON format)
2. Select from sample workouts (recovery, steady, intensive)
3. See real-time predictions with animated visualizations
4. Compare predicted vs. actual heart rate

## 🛠️ Technical Details

**Model**: 2-layer LSTM (hidden_size=128, dropout=0.3)

**Features** (14 total):
- Base: speed, altitude, gender
- Lag: speed_lag_2, speed_lag_5, altitude_lag_30
- Derivatives: speed change, altitude change
- Rolling: 10-step, 30-step windows
- Cumulative: total elevation gain
- **New in V2**: 3 temporal features for physiological modeling

**Training**: Adam optimizer, early stopping, gradient clipping, masked loss

## 📚 Use Cases

- Exercise physiology research
- Training optimization and HR zone prediction
- Wearable device validation
- Anomaly detection in HR patterns
- Personalized coaching systems

## 📖 Citation

If you use this work, please cite:

\`\`\`bibtex
@misc{heart_rate_prediction_v2,
  title={Heart Rate Prediction from Running Data using LSTM},
  author={Riccardo},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data}}
}
\`\`\`

## 📜 License

MIT License - Free for academic and commercial use
```

---

## 🎯 Quick Links

### Model
- **Card**: https://huggingface.co/rricc22/heart-rate-prediction-lstm
- **Files**: best_model.pt (2.4 MB), training animations
- **Performance**: 7.42 BPM MAE on test set

### Dataset
- **Viewer**: https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2
- **Format**: Parquet (647 MB total)
- **Splits**: train (28K), validation (6K), test (6K)
- **Features**: 20 columns including time-series and statistics

### Demo Space
- **App**: https://huggingface.co/spaces/rricc22/heart-rate-predictor
- **Status**: Running on CPU
- **Features**: Upload data, sample workouts, animated predictions

---

## 📊 Complete Ecosystem

```
┌─────────────────────────────────────────────────────────┐
│         Heart Rate Prediction Collection                │
└─────────────────────────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐    ┌──────────┐
    │  Model   │     │ Dataset  │    │   Demo   │
    │          │     │          │    │          │
    │ 7.42 BPM │────▶│  40K     │───▶│ Gradio   │
    │   MAE    │     │ Workouts │    │   App    │
    └──────────┘     └──────────┘    └──────────┘
         │                │                │
         │                │                │
         └────────────────┴────────────────┘
                          │
                          ▼
              Users can explore, test, 
              and reproduce results
```

---

## 🎉 What's Included

### ✅ Model Repository
- [x] V2 model checkpoint (best_model.pt)
- [x] Training strategy documentation
- [x] Animated predictions (3 GIFs)
- [x] Performance metrics and benchmarks
- [x] Model card with architecture details

### ✅ Dataset Repository
- [x] Parquet files (train/val/test splits)
- [x] Dataset viewer compatibility
- [x] Comprehensive README with examples
- [x] Quality filtering documentation
- [x] Statistics and distributions

### ✅ Demo Space
- [x] Interactive Gradio interface
- [x] Sample workout files (3 types)
- [x] Upload custom data feature
- [x] Animated visualization
- [x] V2 model integration

---

## 📅 Timeline

| Date | Action |
|------|--------|
| Jan 14, 2026 | V2 model trained (7.42 BPM MAE) |
| Jan 14, 2026 | Dataset converted to Parquet |
| Jan 14, 2026 | Demo updated with V2 model |
| Jan 14, 2026 | Collection created |

---

**Status**: ✅ Complete and Live

**Collection URL**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3
