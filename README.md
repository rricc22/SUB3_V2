# SUB3_V2: Heart Rate Prediction from Running Data

**Version**: 2.0  
**Goal**: Predict heart rate time-series from speed and altitude during running workouts  
**Target**: Sub-3-hour marathon training optimization through personalized HR prediction
**Status**: ✅ **COMPLETE - DEPLOYED TO HUGGING FACE**

---

## 🎉 V2 Final Results

| Metric | V1 Baseline | V1 Best | V2 Final | Improvement |
|--------|-------------|---------|----------|-------------|
| **Test MAE** | 13.88 BPM | 8.94 BPM | **7.42 BPM** | **17%** from V1 best |
| **Total Improvement** | - | - | **46.5%** | From V1 baseline |
| **Features** | 3 | 3 | **14** | +367% |
| **Dataset** | 974 | 974 | **40,186** | +4,027% |

**🎯 TARGET ACHIEVED!** MAE < 10 BPM on base model (without personal finetuning)

---

## 🚀 Quick Start - Use the Model

### Try the Interactive Demo
**URL**: https://huggingface.co/spaces/rricc22/heart-rate-predictor

Upload your own running workout or try sample workouts!

### Load from Hugging Face

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch

# Load dataset
dataset = load_dataset("rricc22/endomondo-hr-prediction-v2")
train_data = dataset['train']  # 28,130 workouts

# Load V2 model
model_path = hf_hub_download(
    repo_id="rricc22/heart-rate-prediction-lstm",
    filename="best_model.pt"
)
checkpoint = torch.load(model_path)

# Explore data
workout = train_data[0]
print(f"Type: {workout['workout_type']}, HR: {workout['hr_mean']:.1f} BPM")
```

---

## 📦 Hugging Face Collection

**Collection**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3

### What's Included

1. **🤖 Model** - [heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm)
   - V2 LSTM: 7.42 BPM MAE (206K parameters)
   - 14 engineered features
   - Training strategy documentation

2. **📊 Dataset** - [endomondo-hr-prediction-v2](https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2)
   - 40,186 running workouts (Parquet format)
   - Dataset viewer compatible
   - Train/val/test splits included

3. **🎮 Demo** - [heart-rate-predictor](https://huggingface.co/spaces/rricc22/heart-rate-predictor)
   - Interactive Gradio app
   - Upload custom workouts
   - Real-time predictions

---

## Quick Reference: Model Evolution

| Model | Architecture | Parameters | Test MAE | Status |
|-------|-------------|------------|----------|--------|
| V1 LSTM Baseline | 2-layer LSTM (64 units) | ~50K | 13.88 BPM | Baseline |
| V1 Finetuned Stage 1 | LSTM + Transfer Learning | ~50K | 8.94 BPM | Previous Best |
| **V2 Final** | **2-layer LSTM (128 units, 14 features)** | **206K** | **7.42 BPM** | **✅ CURRENT BEST** |

---

## Why Version 2.0 Succeeded

### Key Improvements ✅

1. **14 Engineered Features** (was 3)
   - Added lag features (physiological delays)
   - Derivatives (acceleration patterns)
   - Rolling averages (smoothed patterns)
   - Cumulative elevation (fatigue modeling)
   - **3 new temporal features** for V2 final

2. **Massive Dataset Expansion**
   - From 974 → 40,186 workouts (+4,027%)
   - Quality filtering pipeline
   - Parquet format for easy exploration

3. **Better Architecture**
   - Hidden size: 64 → 128 units
   - Dropout: 0.2 → 0.3 (optimal)
   - Masked loss (ignores padding)
   - Early stopping

4. **Full Deployment**
   - Model Hub with checkpoint
   - Dataset Hub with viewer
   - Interactive demo (Gradio)
   - Complete collection

---

## Project Structure

```
SUB3_V2/
├── README.md                    # This file
├── COLLECTION_INFO.md           # HuggingFace collection details
├── DEPLOYMENT_COMPLETE.md       # Full deployment summary
│
├── doc/
│   ├── PROJECT_SUMMARY.md       # Current status and achievements
│   ├── ROADMAP.md              # Development timeline
│   └── TRAINING_GUIDE.md       # Training documentation
│
├── DATA/
│   ├── processed_v2/           # PyTorch tensors (1.6GB)
│   │   ├── train.pt            # 28,130 workouts
│   │   ├── val.pt              # 6,027 workouts
│   │   ├── test.pt             # 6,029 workouts
│   │   └── metadata.json
│   ├── huggingface_dataset/    # Parquet files (647MB)
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── README.md
│   └── CONVERSION_SUMMARY.md
│
├── Model/
│   ├── train.py                # Training script (V2)
│   ├── evaluate.py             # Evaluation script
│   ├── animate_by_category.py  # Animation generator
│   ├── lstm.py                 # LSTM (14 features)
│   ├── checkpoints_v2/
│   │   └── best_model.pt       # V2 model (7.42 BPM MAE)
│   └── results_v2/             # Evaluation outputs
│
├── huggingface_deployment/     # Deployed files
│   ├── app.py                  # Gradio demo
│   ├── best_model.pt           # V2 checkpoint
│   └── README.md
│
├── Preprocessing/
│   ├── convert_to_hf_format.py # JSON → Parquet
│   └── clean_dataset_v2.json   # Clean data (2.3GB)
│
└── EDA/                        # Exploration tools
```

---

## Quick Start - Local Development

### 1. Load Dataset from Hugging Face

```bash
pip install datasets
python3 -c "from datasets import load_dataset; ds = load_dataset('rricc22/endomondo-hr-prediction-v2'); print(ds)"
```

### 2. Train Model Locally

```bash
# Train V2 model
python3 Model/train.py --batch-size 16 --dropout 0.3 --lr 0.0005 --patience 10

# Evaluate
python3 Model/evaluate.py --checkpoint Model/checkpoints_v2/best_model.pt
```

### 3. Generate Animations

```bash
python3 Model/animate_by_category.py --checkpoint Model/checkpoints_v2/best_model.pt
```

---

## Features (14 Total)

**Base** (3):
- speed [km/h], altitude [m], gender [binary]

**Lag** (3): Physiological response delays
- speed_lag_2 (2 timesteps)
- speed_lag_5 (5 timesteps)
- altitude_lag_30 (30 timesteps)

**Derivatives** (2): Acceleration patterns
- speed_derivative (Δ speed)
- altitude_derivative (Δ altitude)

**Rolling** (2): Smoothed patterns
- rolling_speed_10 (10-step window)
- rolling_speed_30 (30-step window)

**Cumulative** (1): Fatigue modeling
- cumulative_elevation_gain

**New in V2** (3): Temporal features
- Medium-term lag
- Short-term smoothing
- Long-term elevation context

---

## Performance by Workout Type

| Type | Count | Avg MAE | Description |
|------|-------|---------|-------------|
| RECOVERY | 15,095 | ~6.5 BPM | Easy pace, low intensity |
| STEADY | 22,991 | ~7.2 BPM | Moderate pace |
| INTENSIVE | 2,100 | ~9.8 BPM | High intensity, intervals |

---

## Development Status

- [x] V2 model trained (7.42 BPM MAE)
- [x] Dataset expanded (40K+ workouts)
- [x] Feature engineering (14 features)
- [x] Model deployed to HuggingFace
- [x] Dataset deployed with viewer
- [x] Interactive demo created
- [x] Collection published
- [x] Documentation complete

---

## Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.0+
- **Deployment**: Hugging Face (Model Hub, Datasets, Spaces)
- **Tracking**: Weights & Biases
- **Data**: NumPy, Pandas, Parquet
- **Visualization**: Matplotlib, Gradio
- **Demo**: Gradio

---

## References & Links

- **HuggingFace Collection**: https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data-6967bfe0851dd527480d6bd3
- **Model Hub**: https://huggingface.co/rricc22/heart-rate-prediction-lstm
- **Dataset Hub**: https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2
- **Interactive Demo**: https://huggingface.co/spaces/rricc22/heart-rate-predictor
- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Dataset Source**: Endomondo HR dataset from FitRec project

---

## Citation

If you use this work, please cite:

```bibtex
@misc{heart_rate_prediction_v2,
  title={Heart Rate Prediction from Running Data using LSTM},
  author={Riccardo},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/collections/rricc22/heart-rate-prediction-from-running-data}}
}
```

---

**Project Lead**: Riccardo  
**Version**: 2.0  
**Status**: ✅ Complete and Deployed  
**Achievement**: 7.42 BPM MAE (46.5% improvement from baseline)  
**Date**: January 14, 2026
