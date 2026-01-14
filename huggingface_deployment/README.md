---
title: Heart Rate Predictor
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# Heart Rate Predictor for Runners

This is an interactive demo of a deep learning model that predicts heart rate (BPM) from running workout data.

## How to Use

1. **Manual Input**: Enter your speed and altitude data manually (comma-separated values)
2. **CSV Upload**: Upload a CSV file with `speed` and `altitude` columns
3. **View Predictions**: See predicted heart rate visualized alongside your input data

## Model Details

- **Architecture**: 2-layer LSTM with 128 hidden units
- **Parameters**: ~206K trainable parameters
- **Performance**: **7.42 BPM MAE** on test set (17% better than best V1!)
- **Dataset**: 974 quality-filtered running workouts (Endomondo)

## Input Format

- **Speed**: km/h (e.g., 10.5, 11.0, 12.0)
- **Altitude**: meters (e.g., 100, 105, 110)
- **Gender**: Male or Female
- **Timestep**: Each data point represents ~6 seconds

## Limitations

- Trained on recreational runners (European population)
- Best for 8-15 km/h pace range
- Does not account for individual fitness, age, or environment
- **Not for medical use** - research/training optimization only

## Model Repository

Full model and code: [rricc22/heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm)
