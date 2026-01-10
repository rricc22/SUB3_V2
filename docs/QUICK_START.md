# Quick Start Guide - SUB3_V2

**Last Updated**: 2025-01-10

---

## Overview

This guide will walk you through the V2 workflow from data quality validation to model training.

**Estimated time**: 1-2 weeks (depending on annotation pace)

---

## Prerequisites

### System Requirements

- **Python**: 3.11+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 10 GB for data + results

### Install Dependencies

```bash
cd /home/riccardo/Documents/Collaborative-Projects/SUB3_V2

pip install numpy pandas torch scikit-learn matplotlib plotly tqdm streamlit
```

---

## Workflow

### Phase 0: Data Quality Validation (Week 1)

#### Step 1: Sample Workouts for Annotation

```python
# In Python console or Jupyter notebook
import ast
import random

# Load raw data
workouts = []
with open('DATA/raw/endomondoHR.json', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10000:  # Load first 10K
            break
        workouts.append(ast.literal_eval(line.strip()))

# Sample 100 diverse workouts
random.seed(42)
sample_indices = random.sample(range(len(workouts)), 100)
sample_workouts = [workouts[i] for i in sample_indices]

# Save for annotation
import json
with open('DATA/quality_check/sample_workouts.json', 'w') as f:
    json.dump(sample_workouts, f, indent=2)

print(f"Sampled {len(sample_workouts)} workouts for annotation")
```

#### Step 2: Manual Annotation (Streamlit App)

**Option A: Streamlit app** (recommended for interactive workflow):

Create `EDA/quality_annotation_app.py`:

```python
import streamlit as st
import plotly.graph_objects as go
import json
import pandas as pd

# Load sample workouts
with open('DATA/quality_check/sample_workouts.json', 'r') as f:
    workouts = json.load(f)

# Load existing annotations (if any)
try:
    annotations = pd.read_csv('DATA/quality_check/annotations.csv')
except FileNotFoundError:
    annotations = pd.DataFrame(columns=[
        'workout_id', 'quality_score', 'hr_quality', 'gps_quality', 
        'altitude_quality', 'is_valid', 'notes'
    ])

st.title("Workout Quality Annotation")
st.write(f"Total workouts: {len(workouts)}")
st.write(f"Annotated: {len(annotations)}")

# Select workout
workout_idx = st.slider("Workout Index", 0, len(workouts)-1, 0)
w = workouts[workout_idx]

# Plot time-series
fig = go.Figure()
fig.add_trace(go.Scatter(y=w['speed'], name='Speed (km/h)', yaxis='y1'))
fig.add_trace(go.Scatter(y=w['altitude'], name='Altitude (m)', yaxis='y2'))
fig.add_trace(go.Scatter(y=w['heart_rate'], name='HR (BPM)', yaxis='y3'))

fig.update_layout(
    title=f"Workout {w.get('id', 'unknown')} - User {w.get('userId', 'unknown')}",
    xaxis=dict(title="Timestep"),
    yaxis=dict(title="Speed", side="left"),
    yaxis2=dict(title="Altitude", overlaying="y", side="right"),
    yaxis3=dict(title="HR", overlaying="y", side="right", position=0.85),
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Annotation form
st.subheader("Annotation")
workout_id = w.get('id', workout_idx)
quality_score = st.slider("Overall Quality (0-100)", 0, 100, 80)
hr_quality = st.selectbox("HR Sensor", ["Good", "Spikes", "Flatlines", "Dropouts", "Multiple issues"])
gps_quality = st.selectbox("GPS/Speed", ["Good", "Noisy", "Impossible speeds", "Pauses"])
altitude_quality = st.selectbox("Altitude", ["Good", "Jumps", "Noisy"])
is_valid = st.checkbox("Use in training", value=True)
notes = st.text_area("Notes")

# Save annotation
if st.button("Save & Next"):
    new_row = pd.DataFrame([{
        'workout_id': workout_id,
        'quality_score': quality_score,
        'hr_quality': hr_quality,
        'gps_quality': gps_quality,
        'altitude_quality': altitude_quality,
        'is_valid': is_valid,
        'notes': notes
    }])
    annotations = pd.concat([annotations, new_row], ignore_index=True)
    annotations.to_csv('DATA/quality_check/annotations.csv', index=False)
    st.success(f"Saved annotation for workout {workout_id}")
    st.rerun()
```

**Run the app**:
```bash
streamlit run EDA/quality_annotation_app.py
```

**Option B: Simple Jupyter notebook** (faster for solo annotation):

```bash
jupyter notebook EDA/data_quality_check.ipynb
```

#### Step 3: Analyze Annotations

After annotating 100 workouts:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load annotations
annotations = pd.read_csv('DATA/quality_check/annotations.csv')

# Summary statistics
print("Quality Score Distribution:")
print(annotations['quality_score'].describe())

print("\nQuality Grade Distribution:")
annotations['grade'] = pd.cut(
    annotations['quality_score'], 
    bins=[0, 30, 50, 70, 90, 100],
    labels=['Very Poor', 'Poor', 'Acceptable', 'Good', 'Excellent']
)
print(annotations['grade'].value_counts())

print("\nValid for training:")
print(f"{annotations['is_valid'].sum()} / {len(annotations)} ({annotations['is_valid'].mean()*100:.1f}%)")

# Plot distribution
plt.figure(figsize=(10, 6))
annotations['quality_score'].hist(bins=20)
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.title('Quality Score Distribution (n=100)')
plt.savefig('DATA/quality_check/quality_distribution.png')
```

---

### Phase 1: Preprocessing (Week 2)

#### Step 1: Implement Quality Filters

Create `Preprocessing/quality_filters.py` based on your annotation findings:

```python
#!/usr/bin/env python3
"""
Data quality validation functions.

Based on manual annotation of 100 workouts.
"""

import numpy as np

def validate_hr_sensor(hr_array, threshold=70):
    """
    Validate heart rate sensor quality.
    
    Returns:
        (is_valid, quality_score, issues)
    """
    issues = []
    quality_score = 100.0
    
    # Check 1: Spikes (>30 BPM in 1 timestep)
    hr_diff = np.abs(np.diff(hr_array))
    spike_count = np.sum(hr_diff > 30)
    if spike_count > 0:
        issues.append(f"HR spikes: {spike_count}")
        quality_score -= spike_count * 10
    
    # Check 2: Flatlines (>10 identical values)
    hr_diff_zero = (hr_diff == 0)
    max_flatline = 0
    current_flatline = 0
    for is_zero in hr_diff_zero:
        if is_zero:
            current_flatline += 1
            max_flatline = max(max_flatline, current_flatline)
        else:
            current_flatline = 0
    
    if max_flatline > 10:
        issues.append(f"HR flatline: {max_flatline} timesteps")
        quality_score -= 20
    
    # Check 3: Dropouts (HR = 0 mid-workout)
    if len(hr_array) > 20:
        dropout_count = np.sum(hr_array[10:-10] == 0)
        if dropout_count > 0:
            issues.append(f"HR dropouts: {dropout_count}")
            quality_score -= dropout_count * 5
    
    is_valid = quality_score >= threshold
    return is_valid, max(0, quality_score), issues


def validate_gps_speed(speed_array, threshold=60):
    """Validate GPS and speed measurements."""
    issues = []
    quality_score = 100.0
    
    # Check 1: Impossible speeds (>25 km/h)
    impossible_count = np.sum(speed_array > 25)
    if impossible_count > 5:
        issues.append(f"Impossible speed: {impossible_count} timesteps")
        return False, 0, issues  # Hard fail
    
    # Check 2: GPS noise
    speed_diff = np.abs(np.diff(speed_array))
    noise_level = np.std(speed_diff)
    if noise_level > 2.0:
        issues.append(f"GPS noise: std={noise_level:.2f}")
        quality_score -= 15
    
    is_valid = quality_score >= threshold
    return is_valid, max(0, quality_score), issues


def validate_workout_quality(workout, hr_threshold=70, gps_threshold=60):
    """
    Validate entire workout quality.
    
    Returns:
        (is_valid, overall_score, issues)
    """
    all_issues = []
    
    # Validate HR
    hr_valid, hr_score, hr_issues = validate_hr_sensor(
        workout['heart_rate'], hr_threshold
    )
    all_issues.extend([f"HR: {issue}" for issue in hr_issues])
    
    # Validate GPS
    gps_valid, gps_score, gps_issues = validate_gps_speed(
        workout['speed'], gps_threshold
    )
    all_issues.extend([f"GPS: {issue}" for issue in gps_issues])
    
    # Overall score (weighted average)
    overall_score = 0.6 * hr_score + 0.4 * gps_score
    is_valid = hr_valid and gps_valid
    
    return is_valid, overall_score, all_issues
```

#### Step 2: Implement Feature Engineering

Create `Preprocessing/feature_engineering.py`:

```python
#!/usr/bin/env python3
"""
Feature engineering for heart rate prediction.

Creates temporal features from raw speed and altitude.
"""

import numpy as np
import pandas as pd

def engineer_features(speed, altitude):
    """
    Create 8 engineered features.
    
    Args:
        speed: [seq_len] array
        altitude: [seq_len] array
    
    Returns:
        Dictionary with 8 feature arrays
    """
    features = {}
    
    # Lag features
    features['speed_lag_2'] = np.roll(speed, 2)
    features['speed_lag_2'][:2] = speed[0]
    
    features['speed_lag_5'] = np.roll(speed, 5)
    features['speed_lag_5'][:5] = speed[0]
    
    features['altitude_lag_30'] = np.roll(altitude, 30)
    features['altitude_lag_30'][:30] = altitude[0]
    
    # Derivatives
    features['speed_derivative'] = np.diff(speed, prepend=speed[0])
    features['altitude_derivative'] = np.diff(altitude, prepend=altitude[0])
    
    # Rolling statistics
    features['rolling_speed_10'] = (
        pd.Series(speed).rolling(10, min_periods=1).mean().values
    )
    features['rolling_speed_30'] = (
        pd.Series(speed).rolling(30, min_periods=1).mean().values
    )
    
    # Cumulative features
    elevation_gain = np.maximum(features['altitude_derivative'], 0)
    features['cumulative_elevation'] = np.cumsum(elevation_gain)
    
    return features
```

#### Step 3: Run Preprocessing

```bash
python3 Preprocessing/prepare_sequences.py
```

(Note: You'll need to implement the full `prepare_sequences.py` using the components above)

---

### Phase 2: Model Training (Week 3)

#### Step 1: Implement Model

Create `Model/lstm.py`, `Model/loss.py`, `Model/train.py` based on specifications in `docs/ARCHITECTURE.md`.

#### Step 2: Train Model

```bash
python3 Model/train.py --model lstm --epochs 100 --batch_size 16 --use_masking
```

#### Step 3: Monitor Training

```bash
# Watch training logs
tail -f logs/training_*.log

# Or use TensorBoard (if implemented)
tensorboard --logdir results/
```

---

### Phase 3: Evaluation (Week 4)

#### Step 1: Evaluate on Test Set

```bash
python3 Model/evaluate.py --checkpoint checkpoints/best_model.pt
```

#### Step 2: Compare with V1

```bash
python3 Model/compare_with_v1.py
```

Expected output:
```
V1 Baseline:  MAE 13.88 BPM, R² 0.188
V2 Model:     MAE 9.5 BPM,  R² 0.38
Improvement:  -31.6% MAE, +101.6% R²
```

---

## Troubleshooting

### Issue: Out of memory during preprocessing

**Solution**: Process in batches
```python
MAX_SAMPLES = 10000  # Instead of None
```

### Issue: Slow annotation

**Solution**: Annotate in batches
- Day 1: 20 workouts
- Day 2: 20 workouts
- etc.

### Issue: Model not converging

**Solution**: Check learning rate and masking
```python
# Verify masking is working
print(f"Mask sum: {mask.sum()}, Expected: {batch_size * avg_length}")
```

---

## Next Steps After Completion

1. **Ablation study**: Test contribution of each feature
2. **Finetuning**: Train on personal Apple Watch data
3. **Deployment**: Export to ONNX for mobile
4. **Documentation**: Write final report with results

---

**Need help?** Refer to:
- `docs/PRD.md` - Requirements
- `docs/ARCHITECTURE.md` - Technical details
- `docs/DATA_QUALITY.md` - Quality criteria
- `AGENTS.md` - Coding guidelines
