#!/usr/bin/env python3
"""
Convert clean_dataset_v2.json to Hugging Face compatible format.

Transforms nested JSON structure into tabular Parquet format that HF dataset
viewer can display and explore.

Author: OpenCode
Date: 2026-01-14
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import numpy as np

def load_dataset(json_path: str) -> Dict:
    """Load the original JSON dataset."""
    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data['workouts'])} workouts")
    return data

def convert_to_tabular(data: Dict) -> pd.DataFrame:
    """
    Convert nested JSON to flat tabular format.
    
    Strategy:
    - Each workout becomes one row
    - Time-series arrays (heart_rate, speed, altitude) stored as lists
    - Metadata becomes separate columns
    
    Args:
        data: Original nested JSON structure
    
    Returns:
        DataFrame with one workout per row
    """
    print("\nConverting to tabular format...")
    
    rows = []
    for workout in data['workouts']:
        row = {
            # Identifiers
            'workout_id': workout['workout_id'],
            'user_id': workout['user_id'],
            
            # Metadata
            'sport': workout['sport'],
            'workout_type': workout['workout_type'],
            'duration_min': workout['duration_min'],
            'data_points': workout['data_points'],
            
            # Quality flags
            'corrected': workout['corrected'],
            'offset_applied': workout.get('offset_applied', None),
            
            # Time-series data (as lists)
            'heart_rate': workout['heart_rate'],
            'speed': workout['speed'],
            'altitude': workout['altitude'],
            'timestamp': workout['timestamp'],
            
            # Statistics (computed for easy filtering)
            'hr_mean': np.mean(workout['heart_rate']),
            'hr_std': np.std(workout['heart_rate']),
            'hr_min': np.min(workout['heart_rate']),
            'hr_max': np.max(workout['heart_rate']),
            'speed_mean': np.mean(workout['speed']),
            'speed_max': np.max(workout['speed']),
            'altitude_gain': np.sum([max(0, workout['altitude'][i+1] - workout['altitude'][i]) 
                                     for i in range(len(workout['altitude'])-1)]),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"✓ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df

def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add train/val/test split column for reproducibility.
    
    Uses same split as V2 model training:
    - Train: 70%
    - Val: 15%
    - Test: 15%
    """
    print("\nAdding train/val/test splits...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Shuffle indices
    indices = np.random.permutation(len(df))
    
    # Calculate split sizes
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    # Assign splits
    splits = ['train'] * len(df)
    for i in indices[train_size:train_size + val_size]:
        splits[i] = 'validation'
    for i in indices[train_size + val_size:]:
        splits[i] = 'test'
    
    df['split'] = splits
    
    print(f"✓ Train: {sum(s == 'train' for s in splits)} workouts")
    print(f"✓ Val: {sum(s == 'validation' for s in splits)} workouts")
    print(f"✓ Test: {sum(s == 'test' for s in splits)} workouts")
    
    return df

def save_parquet(df: pd.DataFrame, output_dir: str):
    """
    Save dataset as Parquet files (one per split).
    
    Parquet format is:
    - Efficient (compressed)
    - Supports complex types (lists)
    - Native HF dataset viewer support
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving Parquet files to {output_dir}...")
    
    # Save full dataset
    full_path = output_path / "dataset.parquet"
    df.to_parquet(full_path, index=False, engine='pyarrow')
    print(f"✓ Saved full dataset: {full_path} ({full_path.stat().st_size / 1e6:.1f} MB)")
    
    # Save splits separately (optional, for easier loading)
    for split in ['train', 'validation', 'test']:
        split_df = df[df['split'] == split]
        split_path = output_path / f"{split}.parquet"
        split_df.to_parquet(split_path, index=False, engine='pyarrow')
        print(f"✓ Saved {split}: {split_path} ({split_path.stat().st_size / 1e6:.1f} MB)")

def create_dataset_card(metadata: Dict, df: pd.DataFrame, output_dir: str):
    """Create README.md for Hugging Face dataset viewer."""
    
    output_path = Path(output_dir) / "README.md"
    
    content = f"""---
license: mit
task_categories:
- time-series-forecasting
tags:
- heart-rate
- running
- physiological-modeling
- lstm
- endomondo
size_categories:
- 10K<n<100K
---

# Endomondo Heart Rate Prediction Dataset V2

## Dataset Summary

This dataset contains **{len(df):,} running workouts** from **{df['user_id'].nunique()}** athletes, designed for heart rate prediction from speed and altitude time-series.

Each workout includes:
- **Time-series**: Heart rate (target), speed, altitude, timestamps
- **Metadata**: Workout type, duration, user ID
- **Statistics**: Pre-computed HR/speed metrics for filtering

## Dataset Structure

### Splits

| Split | Workouts | Description |
|-------|----------|-------------|
| Train | {len(df[df['split'] == 'train']):,} | Training set (70%) |
| Validation | {len(df[df['split'] == 'validation']):,} | Validation set (15%) |
| Test | {len(df[df['split'] == 'test']):,} | Test set (15%) |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `workout_id` | int | Unique workout identifier |
| `user_id` | int | Anonymous user identifier |
| `workout_type` | string | RECOVERY, STEADY, or INTENSIVE |
| `duration_min` | float | Workout duration in minutes |
| `data_points` | int | Number of timesteps (max 500) |
| `heart_rate` | list[float] | Heart rate time-series [BPM] |
| `speed` | list[float] | Speed time-series [km/h] |
| `altitude` | list[float] | Altitude time-series [meters] |
| `timestamp` | list[float] | Unix timestamps [seconds] |
| `hr_mean` | float | Average heart rate [BPM] |
| `hr_std` | float | HR standard deviation |
| `hr_min` | float | Minimum HR [BPM] |
| `hr_max` | float | Maximum HR [BPM] |
| `speed_mean` | float | Average speed [km/h] |
| `speed_max` | float | Maximum speed [km/h] |
| `altitude_gain` | float | Cumulative elevation gain [m] |
| `split` | string | train / validation / test |

### Workout Type Distribution

| Type | Count | Description |
|------|-------|-------------|
| RECOVERY | {len(df[df['workout_type'] == 'RECOVERY']):,} | Easy runs (low intensity) |
| STEADY | {len(df[df['workout_type'] == 'STEADY']):,} | Moderate pace runs |
| INTENSIVE | {len(df[df['workout_type'] == 'INTENSIVE']):,} | High intensity workouts |

## Data Quality

All workouts have been:
1. **Filtered** for quality (removed HR anomalies, corrupted data)
2. **Smoothed** with 7-point moving average (reduces GPS noise)
3. **Validated** against physiological constraints:
   - HR mean ≥ 120 BPM
   - HR max ≤ 200 BPM
   - HR std ≥ 5 BPM
   - Speed-HR correlation ≥ -0.3

Removed: {metadata['removed']['total_removed']:,} low-quality workouts

## Usage Example

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("rricc22/endomondo-hr-prediction-v2")

# Access splits
train_data = dataset['train']
test_data = dataset['test']

# Example workout
workout = train_data[0]
print(f"Workout Type: {{workout['workout_type']}}")
print(f"Duration: {{workout['duration_min']:.1f}} min")
print(f"Avg HR: {{workout['hr_mean']:.1f}} BPM")
print(f"Avg Speed: {{workout['speed_mean']:.1f}} km/h")

# Access time-series
heart_rate = workout['heart_rate']  # List of HR values
speed = workout['speed']            # List of speed values
```

## Model Performance

This dataset was used to train an LSTM model achieving:
- **7.42 BPM** Mean Absolute Error
- **17% improvement** over baseline

See the model card: [rricc22/heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm)

Try the demo: [Heart Rate Predictor](https://huggingface.co/spaces/rricc22/heart-rate-predictor)

## Source

- **Original Data**: Endomondo dataset
- **Processing Pipeline**: Quality filtering → Smoothing → Feature engineering
- **Version**: V2 (January 2026)
- **License**: MIT

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{endomondo_hr_v2,
  title={{Endomondo Heart Rate Prediction Dataset V2}},
  author={{Riccardo}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2}}
}}
```

## Related Resources

- 🤗 **Model**: [heart-rate-prediction-lstm](https://huggingface.co/rricc22/heart-rate-prediction-lstm)
- 🚀 **Demo**: [Interactive Predictor](https://huggingface.co/spaces/rricc22/heart-rate-predictor)
- 📊 **GitHub**: [SUB3_V2 Repository](https://github.com/rricc22/SUB3_V2)
"""
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Created README.md: {output_path}")

def main():
    """Main conversion pipeline."""
    
    # Paths
    input_json = "Preprocessing/clean_dataset_v2.json"
    output_dir = "DATA/huggingface_dataset"
    
    print("=" * 60)
    print("Converting Dataset to Hugging Face Format")
    print("=" * 60)
    
    # Load data
    data = load_dataset(input_json)
    
    # Convert to tabular
    df = convert_to_tabular(data)
    
    # Add splits
    df = add_splits(df)
    
    # Save Parquet files
    save_parquet(df, output_dir)
    
    # Create README
    create_dataset_card(data['metadata'], df, output_dir)
    
    print("\n" + "=" * 60)
    print("✓ Conversion Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. cd {output_dir}")
    print(f"2. Upload to Hugging Face:")
    print(f"   huggingface-cli upload rricc22/endomondo-hr-prediction-v2 . --repo-type=dataset")
    print(f"\nThe dataset will be viewable at:")
    print(f"https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2")

if __name__ == "__main__":
    main()
