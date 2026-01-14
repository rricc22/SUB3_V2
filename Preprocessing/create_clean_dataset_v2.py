#!/usr/bin/env python3
"""
Create clean dataset v2:
1. Remove flagged samples (low HR, high HR, flat HR, negative correlation)
2. Drop INTERVALS and UNKNOWN workout types
3. Add one-hot workout_type encoding (RECOVERY, STEADY, INTENSIVE)
"""

import json
import numpy as np
from datetime import datetime

INPUT_FILE = 'Preprocessing/clean_dataset_smoothed.json'
OUTPUT_FILE = 'Preprocessing/clean_dataset_v2.json'

def compute_hr_speed_correlation(hr, speed):
    """Compute Pearson correlation between HR and speed."""
    hr = np.array(hr, dtype=float)
    speed = np.array(speed, dtype=float)
    mask = ~(np.isnan(hr) | np.isnan(speed))
    if mask.sum() > 10:
        return np.corrcoef(hr[mask], speed[mask])[0, 1]
    return 0

def check_flags(workout):
    """Check if workout has any data quality flags."""
    hr = np.array(workout['heart_rate'], dtype=float)
    speed = np.array(workout['speed'], dtype=float)

    hr_mean = np.nanmean(hr)
    hr_max = np.nanmax(hr)
    hr_std = np.nanstd(hr)
    corr = compute_hr_speed_correlation(hr, speed)

    flags = []
    if hr_mean < 120:
        flags.append('low_hr_mean')
    if hr_max > 200:
        flags.append('high_hr_max')
    if hr_std < 5:
        flags.append('flat_hr')
    if corr < -0.3:
        flags.append('negative_correlation')

    return flags

def main():
    print("Loading dataset...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    original_count = len(data['workouts'])
    print(f"Original workouts: {original_count:,}")

    # Track removal reasons
    removed_flags = 0
    removed_intervals = 0
    removed_unknown = 0

    clean_workouts = []

    # Type counts for new dataset
    type_counts = {'RECOVERY': 0, 'STEADY': 0, 'INTENSIVE': 0}

    print("Processing workouts...")
    for i, workout in enumerate(data['workouts']):
        if i % 5000 == 0:
            print(f"  Processing {i:,}/{original_count:,}...")

        workout_type = workout.get('workout_type', 'UNKNOWN')

        # Skip INTERVALS and UNKNOWN
        if workout_type == 'INTERVALS':
            removed_intervals += 1
            continue
        if workout_type == 'UNKNOWN':
            removed_unknown += 1
            continue

        # Check flags
        flags = check_flags(workout)
        if flags:
            removed_flags += 1
            continue

        # Add one-hot encoding
        workout['workout_type_onehot'] = {
            'RECOVERY': 1 if workout_type == 'RECOVERY' else 0,
            'STEADY': 1 if workout_type == 'STEADY' else 0,
            'INTENSIVE': 1 if workout_type == 'INTENSIVE' else 0
        }

        clean_workouts.append(workout)
        type_counts[workout_type] += 1

    # Update metadata
    new_metadata = {
        'timestamp': datetime.now().isoformat(),
        'source_file': INPUT_FILE,
        'original_count': original_count,
        'final_count': len(clean_workouts),
        'removed': {
            'flagged_samples': removed_flags,
            'intervals': removed_intervals,
            'unknown': removed_unknown,
            'total_removed': removed_flags + removed_intervals + removed_unknown
        },
        'workout_type_counts': type_counts,
        'flags_applied': [
            'hr_mean < 120 BPM',
            'hr_max > 200 BPM',
            'hr_std < 5 BPM',
            'hr_speed_correlation < -0.3'
        ],
        'one_hot_encoding': ['RECOVERY', 'STEADY', 'INTENSIVE'],
        'smoothing_inherited': data['metadata'].get('smoothing_applied', {})
    }

    output_data = {
        'metadata': new_metadata,
        'workouts': clean_workouts
    }

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original workouts: {original_count:,}")
    print(f"Removed (flagged): {removed_flags:,}")
    print(f"Removed (INTERVALS): {removed_intervals:,}")
    print(f"Removed (UNKNOWN): {removed_unknown:,}")
    print(f"Final clean workouts: {len(clean_workouts):,}")
    print(f"\nWorkout type distribution:")
    for wtype, count in type_counts.items():
        pct = count / len(clean_workouts) * 100
        print(f"  {wtype}: {count:,} ({pct:.1f}%)")
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
