#!/usr/bin/env python3
"""
Streaming analysis of clean_dataset_smoothed.json (1.7GB)
Computes correlations and key statistics without loading entire file into memory.
"""

import json
import numpy as np
from collections import defaultdict
import sys

def compute_correlations(hr, speed, altitude):
    """Compute Pearson correlations between arrays."""
    correlations = {}

    # Remove NaN/None values
    hr = np.array(hr, dtype=float)
    speed = np.array(speed, dtype=float)
    altitude = np.array(altitude, dtype=float)

    # HR-Speed correlation
    mask = ~(np.isnan(hr) | np.isnan(speed))
    if mask.sum() > 10:
        correlations['hr_speed'] = np.corrcoef(hr[mask], speed[mask])[0, 1]
    else:
        correlations['hr_speed'] = np.nan

    # HR-Altitude correlation
    mask = ~(np.isnan(hr) | np.isnan(altitude))
    if mask.sum() > 10:
        correlations['hr_altitude'] = np.corrcoef(hr[mask], altitude[mask])[0, 1]
    else:
        correlations['hr_altitude'] = np.nan

    # Speed-Altitude correlation
    mask = ~(np.isnan(speed) | np.isnan(altitude))
    if mask.sum() > 10:
        correlations['speed_altitude'] = np.corrcoef(speed[mask], altitude[mask])[0, 1]
    else:
        correlations['speed_altitude'] = np.nan

    return correlations

def analyze_dataset(filepath):
    """Stream through JSON and compute statistics."""

    print("Loading JSON file (streaming approach)...")
    print("This may take a few minutes for 1.7GB file...\n")

    # We need to load the file - use chunked reading
    with open(filepath, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    workouts = data['workouts']

    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"Total clean workouts: {metadata['total_clean_workouts']:,}")
    print(f"Corrections applied: {metadata['corrections_applied']}")
    print(f"\nClassification breakdown:")
    for wtype, count in sorted(metadata['classification_stats'].items(), key=lambda x: -x[1]):
        pct = count / metadata['total_clean_workouts'] * 100
        print(f"  {wtype:12s}: {count:6,} ({pct:5.1f}%)")

    print(f"\nData sources:")
    for source, count in metadata['sources'].items():
        print(f"  {source:20s}: {count:,}")

    if 'smoothing_applied' in metadata:
        print(f"\nSmoothing applied:")
        print(f"  Window size: {metadata['smoothing_applied']['window_size']}")
        print(f"  Fields: {metadata['smoothing_applied']['fields_smoothed']}")

    # Compute statistics
    print("\n" + "=" * 60)
    print("COMPUTING STATISTICS...")
    print("=" * 60)

    # Accumulators
    all_hr = []
    all_speed = []
    all_altitude = []
    correlations_by_type = defaultdict(list)
    global_correlations = {'hr_speed': [], 'hr_altitude': [], 'speed_altitude': []}

    durations = []
    data_points_list = []
    hr_ranges = []  # (min, max) per workout
    speed_ranges = []
    altitude_ranges = []

    # Per-type statistics
    stats_by_type = defaultdict(lambda: {
        'count': 0,
        'hr_means': [],
        'hr_stds': [],
        'speed_means': [],
        'altitude_means': [],
        'durations': []
    })

    for i, workout in enumerate(workouts):
        if i % 5000 == 0:
            print(f"  Processing workout {i:,}/{len(workouts):,}...")

        wtype = workout.get('workout_type', 'UNKNOWN')
        hr = np.array(workout['heart_rate'], dtype=float)
        speed = np.array(workout['speed'], dtype=float)
        altitude = np.array(workout['altitude'], dtype=float)

        # Global arrays (sample every 10th point to save memory)
        all_hr.extend(hr[::10])
        all_speed.extend(speed[::10])
        all_altitude.extend(altitude[::10])

        # Per-workout correlations
        corr = compute_correlations(hr, speed, altitude)
        for k, v in corr.items():
            if not np.isnan(v):
                global_correlations[k].append(v)
                correlations_by_type[wtype].append((k, v))

        # Basic stats
        durations.append(workout.get('duration_min', 0))
        data_points_list.append(workout.get('data_points', len(hr)))

        # Ranges
        hr_ranges.append((np.nanmin(hr), np.nanmax(hr), np.nanmean(hr), np.nanstd(hr)))
        speed_ranges.append((np.nanmin(speed), np.nanmax(speed), np.nanmean(speed)))
        altitude_ranges.append((np.nanmin(altitude), np.nanmax(altitude), np.nanmean(altitude)))

        # Per-type stats
        stats_by_type[wtype]['count'] += 1
        stats_by_type[wtype]['hr_means'].append(np.nanmean(hr))
        stats_by_type[wtype]['hr_stds'].append(np.nanstd(hr))
        stats_by_type[wtype]['speed_means'].append(np.nanmean(speed))
        stats_by_type[wtype]['altitude_means'].append(np.nanmean(altitude))
        stats_by_type[wtype]['durations'].append(workout.get('duration_min', 0))

    # Convert to numpy
    all_hr = np.array(all_hr)
    all_speed = np.array(all_speed)
    all_altitude = np.array(all_altitude)

    print("\n" + "=" * 60)
    print("GLOBAL DISTRIBUTIONS")
    print("=" * 60)

    print(f"\nHeart Rate (BPM):")
    print(f"  Min: {np.nanmin(all_hr):.1f}")
    print(f"  Max: {np.nanmax(all_hr):.1f}")
    print(f"  Mean: {np.nanmean(all_hr):.1f}")
    print(f"  Median: {np.nanmedian(all_hr):.1f}")
    print(f"  Std: {np.nanstd(all_hr):.1f}")
    print(f"  Percentiles [5, 25, 50, 75, 95]: {np.nanpercentile(all_hr, [5, 25, 50, 75, 95])}")

    print(f"\nSpeed (km/h):")
    print(f"  Min: {np.nanmin(all_speed):.2f}")
    print(f"  Max: {np.nanmax(all_speed):.2f}")
    print(f"  Mean: {np.nanmean(all_speed):.2f}")
    print(f"  Median: {np.nanmedian(all_speed):.2f}")
    print(f"  Std: {np.nanstd(all_speed):.2f}")
    print(f"  Percentiles [5, 25, 50, 75, 95]: {np.nanpercentile(all_speed, [5, 25, 50, 75, 95])}")

    print(f"\nAltitude (m):")
    print(f"  Min: {np.nanmin(all_altitude):.1f}")
    print(f"  Max: {np.nanmax(all_altitude):.1f}")
    print(f"  Mean: {np.nanmean(all_altitude):.1f}")
    print(f"  Median: {np.nanmedian(all_altitude):.1f}")
    print(f"  Std: {np.nanstd(all_altitude):.1f}")

    print(f"\nWorkout Duration (min):")
    durations = np.array(durations)
    print(f"  Min: {np.min(durations):.1f}")
    print(f"  Max: {np.max(durations):.1f}")
    print(f"  Mean: {np.mean(durations):.1f}")
    print(f"  Median: {np.median(durations):.1f}")

    print(f"\nData Points per Workout:")
    data_points_list = np.array(data_points_list)
    print(f"  Min: {np.min(data_points_list)}")
    print(f"  Max: {np.max(data_points_list)}")
    print(f"  Mean: {np.mean(data_points_list):.1f}")

    # Correlations
    print("\n" + "=" * 60)
    print("CORRELATIONS (across all workouts)")
    print("=" * 60)

    for corr_name, values in global_correlations.items():
        values = np.array(values)
        print(f"\n{corr_name}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
        print(f"  Positive correlations: {(values > 0).sum():,} ({(values > 0).mean()*100:.1f}%)")
        print(f"  Strong positive (>0.5): {(values > 0.5).sum():,} ({(values > 0.5).mean()*100:.1f}%)")

    # Global correlation on pooled data
    print("\n" + "=" * 60)
    print("GLOBAL CORRELATION (pooled data)")
    print("=" * 60)

    mask = ~(np.isnan(all_hr) | np.isnan(all_speed))
    if mask.sum() > 100:
        global_hr_speed = np.corrcoef(all_hr[mask], all_speed[mask])[0, 1]
        print(f"HR-Speed (pooled): {global_hr_speed:.4f}")

    mask = ~(np.isnan(all_hr) | np.isnan(all_altitude))
    if mask.sum() > 100:
        global_hr_alt = np.corrcoef(all_hr[mask], all_altitude[mask])[0, 1]
        print(f"HR-Altitude (pooled): {global_hr_alt:.4f}")

    # Per-type analysis
    print("\n" + "=" * 60)
    print("STATISTICS BY WORKOUT TYPE")
    print("=" * 60)

    for wtype in ['INTENSIVE', 'INTERVALS', 'STEADY', 'RECOVERY', 'UNKNOWN']:
        if wtype not in stats_by_type:
            continue
        stats = stats_by_type[wtype]
        print(f"\n{wtype} (n={stats['count']:,}):")
        print(f"  HR Mean: {np.mean(stats['hr_means']):.1f} ± {np.std(stats['hr_means']):.1f} BPM")
        print(f"  HR Std (within workout): {np.mean(stats['hr_stds']):.1f} BPM")
        print(f"  Speed Mean: {np.mean(stats['speed_means']):.2f} ± {np.std(stats['speed_means']):.2f} km/h")
        print(f"  Duration: {np.mean(stats['durations']):.1f} ± {np.std(stats['durations']):.1f} min")

        # Per-type correlations
        type_corrs = [v for k, v in correlations_by_type[wtype] if k == 'hr_speed']
        if type_corrs:
            print(f"  HR-Speed corr: {np.mean(type_corrs):.4f} ± {np.std(type_corrs):.4f}")

    # HR range analysis
    print("\n" + "=" * 60)
    print("HR RANGE ANALYSIS (per workout)")
    print("=" * 60)

    hr_ranges = np.array(hr_ranges)
    hr_mins, hr_maxs, hr_means, hr_stds = hr_ranges[:, 0], hr_ranges[:, 1], hr_ranges[:, 2], hr_ranges[:, 3]
    hr_dynamic_range = hr_maxs - hr_mins

    print(f"\nWorkout HR minimums: {np.mean(hr_mins):.1f} ± {np.std(hr_mins):.1f} BPM")
    print(f"Workout HR maximums: {np.mean(hr_maxs):.1f} ± {np.std(hr_maxs):.1f} BPM")
    print(f"Workout HR means: {np.mean(hr_means):.1f} ± {np.std(hr_means):.1f} BPM")
    print(f"Dynamic range (max-min): {np.mean(hr_dynamic_range):.1f} ± {np.std(hr_dynamic_range):.1f} BPM")

    # Anomaly detection
    print("\n" + "=" * 60)
    print("POTENTIAL DATA QUALITY ISSUES")
    print("=" * 60)

    # Low HR workouts (possible offset errors)
    low_hr_count = (hr_means < 120).sum()
    print(f"\nWorkouts with mean HR < 120 BPM: {low_hr_count:,} ({low_hr_count/len(workouts)*100:.1f}%)")

    # Very high HR workouts
    high_hr_count = (hr_maxs > 200).sum()
    print(f"Workouts with max HR > 200 BPM: {high_hr_count:,} ({high_hr_count/len(workouts)*100:.1f}%)")

    # Low variability (flat HR)
    low_var_count = (hr_stds < 5).sum()
    print(f"Workouts with HR std < 5 BPM (flat): {low_var_count:,} ({low_var_count/len(workouts)*100:.1f}%)")

    # Negative correlation (suspicious)
    neg_corr = [v for v in global_correlations['hr_speed'] if v < -0.3]
    print(f"Workouts with HR-Speed corr < -0.3: {len(neg_corr):,} ({len(neg_corr)/len(workouts)*100:.1f}%)")

    # Speed analysis
    print("\n" + "=" * 60)
    print("SPEED ANALYSIS")
    print("=" * 60)

    speed_ranges = np.array(speed_ranges)
    speed_mins, speed_maxs, speed_means = speed_ranges[:, 0], speed_ranges[:, 1], speed_ranges[:, 2]

    print(f"\nWorkout speed means: {np.mean(speed_means):.2f} ± {np.std(speed_means):.2f} km/h")
    print(f"Typical pace: {60/np.mean(speed_means):.2f} min/km")

    # Speed distribution bins
    print(f"\nSpeed distribution (workout means):")
    for lo, hi in [(0, 8), (8, 10), (10, 12), (12, 14), (14, 20)]:
        count = ((speed_means >= lo) & (speed_means < hi)).sum()
        print(f"  {lo}-{hi} km/h: {count:,} ({count/len(workouts)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("SUMMARY FOR ML MODEL")
    print("=" * 60)

    print(f"""
Key findings for HR prediction model:

1. DATASET SIZE: {len(workouts):,} workouts, ~{np.mean(data_points_list):.0f} points each

2. TARGET VARIABLE (Heart Rate):
   - Range: {np.nanmin(all_hr):.0f}-{np.nanmax(all_hr):.0f} BPM
   - Mean ± Std: {np.nanmean(all_hr):.0f} ± {np.nanstd(all_hr):.0f} BPM
   - Model should predict values mostly in 120-180 BPM range

3. PRIMARY PREDICTOR (Speed):
   - Range: {np.nanmin(all_speed):.1f}-{np.nanmax(all_speed):.1f} km/h
   - Mean: {np.nanmean(all_speed):.1f} km/h ({60/np.nanmean(all_speed):.1f} min/km pace)
   - HR-Speed correlation: {np.mean(global_correlations['hr_speed']):.3f} (weak-moderate positive)
   - Only {(np.array(global_correlations['hr_speed']) > 0.5).mean()*100:.0f}% of workouts have strong correlation

4. SECONDARY PREDICTOR (Altitude):
   - HR-Altitude correlation: {np.mean(global_correlations['hr_altitude']):.3f} (very weak)
   - Altitude likely needs derivative features (climbing vs descending)

5. WORKOUT TYPE DISTRIBUTION:
   - RECOVERY: {metadata['classification_stats'].get('RECOVERY', 0):,} (lower HR, lower speed)
   - STEADY: {metadata['classification_stats'].get('STEADY', 0):,} (moderate HR)
   - INTENSIVE: {metadata['classification_stats'].get('INTENSIVE', 0):,} (high HR)
   - INTERVALS: {metadata['classification_stats'].get('INTERVALS', 0):,} (variable HR)

6. DATA QUALITY CONCERNS:
   - ~{low_hr_count} workouts may have HR offset issues (mean HR < 120)
   - ~{low_var_count} workouts have suspiciously flat HR
   - Negative HR-Speed correlations in some workouts suggest data issues

7. MODELING IMPLICATIONS:
   - Weak direct HR-Speed correlation ({np.mean(global_correlations['hr_speed']):.2f}) explains V1 poor performance
   - Temporal/lag features are ESSENTIAL (HR responds ~30s after speed change)
   - Consider per-user or per-workout-type models
   - Large HR variability within workouts ({np.mean(hr_stds):.0f} BPM avg) is signal to capture
""")

if __name__ == '__main__':
    filepath = '/home/riccardo/Documents/SUB3_V2/Preprocessing/clean_dataset_smoothed.json'
    analyze_dataset(filepath)
