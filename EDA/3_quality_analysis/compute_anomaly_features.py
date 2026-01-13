#!/usr/bin/env python3
"""
Compute Anomaly Features for Workout Quality Assessment.

Extracts statistical features to detect physiologically weird patterns:
- Inverse relationships (speed down, HR up)
- Stable speed with rising HR
- Low/negative correlations
- Chaotic segment patterns

Usage:
    # Single workout
    python3 compute_anomaly_features.py --line 31
    
    # Multiple workouts
    python3 compute_anomaly_features.py --lines 31,33,36,37
    
    # All good workouts (from preprocessing)
    python3 compute_anomaly_features.py --all-good --output features.jsonl

Author: OpenCode
Date: 2026-01-13
"""

import ast
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
import sys

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA = PROJECT_ROOT / "DATA" / "raw" / "endomondoHR_proper-002.json"
COMPUTED_SPEED_FILE = PROJECT_ROOT / "DATA" / "processed" / "running_computed_speed.jsonl"
STAGE1_OUTPUT = PROJECT_ROOT / "Preprocessing" / "stage1_full_output.json"
STAGE2_OUTPUT = PROJECT_ROOT / "Preprocessing" / "stage2_output.json"


# =============================================================================
# Data Loading
# =============================================================================

def load_computed_speeds() -> Dict[int, np.ndarray]:
    """Load computed speeds into lookup dict."""
    speed_lookup = {}
    if not COMPUTED_SPEED_FILE.exists():
        print("WARNING: No computed speed file found!")
        return speed_lookup
    
    with open(COMPUTED_SPEED_FILE, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            speed_lookup[data['line_num']] = np.array(data['speed'])
    
    return speed_lookup


def load_workout(line_number: int, speed_lookup: Dict[int, np.ndarray]) -> Optional[Dict]:
    """Load workout data from raw file."""
    try:
        with open(RAW_DATA, 'r') as f:
            for i, line in enumerate(f, start=1):
                if i == line_number:
                    if not line.strip():
                        return None
                    workout = ast.literal_eval(line.strip())
                    
                    # Add computed speed if missing
                    if 'speed' not in workout or len(workout.get('speed', [])) == 0:
                        if line_number in speed_lookup:
                            workout['speed'] = speed_lookup[line_number].tolist()
                    
                    return workout
        return None
    except Exception as e:
        print(f"Error loading workout {line_number}: {e}")
        return None


def load_good_workout_lines() -> List[int]:
    """Load all good workout line numbers from preprocessing outputs."""
    lines = []
    
    # Stage 1 PASS
    with open(STAGE1_OUTPUT, 'r') as f:
        stage1 = json.load(f)
        for w in stage1['workouts']:
            if w['decision'] == 'PASS':
                lines.append(w['line_number'])
    
    # Stage 2
    with open(STAGE2_OUTPUT, 'r') as f:
        stage2 = json.load(f)
        for w in stage2['auto_passed']:
            lines.append(w['line_number'])
        for w in stage2['llm_validated']:
            if w.get('decision') in ['KEEP', 'FIX']:
                lines.append(w['line_number'])
    
    return sorted(lines)


# =============================================================================
# Feature Computation
# =============================================================================

def compute_anomaly_features(
    hr: np.ndarray, 
    speed: np.ndarray, 
    altitude: np.ndarray,
    timestamps: np.ndarray
) -> Dict:
    """
    Detect physiologically weird patterns.
    
    Args:
        hr: Heart rate array (BPM)
        speed: Speed array (km/h)
        altitude: Altitude array (meters)
        timestamps: Unix timestamps
    
    Returns:
        Dictionary with anomaly features and scores
    """
    features = {
        'hr_mean': float(np.mean(hr)),
        'hr_std': float(np.std(hr)),
        'speed_mean': float(np.mean(speed)),
        'speed_std': float(np.std(speed)),
        'altitude_std': float(np.std(altitude)),
        'duration_min': float((timestamps[-1] - timestamps[0]) / 60),
    }
    
    # Guard against edge cases
    if len(hr) < 10 or len(speed) < 10:
        features['error'] = 'Too short'
        return features
    
    if np.std(speed) < 0.1:
        features['error'] = 'Speed constant (no variance)'
        return features
    
    # 1. Overall correlation (should be positive for running)
    try:
        features['hr_speed_pearson'] = float(np.corrcoef(hr, speed)[0, 1])
        features['hr_speed_spearman'] = float(spearmanr(hr, speed)[0])
    except:
        features['hr_speed_pearson'] = 0.0
        features['hr_speed_spearman'] = 0.0
    
    # 2. Linear trends (detect inverse relationships)
    hr_trend = np.polyfit(range(len(hr)), hr, 1)[0]  # Slope
    speed_trend = np.polyfit(range(len(speed)), speed, 1)[0]
    
    features['hr_trend'] = float(hr_trend)  # BPM per timestep
    features['speed_trend'] = float(speed_trend)  # km/h per timestep
    
    # Type 2: Speed drops, HR rises (inverse trend)
    features['inverse_trend'] = bool((speed_trend < -0.05) and (hr_trend > 0.1))
    
    # 3. Type 1: Stable speed, rising HR
    speed_is_stable = np.std(speed) < 1.5  # Speed variance < 1.5 km/h
    hr_is_rising = hr_trend > 0.3  # HR rising > 0.3 BPM/timestep
    altitude_is_flat = np.std(altitude) < 10.0  # < 10m variation
    
    features['speed_stable'] = bool(speed_is_stable)
    features['hr_rising'] = bool(hr_is_rising)
    features['altitude_flat'] = bool(altitude_is_flat)
    features['stable_speed_rising_hr'] = bool(speed_is_stable and hr_is_rising and altitude_is_flat)
    
    # 4. Segment-based correlation (detect chaotic patterns)
    segment_corrs = []
    segment_size = min(60, len(hr) // 5)  # ~10 min segments or 1/5 of workout
    
    if segment_size >= 10:
        for i in range(0, len(hr) - segment_size, segment_size):
            hr_seg = hr[i:i+segment_size]
            speed_seg = speed[i:i+segment_size]
            
            if np.std(speed_seg) > 0.5:  # Only if speed varies enough
                try:
                    corr = np.corrcoef(hr_seg, speed_seg)[0, 1]
                    if not np.isnan(corr):
                        segment_corrs.append(corr)
                except:
                    pass
    
    if segment_corrs:
        features['mean_segment_corr'] = float(np.mean(segment_corrs))
        features['min_segment_corr'] = float(np.min(segment_corrs))
        features['max_segment_corr'] = float(np.max(segment_corrs))
        features['negative_segments'] = int(sum(c < 0 for c in segment_corrs))
        features['low_corr_segments'] = int(sum(c < 0.15 for c in segment_corrs))
        features['num_segments'] = len(segment_corrs)
    else:
        features['mean_segment_corr'] = 0.0
        features['min_segment_corr'] = 0.0
        features['max_segment_corr'] = 0.0
        features['negative_segments'] = 0
        features['low_corr_segments'] = 0
        features['num_segments'] = 0
    
    # 5. Coefficient of variation (detect noisy data)
    features['hr_cv'] = float(np.std(hr) / np.mean(hr))
    features['speed_cv'] = float(np.std(speed) / np.mean(speed))
    
    # 6. HR response quality (does HR follow speed changes?)
    speed_changes = np.abs(np.diff(speed))
    hr_changes = np.abs(np.diff(hr))
    
    if np.sum(speed_changes > 1.0) > 5:  # If there are speed changes
        # Check if HR responds
        speed_change_idx = np.where(speed_changes > 1.0)[0]
        hr_response = np.mean([hr_changes[min(i+1, len(hr_changes)-1)] for i in speed_change_idx])
        features['hr_response_to_speed'] = float(hr_response)
    else:
        features['hr_response_to_speed'] = 0.0
    
    return features


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute anomaly features for workouts')
    parser.add_argument('--line', type=int, help='Single workout line number')
    parser.add_argument('--lines', type=str, help='Comma-separated line numbers (e.g., 31,33,36)')
    parser.add_argument('--all-good', action='store_true', help='Process all good workouts')
    parser.add_argument('--output', type=str, help='Output JSONL file (for --all-good)')
    parser.add_argument('--limit', type=int, help='Limit number of workouts (for testing)')
    
    args = parser.parse_args()
    
    # Determine which workouts to process
    if args.line:
        line_numbers = [args.line]
    elif args.lines:
        line_numbers = [int(x.strip()) for x in args.lines.split(',')]
    elif args.all_good:
        print("Loading good workout list...")
        line_numbers = load_good_workout_lines()
        print(f"Found {len(line_numbers):,} good workouts")
        if args.limit:
            line_numbers = line_numbers[:args.limit]
            print(f"Limited to {args.limit:,} workouts")
    else:
        parser.error("Must specify --line, --lines, or --all-good")
    
    # Load computed speeds
    print("Loading computed speeds...")
    speed_lookup = load_computed_speeds()
    print(f"Loaded {len(speed_lookup):,} computed speeds\n")
    
    # Process workouts
    results = []
    
    for i, line_num in enumerate(line_numbers, 1):
        if i % 1000 == 0:
            print(f"Processing {i:,}/{len(line_numbers):,}...")
        
        workout = load_workout(line_num, speed_lookup)
        if workout is None:
            print(f"Could not load workout {line_num}")
            continue
        
        # Extract arrays
        hr = np.array(workout.get('heart_rate', []))
        speed = np.array(workout.get('speed', []))
        altitude = np.array(workout.get('altitude', []))
        timestamps = np.array(workout.get('timestamp', []))
        
        # Check validity
        if len(hr) == 0 or len(speed) == 0 or len(timestamps) == 0:
            print(f"Workout {line_num}: Missing data")
            continue
        
        # Compute features
        features = compute_anomaly_features(hr, speed, altitude, timestamps)
        features['line_num'] = line_num
        features['workout_id'] = workout.get('id', '')
        
        results.append(features)
        
        # Print for single/few workouts
        if len(line_numbers) <= 10:
            print(f"\n{'='*60}")
            print(f"Workout {line_num} (ID: {workout.get('id', 'N/A')})")
            print(f"{'='*60}")
            print(f"Duration: {features['duration_min']:.1f} min")
            print(f"HR: {features['hr_mean']:.1f} ± {features['hr_std']:.1f} BPM")
            print(f"Speed: {features['speed_mean']:.1f} ± {features['speed_std']:.1f} km/h")
            print(f"\nCorrelations:")
            print(f"  Pearson: {features['hr_speed_pearson']:.3f}")
            print(f"  Spearman: {features['hr_speed_spearman']:.3f}")
            print(f"  Mean segment: {features['mean_segment_corr']:.3f}")
            print(f"\nTrends:")
            print(f"  HR trend: {features['hr_trend']:.4f} BPM/timestep")
            print(f"  Speed trend: {features['speed_trend']:.4f} km/h/timestep")
            print(f"\nAnomalies:")
            print(f"  Inverse trend: {features['inverse_trend']}")
            print(f"  Stable speed + rising HR: {features['stable_speed_rising_hr']}")
            print(f"  Negative segments: {features['negative_segments']}/{features['num_segments']}")
    
    # Save results
    if args.output and results:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"\n✓ Saved {len(results):,} results to {output_path}")
    elif not args.output and args.all_good:
        print("\nWARNING: Use --output to save results when processing all workouts")
    
    # Summary statistics
    if len(results) > 10:
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        correlations = [r['hr_speed_pearson'] for r in results if 'hr_speed_pearson' in r]
        inverse_trends = sum(r.get('inverse_trend', False) for r in results)
        stable_rising = sum(r.get('stable_speed_rising_hr', False) for r in results)
        
        print(f"\nTotal workouts: {len(results):,}")
        print(f"\nCorrelation (Pearson):")
        print(f"  Mean: {np.mean(correlations):.3f}")
        print(f"  Median: {np.median(correlations):.3f}")
        print(f"  Std: {np.std(correlations):.3f}")
        print(f"  Min: {np.min(correlations):.3f}")
        print(f"  Max: {np.max(correlations):.3f}")
        
        print(f"\nAnomalies detected:")
        print(f"  Inverse trends: {inverse_trends:,} ({inverse_trends/len(results)*100:.1f}%)")
        print(f"  Stable speed + rising HR: {stable_rising:,} ({stable_rising/len(results)*100:.1f}%)")
        print(f"  Low correlation (<0.15): {sum(c < 0.15 for c in correlations):,} ({sum(c < 0.15 for c in correlations)/len(correlations)*100:.1f}%)")
        print(f"  Negative correlation: {sum(c < 0 for c in correlations):,} ({sum(c < 0 for c in correlations)/len(correlations)*100:.1f}%)")


if __name__ == "__main__":
    main()
