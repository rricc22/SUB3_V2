#!/usr/bin/env python3
"""
Analyze Manually Checked Patterns from Gallery.

Takes exported checked workouts from gallery and compares their features
to unchecked workouts to learn discriminative thresholds.

This implements your 10% annotation strategy:
1. You manually check ~4,600 workouts in gallery for weird patterns
2. Export checked list as JSON
3. This script computes features for checked vs unchecked
4. Suggests thresholds for auto-flagging remaining 90%

Usage:
    python3 analyze_checked_patterns.py --input checked_workouts_2026-01-13.json

Author: OpenCode
Date: 2026-01-13
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from compute_anomaly_features import (
    load_computed_speeds,
    load_workout,
    compute_anomaly_features,
    load_good_workout_lines
)

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "EDA" / "outputs"


# =============================================================================
# Analysis
# =============================================================================

def load_checked_workouts(json_path: Path) -> Set[int]:
    """Load checked workout line numbers from gallery export."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Gallery exports as: {'checked_line_numbers': [31, 33, ...], 'total': N, ...}
    return set(data['checked_line_numbers'])


def analyze_patterns(
    checked_lines: Set[int],
    sample_lines: List[int],
    speed_lookup: Dict
) -> tuple:
    """
    Compute features for checked (bad) vs unchecked (good) workouts.
    
    Returns:
        (bad_features, good_features) - lists of feature dicts
    """
    bad_features = []
    good_features = []
    
    print(f"\nProcessing {len(sample_lines):,} workouts...")
    print(f"  Checked (bad): {len(checked_lines):,}")
    print(f"  Unchecked (good): {len(sample_lines) - len(checked_lines):,}\n")
    
    for i, line_num in enumerate(sample_lines, 1):
        if i % 500 == 0:
            print(f"  Progress: {i:,}/{len(sample_lines):,}")
        
        workout = load_workout(line_num, speed_lookup)
        if workout is None:
            continue
        
        # Extract arrays
        hr = np.array(workout.get('heart_rate', []))
        speed = np.array(workout.get('speed', []))
        altitude = np.array(workout.get('altitude', []))
        timestamps = np.array(workout.get('timestamp', []))
        
        if len(hr) == 0 or len(speed) == 0:
            continue
        
        # Compute features
        features = compute_anomaly_features(hr, speed, altitude, timestamps)
        features['line_num'] = line_num
        
        # Categorize
        if line_num in checked_lines:
            bad_features.append(features)
        else:
            good_features.append(features)
    
    return bad_features, good_features


def print_statistics(bad_features: List[Dict], good_features: List[Dict]):
    """Print comparative statistics."""
    
    print(f"\n{'='*70}")
    print("COMPARATIVE STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nSample sizes:")
    print(f"  Bad (checked): {len(bad_features):,}")
    print(f"  Good (unchecked): {len(good_features):,}")
    
    # Helper to extract feature values
    def get_values(features, key):
        return [f[key] for f in features if key in f and not isinstance(f[key], bool)]
    
    # Correlation analysis
    print(f"\n{'─'*70}")
    print("HR-SPEED CORRELATION (Pearson)")
    print(f"{'─'*70}")
    
    bad_corr = get_values(bad_features, 'hr_speed_pearson')
    good_corr = get_values(good_features, 'hr_speed_pearson')
    
    print(f"\nBAD workouts (checked):")
    print(f"  Mean: {np.mean(bad_corr):.3f}")
    print(f"  Median: {np.median(bad_corr):.3f}")
    print(f"  Std: {np.std(bad_corr):.3f}")
    print(f"  Min/Max: {np.min(bad_corr):.3f} / {np.max(bad_corr):.3f}")
    print(f"  < 0.15: {sum(c < 0.15 for c in bad_corr)} ({sum(c < 0.15 for c in bad_corr)/len(bad_corr)*100:.1f}%)")
    print(f"  < 0: {sum(c < 0 for c in bad_corr)} ({sum(c < 0 for c in bad_corr)/len(bad_corr)*100:.1f}%)")
    
    print(f"\nGOOD workouts (unchecked):")
    print(f"  Mean: {np.mean(good_corr):.3f}")
    print(f"  Median: {np.median(good_corr):.3f}")
    print(f"  Std: {np.std(good_corr):.3f}")
    print(f"  Min/Max: {np.min(good_corr):.3f} / {np.max(good_corr):.3f}")
    print(f"  < 0.15: {sum(c < 0.15 for c in good_corr)} ({sum(c < 0.15 for c in good_corr)/len(good_corr)*100:.1f}%)")
    print(f"  < 0: {sum(c < 0 for c in good_corr)} ({sum(c < 0 for c in good_corr)/len(good_corr)*100:.1f}%)")
    
    # Anomaly flags
    print(f"\n{'─'*70}")
    print("ANOMALY FLAGS")
    print(f"{'─'*70}")
    
    bad_inverse = sum(f.get('inverse_trend', False) for f in bad_features)
    good_inverse = sum(f.get('inverse_trend', False) for f in good_features)
    
    bad_stable_rising = sum(f.get('stable_speed_rising_hr', False) for f in bad_features)
    good_stable_rising = sum(f.get('stable_speed_rising_hr', False) for f in good_features)
    
    print(f"\nInverse trend (speed↓, HR↑):")
    print(f"  BAD: {bad_inverse}/{len(bad_features)} ({bad_inverse/len(bad_features)*100:.1f}%)")
    print(f"  GOOD: {good_inverse}/{len(good_features)} ({good_inverse/len(good_features)*100:.1f}%)")
    
    print(f"\nStable speed + rising HR:")
    print(f"  BAD: {bad_stable_rising}/{len(bad_features)} ({bad_stable_rising/len(bad_features)*100:.1f}%)")
    print(f"  GOOD: {good_stable_rising}/{len(good_features)} ({good_stable_rising/len(good_features)*100:.1f}%)")
    
    # Segment correlation
    print(f"\n{'─'*70}")
    print("SEGMENT CORRELATION")
    print(f"{'─'*70}")
    
    bad_seg = get_values(bad_features, 'mean_segment_corr')
    good_seg = get_values(good_features, 'mean_segment_corr')
    
    print(f"\nMean segment correlation:")
    print(f"  BAD: {np.mean(bad_seg):.3f} ± {np.std(bad_seg):.3f}")
    print(f"  GOOD: {np.mean(good_seg):.3f} ± {np.std(good_seg):.3f}")
    
    bad_neg_segs = [f['negative_segments'] for f in bad_features if 'negative_segments' in f]
    good_neg_segs = [f['negative_segments'] for f in good_features if 'negative_segments' in f]
    
    print(f"\nNegative correlation segments:")
    print(f"  BAD: {np.mean(bad_neg_segs):.1f} ± {np.std(bad_neg_segs):.1f}")
    print(f"  GOOD: {np.mean(good_neg_segs):.1f} ± {np.std(good_neg_segs):.1f}")


def suggest_thresholds(bad_features: List[Dict], good_features: List[Dict]):
    """Suggest classification thresholds based on analysis."""
    
    print(f"\n{'='*70}")
    print("RECOMMENDED THRESHOLDS (Option A: Threshold-Based)")
    print(f"{'='*70}")
    
    # Extract correlations
    bad_corr = [f['hr_speed_pearson'] for f in bad_features if 'hr_speed_pearson' in f]
    good_corr = [f['hr_speed_pearson'] for f in good_features if 'hr_speed_pearson' in f]
    
    # Find separating threshold
    # Goal: Minimize false positives (flagging good workouts as bad)
    # Use 10th percentile of good workouts as threshold
    threshold = np.percentile(good_corr, 10)
    
    # Test threshold performance
    bad_below = sum(c < threshold for c in bad_corr)
    good_below = sum(c < threshold for c in good_corr)
    
    print(f"\n1. Correlation Threshold: {threshold:.3f}")
    print(f"   Flag workout if: correlation < {threshold:.3f}")
    print(f"   Performance on sample:")
    print(f"     - Catches {bad_below}/{len(bad_corr)} bad workouts ({bad_below/len(bad_corr)*100:.1f}%)")
    print(f"     - False positives: {good_below}/{len(good_corr)} good workouts ({good_below/len(good_corr)*100:.1f}%)")
    
    # Alternative thresholds
    print(f"\n2. Conservative Threshold: 0.10")
    print(f"   Flag workout if: correlation < 0.10")
    bad_010 = sum(c < 0.10 for c in bad_corr)
    good_010 = sum(c < 0.10 for c in good_corr)
    print(f"     - Catches {bad_010}/{len(bad_corr)} bad workouts ({bad_010/len(bad_corr)*100:.1f}%)")
    print(f"     - False positives: {good_010}/{len(good_corr)} ({good_010/len(good_corr)*100:.1f}%)")
    
    print(f"\n3. Aggressive Threshold: 0.20")
    print(f"   Flag workout if: correlation < 0.20")
    bad_020 = sum(c < 0.20 for c in bad_corr)
    good_020 = sum(c < 0.20 for c in good_corr)
    print(f"     - Catches {bad_020}/{len(bad_corr)} bad workouts ({bad_020/len(bad_corr)*100:.1f}%)")
    print(f"     - False positives: {good_020}/{len(good_corr)} ({good_020/len(good_corr)*100:.1f}%)")
    
    # Combined rules
    print(f"\n4. Combined Rules (recommended):")
    print(f"   Flag if ANY of:")
    print(f"     - correlation < {threshold:.3f}")
    print(f"     - inverse_trend = True")
    print(f"     - stable_speed_rising_hr = True")
    print(f"     - negative_segments >= 3")
    
    # Test combined rules
    def apply_rules(f, thresh):
        return (f.get('hr_speed_pearson', 1.0) < thresh or
                f.get('inverse_trend', False) or
                f.get('stable_speed_rising_hr', False) or
                f.get('negative_segments', 0) >= 3)
    
    bad_flagged = sum(apply_rules(f, threshold) for f in bad_features)
    good_flagged = sum(apply_rules(f, threshold) for f in good_features)
    
    print(f"   Performance:")
    print(f"     - Catches {bad_flagged}/{len(bad_features)} bad workouts ({bad_flagged/len(bad_features)*100:.1f}%)")
    print(f"     - False positives: {good_flagged}/{len(good_features)} ({good_flagged/len(good_features)*100:.1f}%)")
    
    print(f"\n{'─'*70}")
    print("NEXT STEP:")
    print(f"  Run: python3 flag_weird_patterns.py --threshold {threshold:.3f}")
    print(f"{'─'*70}")
    
    return threshold


def plot_distributions(bad_features: List[Dict], good_features: List[Dict]):
    """Plot correlation distributions."""
    
    bad_corr = [f['hr_speed_pearson'] for f in bad_features if 'hr_speed_pearson' in f]
    good_corr = [f['hr_speed_pearson'] for f in good_features if 'hr_speed_pearson' in f]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1.hist(good_corr, bins=50, alpha=0.6, label=f'Good (n={len(good_corr)})', color='green')
    ax1.hist(bad_corr, bins=50, alpha=0.6, label=f'Bad (n={len(bad_corr)})', color='red')
    ax1.axvline(np.mean(good_corr), color='green', linestyle='--', linewidth=2, label=f'Good mean: {np.mean(good_corr):.3f}')
    ax1.axvline(np.mean(bad_corr), color='red', linestyle='--', linewidth=2, label=f'Bad mean: {np.mean(bad_corr):.3f}')
    ax1.set_xlabel('HR-Speed Correlation (Pearson)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of HR-Speed Correlations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot([good_corr, bad_corr], labels=['Good', 'Bad'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.6))
    ax2.set_ylabel('HR-Speed Correlation (Pearson)')
    ax2.set_title('Correlation by Workout Quality')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "checked_patterns_analysis.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze checked patterns from gallery')
    parser.add_argument('--input', type=str, required=True, help='Gallery export JSON (checked_workouts_*.json)')
    parser.add_argument('--sample-size', type=int, help='Number of workouts in your sample (default: auto-detect)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print("ANALYZING CHECKED PATTERNS (10% Annotation Strategy)")
    print(f"{'='*70}")
    
    # Load checked workouts
    print(f"\nLoading checked workouts from: {input_path}")
    checked_lines = load_checked_workouts(input_path)
    print(f"Found {len(checked_lines):,} checked workouts")
    
    # Load all good workout lines
    print("\nLoading good workout list...")
    all_good_lines = load_good_workout_lines()
    print(f"Total good workouts: {len(all_good_lines):,}")
    
    # Determine sample size
    if args.sample_size:
        sample_lines = all_good_lines[:args.sample_size]
    else:
        # Auto-detect: assume checked workouts are from first N lines
        max_checked = max(checked_lines)
        max_checked_idx = all_good_lines.index(max_checked) if max_checked in all_good_lines else -1
        sample_lines = all_good_lines[:max_checked_idx+1000]  # Add buffer
        print(f"Auto-detected sample size: {len(sample_lines):,} workouts")
    
    # Load computed speeds
    print("\nLoading computed speeds...")
    speed_lookup = load_computed_speeds()
    print(f"Loaded {len(speed_lookup):,} computed speeds")
    
    # Analyze patterns
    bad_features, good_features = analyze_patterns(checked_lines, sample_lines, speed_lookup)
    
    if len(bad_features) == 0:
        print("\nERROR: No bad workouts found in sample!")
        print("Make sure the checked line numbers are in the sample range.")
        sys.exit(1)
    
    # Print statistics
    print_statistics(bad_features, good_features)
    
    # Suggest thresholds
    threshold = suggest_thresholds(bad_features, good_features)
    
    # Plot distributions
    plot_distributions(bad_features, good_features)
    
    # Save results
    output_file = OUTPUT_DIR / "checked_patterns_thresholds.json"
    with open(output_file, 'w') as f:
        json.dump({
            'recommended_threshold': float(threshold),
            'bad_workouts_count': len(bad_features),
            'good_workouts_count': len(good_features),
            'bad_correlation_mean': float(np.mean([f['hr_speed_pearson'] for f in bad_features if 'hr_speed_pearson' in f])),
            'good_correlation_mean': float(np.mean([f['hr_speed_pearson'] for f in good_features if 'hr_speed_pearson' in f])),
        }, f, indent=2)
    
    print(f"\n✓ Saved thresholds to {output_file}")


if __name__ == "__main__":
    main()
