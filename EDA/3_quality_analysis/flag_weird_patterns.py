#!/usr/bin/env python3
"""
Auto-Flag Weird Patterns in Remaining Dataset.

Applies learned thresholds from 10% annotated sample to flag
physiologically weird patterns in the remaining 90% of data.

Usage:
    # Use recommended threshold from analyze_checked_patterns.py
    python3 flag_weird_patterns.py --threshold 0.15
    
    # Custom threshold
    python3 flag_weird_patterns.py --threshold 0.10 --output flagged_workouts.json
    
    # Dry run (no output, just stats)
    python3 flag_weird_patterns.py --threshold 0.15 --dry-run

Author: OpenCode
Date: 2026-01-13
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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
# Classification Rules
# =============================================================================

def is_weird_pattern(features: Dict, threshold: float) -> Tuple[bool, str]:
    """
    Apply classification rules to detect weird patterns.
    
    Args:
        features: Anomaly features dict from compute_anomaly_features()
        threshold: Correlation threshold (e.g., 0.15)
    
    Returns:
        (is_weird, reason) tuple
    """
    
    # Rule 1: Very low or negative correlation
    corr = features.get('hr_speed_pearson', 1.0)
    if corr < threshold:
        return True, f"Low correlation ({corr:.3f} < {threshold})"
    
    # Rule 2: Inverse trends (speed down, HR up)
    if features.get('inverse_trend', False):
        return True, "Inverse trend (speed↓ while HR↑)"
    
    # Rule 3: Stable speed, rising HR (on flat terrain)
    if features.get('stable_speed_rising_hr', False):
        return True, "Stable speed but HR rising (flat terrain)"
    
    # Rule 4: Multiple negative correlation segments
    if features.get('negative_segments', 0) >= 3:
        return True, f"Multiple negative segments ({features['negative_segments']})"
    
    # Rule 5: Very low mean segment correlation
    if features.get('mean_segment_corr', 1.0) < 0.10:
        return True, f"Very low segment correlation ({features['mean_segment_corr']:.3f})"
    
    # Rule 6: Negative correlation
    if corr < 0:
        return True, f"Negative correlation ({corr:.3f})"
    
    return False, ""


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Auto-flag weird patterns in dataset')
    parser.add_argument('--threshold', type=float, required=True, 
                       help='Correlation threshold (e.g., 0.15)')
    parser.add_argument('--output', type=str, default='flagged_workouts.json',
                       help='Output JSON file')
    parser.add_argument('--exclude-checked', type=str,
                       help='Exclude workouts from this checked list (optional)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of workouts to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show statistics without saving output')
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print("AUTO-FLAGGING WEIRD PATTERNS")
    print(f"{'='*70}")
    print(f"\nThreshold: {args.threshold}")
    print(f"Output: {args.output}")
    
    # Load good workout lines
    print("\nLoading good workout list...")
    all_good_lines = load_good_workout_lines()
    print(f"Total good workouts: {len(all_good_lines):,}")
    
    # Optionally exclude already-checked workouts
    exclude_lines = set()
    if args.exclude_checked:
        with open(args.exclude_checked, 'r') as f:
            data = json.load(f)
            exclude_lines = set(data['checked_line_numbers'])
        print(f"Excluding {len(exclude_lines):,} already-checked workouts")
    
    # Filter lines
    lines_to_process = [l for l in all_good_lines if l not in exclude_lines]
    
    if args.limit:
        lines_to_process = lines_to_process[:args.limit]
        print(f"Limited to {args.limit:,} workouts (testing mode)")
    
    print(f"Processing {len(lines_to_process):,} workouts...")
    
    # Load computed speeds
    print("\nLoading computed speeds...")
    speed_lookup = load_computed_speeds()
    print(f"Loaded {len(speed_lookup):,} computed speeds\n")
    
    # Process workouts
    flagged_workouts = []
    processed = 0
    skipped = 0
    
    print("Processing...")
    for i, line_num in enumerate(lines_to_process, 1):
        if i % 1000 == 0:
            print(f"  {i:,}/{len(lines_to_process):,} | Flagged so far: {len(flagged_workouts):,}")
        
        workout = load_workout(line_num, speed_lookup)
        if workout is None:
            skipped += 1
            continue
        
        # Extract arrays
        hr = np.array(workout.get('heart_rate', []))
        speed = np.array(workout.get('speed', []))
        altitude = np.array(workout.get('altitude', []))
        timestamps = np.array(workout.get('timestamp', []))
        
        if len(hr) == 0 or len(speed) == 0:
            skipped += 1
            continue
        
        # Compute features
        features = compute_anomaly_features(hr, speed, altitude, timestamps)
        processed += 1
        
        # Apply classification rules
        is_weird, reason = is_weird_pattern(features, args.threshold)
        
        if is_weird:
            flagged_workouts.append({
                'line_num': line_num,
                'workout_id': workout.get('id', ''),
                'reason': reason,
                'correlation': features.get('hr_speed_pearson', 0.0),
                'inverse_trend': features.get('inverse_trend', False),
                'stable_speed_rising_hr': features.get('stable_speed_rising_hr', False),
                'negative_segments': features.get('negative_segments', 0),
                'mean_segment_corr': features.get('mean_segment_corr', 0.0),
                'hr_mean': features.get('hr_mean', 0.0),
                'speed_mean': features.get('speed_mean', 0.0),
                'duration_min': features.get('duration_min', 0.0),
            })
    
    # Statistics
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nProcessed: {processed:,} workouts")
    print(f"Skipped: {skipped:,} (couldn't load)")
    print(f"Flagged as weird: {len(flagged_workouts):,} ({len(flagged_workouts)/processed*100:.1f}%)")
    
    # Breakdown by reason
    print(f"\nFlagged by reason:")
    reasons = {}
    for w in flagged_workouts:
        reason = w['reason'].split('(')[0].strip()  # Group similar reasons
        reasons[reason] = reasons.get(reason, 0) + 1
    
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,} ({count/len(flagged_workouts)*100:.1f}%)")
    
    # Correlation statistics
    flagged_corrs = [w['correlation'] for w in flagged_workouts]
    print(f"\nFlagged workouts - correlation stats:")
    print(f"  Mean: {np.mean(flagged_corrs):.3f}")
    print(f"  Median: {np.median(flagged_corrs):.3f}")
    print(f"  Min/Max: {np.min(flagged_corrs):.3f} / {np.max(flagged_corrs):.3f}")
    
    # Save results
    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / args.output
        
        output_data = {
            'threshold': args.threshold,
            'total_processed': processed,
            'total_flagged': len(flagged_workouts),
            'flagged_percentage': len(flagged_workouts) / processed * 100 if processed > 0 else 0,
            'flagged_workouts': flagged_workouts,
            'reason_breakdown': reasons,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved {len(flagged_workouts):,} flagged workouts to {output_path}")
        
        # Also save line numbers only (for easy exclusion in preprocessing)
        lines_output = OUTPUT_DIR / args.output.replace('.json', '_lines.txt')
        with open(lines_output, 'w') as f:
            for w in flagged_workouts:
                f.write(f"{w['line_num']}\n")
        
        print(f"✓ Saved line numbers to {lines_output}")
    else:
        print("\n(Dry run - no files saved)")
    
    print(f"\n{'─'*70}")
    print("NEXT STEPS:")
    print("1. Review flagged workouts in gallery:")
    print(f"   - Open gallery and search for flagged line numbers")
    print("2. Verify false positives (good workouts incorrectly flagged)")
    print("3. Adjust threshold if needed and re-run")
    print("4. Use flagged list to exclude from training dataset")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
