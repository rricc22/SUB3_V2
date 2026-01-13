#!/usr/bin/env python3
"""
Stage 3: Apply Corrections and Produce Clean Dataset

This script:
1. Merges Stage 1 (PASS) + Stage 2 (AUTO_PASS + LLM KEEP/FIX) results
2. Applies HR offset corrections where needed
3. Classifies workout types for all workouts
4. Produces final clean dataset ready for training

Author: OpenCode
Date: 2025-01-11
"""

import json
import ast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse
import linecache


# =============================================================================
# Configuration
# =============================================================================

# Workout type classification thresholds
class WorkoutClassifier:
    """Rule-based workout type classifier for workouts not classified by LLM."""
    
    @staticmethod
    def classify(hr: np.ndarray, speed: np.ndarray, timestamps: np.ndarray) -> str:
        """
        Classify workout type based on HR patterns.
        
        Returns: INTENSIVE, INTERVALS, STEADY, RECOVERY, or UNKNOWN
        """
        if len(hr) < 50:
            return "UNKNOWN"
        
        mean_hr = np.mean(hr)
        max_hr = np.max(hr)
        hr_std = np.std(hr)
        
        # Calculate HR zones time
        high_effort = np.sum(hr >= 170) / len(hr)  # % time in high zone
        recovery_zone = np.sum(hr < 150) / len(hr)  # % time in recovery
        
        # Detect oscillation pattern (for intervals)
        hr_diff = np.abs(np.diff(hr))
        large_swings = np.sum(hr_diff > 15) / len(hr_diff)  # % of large HR changes
        
        # Classification rules
        if mean_hr >= 170 and hr_std < 12:
            return "INTENSIVE"
        
        if large_swings > 0.05 and hr_std > 15 and high_effort > 0.2 and recovery_zone > 0.2:
            return "INTERVALS"
        
        if 145 <= mean_hr < 170 and hr_std < 15:
            return "STEADY"
        
        if mean_hr < 150 and max_hr < 170:
            return "RECOVERY"
        
        # Fallback based on mean HR
        if mean_hr >= 165:
            return "INTENSIVE"
        elif mean_hr >= 150:
            return "STEADY"
        elif mean_hr >= 130:
            return "RECOVERY"
        
        return "UNKNOWN"


# =============================================================================
# Correction Functions
# =============================================================================

def apply_offset_correction(hr: np.ndarray, offset: float, 
                           start_pct: Optional[float] = None) -> np.ndarray:
    """
    Apply HR offset correction.
    
    Args:
        hr: Original HR array
        offset: Offset to add (positive = increase HR)
        start_pct: If provided, only apply offset from this % of workout onward
    
    Returns:
        Corrected HR array
    """
    hr_corrected = hr.copy().astype(float)
    
    if start_pct is not None and 0 < start_pct < 100:
        # Partial offset - only apply from start_pct onward
        start_idx = int(len(hr) * start_pct / 100)
        hr_corrected[start_idx:] += offset
    else:
        # Full workout offset
        hr_corrected += offset
    
    # Clamp to valid range
    hr_corrected = np.clip(hr_corrected, 30, 220)
    
    return hr_corrected.astype(int)


def validate_corrected_hr(hr: np.ndarray, speed: np.ndarray) -> bool:
    """Check if corrected HR is physiologically plausible."""
    if len(hr) == 0:
        return False
    
    mean_hr = np.mean(hr)
    max_hr = np.max(hr)
    mean_speed = np.mean(speed) if len(speed) > 0 else 0
    
    # After correction, HR should be in reasonable range
    if mean_hr < 100 or mean_hr > 200:
        return False
    
    # If running (speed > 8 km/h), HR should be elevated
    if mean_speed > 8 and mean_hr < 120:
        return False
    
    return True


# =============================================================================
# Main Processing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 3: Apply corrections and produce clean dataset')
    parser.add_argument('--raw-data', '-r', type=str,
                        default='DATA/raw/endomondoHR_proper-002.json',
                        help='Raw data file')
    parser.add_argument('--stage1', '-s1', type=str,
                        default='Preprocessing/stage1_full_output.json',
                        help='Stage 1 output file')
    parser.add_argument('--stage2', '-s2', type=str,
                        default='Preprocessing/stage2_output.json',
                        help='Stage 2 output file')
    parser.add_argument('--output', '-o', type=str,
                        default='Preprocessing/clean_dataset.json',
                        help='Output clean dataset')
    parser.add_argument('--output-index', '-oi', type=str,
                        default='DATA/indices/clean_workouts.txt',
                        help='Output index file with clean workout line numbers')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of workouts to process')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    raw_file = project_root / args.raw_data
    stage1_file = project_root / args.stage1
    stage2_file = project_root / args.stage2
    output_file = project_root / args.output
    index_file = project_root / args.output_index
    
    print(f"Stage 3: Apply Corrections & Produce Clean Dataset")
    print(f"=" * 60)
    print(f"Raw data: {raw_file}")
    print(f"Stage 1: {stage1_file}")
    print(f"Stage 2: {stage2_file}")
    print(f"Output: {output_file}")
    print()
    
    # Load stage 1 results
    with open(stage1_file, 'r') as f:
        stage1_data = json.load(f)
    
    # Load stage 2 results
    with open(stage2_file, 'r') as f:
        stage2_data = json.load(f)
    
    # Build lookup from stage 2
    stage2_lookup = {}
    
    # Auto-passed workouts from stage 2 triage
    for w in stage2_data.get('auto_passed', []):
        stage2_lookup[w['line_number']] = {
            'decision': 'KEEP',
            'source': 'TRIAGE_AUTO_PASS',
            'workout_type': w.get('workout_type'),
            'has_offset_error': False,
            'estimated_offset': None
        }
    
    # LLM-validated workouts
    for w in stage2_data.get('llm_validated', []):
        stage2_lookup[w['line_number']] = {
            'decision': w.get('decision', 'EXCLUDE'),
            'source': 'LLM',
            'workout_type': w.get('workout_type'),
            'has_offset_error': w.get('has_offset_error', False),
            'estimated_offset': w.get('estimated_offset'),
            'offset_start_pct': w.get('offset_start_pct'),
            'confidence': w.get('confidence', 0)
        }
    
    # Collect all workouts to include
    workouts_to_process = []
    
    for w in stage1_data['workouts']:
        line_num = w['line_number']
        decision = w['decision']
        
        if decision == 'PASS':
            # Stage 1 passed - include directly
            workouts_to_process.append({
                'line_number': line_num,
                'workout_id': w['workout_id'],
                'user_id': w['user_id'],
                'source': 'STAGE1_PASS',
                'needs_correction': False,
                'stats': w['stats']
            })
        
        elif decision == 'FLAG':
            # Check stage 2 result
            s2 = stage2_lookup.get(line_num)
            
            if s2 is None:
                # Not yet processed by stage 2 - skip for now
                continue
            
            if s2['decision'] == 'KEEP':
                workouts_to_process.append({
                    'line_number': line_num,
                    'workout_id': w['workout_id'],
                    'user_id': w['user_id'],
                    'source': s2['source'],
                    'needs_correction': False,
                    'workout_type': s2.get('workout_type'),
                    'stats': w['stats']
                })
            
            elif s2['decision'] == 'FIX':
                workouts_to_process.append({
                    'line_number': line_num,
                    'workout_id': w['workout_id'],
                    'user_id': w['user_id'],
                    'source': 'LLM_FIX',
                    'needs_correction': True,
                    'offset': s2.get('estimated_offset', 0),
                    'offset_start_pct': s2.get('offset_start_pct'),
                    'workout_type': s2.get('workout_type'),
                    'stats': w['stats']
                })
            # EXCLUDE decisions are skipped
    
    print(f"Workouts to process: {len(workouts_to_process)}")
    print(f"  - Stage 1 PASS: {sum(1 for w in workouts_to_process if w['source'] == 'STAGE1_PASS')}")
    print(f"  - Triage AUTO_PASS: {sum(1 for w in workouts_to_process if w['source'] == 'TRIAGE_AUTO_PASS')}")
    print(f"  - LLM KEEP: {sum(1 for w in workouts_to_process if w['source'] == 'LLM' and not w['needs_correction'])}")
    print(f"  - LLM FIX: {sum(1 for w in workouts_to_process if w['needs_correction'])}")
    print()
    
    if args.limit:
        workouts_to_process = workouts_to_process[:args.limit]
        print(f"Limited to: {len(workouts_to_process)}")
    
    # Process and build clean dataset
    classifier = WorkoutClassifier()
    
    clean_workouts = []
    corrections_applied = 0
    classification_stats = {}
    
    print("Processing workouts...")
    
    for i, meta in enumerate(workouts_to_process):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(workouts_to_process)}...")
        
        line_num = meta['line_number']
        
        try:
            # Load raw workout
            line = linecache.getline(str(raw_file), line_num)
            workout = ast.literal_eval(line.strip())
            
            hr = np.array(workout.get('heart_rate', []))
            speed = np.array(workout.get('speed', []))
            timestamps = np.array(workout.get('timestamp', []))
            altitude = np.array(workout.get('altitude', []))
            
            # Apply correction if needed
            if meta['needs_correction'] and meta.get('offset'):
                hr = apply_offset_correction(
                    hr, 
                    meta['offset'],
                    meta.get('offset_start_pct')
                )
                corrections_applied += 1
                
                # Validate correction
                if not validate_corrected_hr(hr, speed):
                    continue  # Skip if correction resulted in bad data
            
            # Classify workout type if not already classified
            workout_type = meta.get('workout_type')
            if not workout_type or workout_type == 'UNKNOWN':
                workout_type = classifier.classify(hr, speed, timestamps)
            
            classification_stats[workout_type] = classification_stats.get(workout_type, 0) + 1
            
            # Build clean workout entry
            clean_entry = {
                'line_number': line_num,
                'workout_id': workout.get('id'),
                'user_id': workout.get('userId'),
                'sport': workout.get('sport', 'running'),
                'workout_type': workout_type,
                'corrected': meta['needs_correction'],
                'offset_applied': meta.get('offset') if meta['needs_correction'] else None,
                'duration_min': meta['stats'].get('duration_min', 0),
                'data_points': len(hr),
                # Store corrected/original data
                'heart_rate': hr.tolist(),
                'speed': speed.tolist() if len(speed) > 0 else [],
                'timestamp': timestamps.tolist(),
                'altitude': altitude.tolist() if len(altitude) > 0 else [],
                # Stats on clean data
                'hr_mean': float(np.mean(hr)),
                'hr_std': float(np.std(hr)),
                'hr_min': int(np.min(hr)),
                'hr_max': int(np.max(hr)),
                'speed_mean': float(np.mean(speed)) if len(speed) > 0 else 0,
            }
            
            clean_workouts.append(clean_entry)
        
        except Exception as e:
            print(f"  Error processing line {line_num}: {e}")
            continue
    
    # Clear linecache
    linecache.clearcache()
    
    # Build output
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_clean_workouts': len(clean_workouts),
            'corrections_applied': corrections_applied,
            'classification_stats': classification_stats,
            'sources': {
                'stage1_pass': sum(1 for w in workouts_to_process if w['source'] == 'STAGE1_PASS'),
                'triage_auto_pass': sum(1 for w in workouts_to_process if w['source'] == 'TRIAGE_AUTO_PASS'),
                'llm_keep': sum(1 for w in workouts_to_process if w['source'] == 'LLM'),
                'llm_fix': sum(1 for w in workouts_to_process if w['source'] == 'LLM_FIX')
            }
        },
        'workouts': clean_workouts
    }
    
    # Save clean dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f)
    
    # Save index file (line numbers only)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, 'w') as f:
        for w in clean_workouts:
            f.write(f"{w['line_number']}\n")
    
    # Summary
    print()
    print(f"=" * 60)
    print(f"STAGE 3 SUMMARY")
    print(f"=" * 60)
    print(f"Total clean workouts: {len(clean_workouts)}")
    print(f"Corrections applied: {corrections_applied}")
    print()
    print("Workout type distribution:")
    for wt, count in sorted(classification_stats.items(), key=lambda x: -x[1]):
        pct = count / len(clean_workouts) * 100
        print(f"  {wt:12s}: {count:6d} ({pct:5.1f}%)")
    print()
    print(f"Clean dataset saved to: {output_file}")
    print(f"Index file saved to: {index_file}")
    print()
    
    # Quick validation
    print("Sample clean workouts:")
    for w in clean_workouts[:3]:
        print(f"  ID {w['workout_id']}: {w['workout_type']}, "
              f"HR {w['hr_mean']:.0f}±{w['hr_std']:.0f} BPM, "
              f"{w['duration_min']:.0f} min"
              f"{' [CORRECTED +' + str(w['offset_applied']) + ' BPM]' if w['corrected'] else ''}")


if __name__ == "__main__":
    main()
