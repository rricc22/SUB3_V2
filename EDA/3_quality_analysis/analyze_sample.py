#!/usr/bin/env python3
"""
Analyze a random sample of running workouts for quality issues.

Memory-efficient: processes one workout at a time.

Author: OpenCode
Date: 2025-01-11
"""

import subprocess
import ast
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

DATA_FILE = Path(__file__).parent.parent.parent / "DATA" / "raw" / "endomondoHR_proper-002.json"


def get_running_lines_with_speed(limit=1000):
    """Get line numbers of running workouts with speed field."""
    result = subprocess.run(
        ['grep', '-n', "'sport': 'run'", str(DATA_FILE)],
        capture_output=True, text=True
    )
    
    lines = []
    for line in result.stdout.split('\n'):
        if line and "'speed':" in line:
            lines.append(int(line.split(':')[0]))
            if len(lines) >= limit:
                break
    return lines


def get_workout(line_num):
    """Get a single workout by line number."""
    result = subprocess.run(
        ['sed', '-n', f'{line_num}p', str(DATA_FILE)],
        capture_output=True, text=True
    )
    return ast.literal_eval(result.stdout.strip())


def analyze_workout(workout):
    """Analyze a single workout for quality issues."""
    hr = np.array(workout['heart_rate'])
    speed = np.array(workout['speed'])
    
    issues = []
    
    # 1. HR spikes (>30 BPM change per timestep)
    hr_diff = np.abs(np.diff(hr))
    spike_count = np.sum(hr_diff > 30)
    if spike_count > 0:
        issues.append(f"HR_SPIKE:{spike_count}")
    
    # 2. HR out of range
    if np.any(hr < 40):
        issues.append(f"HR_LOW:{np.sum(hr < 40)}")
    if np.any(hr > 210):
        issues.append(f"HR_HIGH:{np.sum(hr > 210)}")
    
    # 3. Zero HR values
    if np.any(hr == 0):
        issues.append(f"HR_ZERO:{np.sum(hr == 0)}")
    
    # 4. HR flatlines (>15 identical consecutive values)
    max_run = 1
    current_run = 1
    for i in range(1, len(hr)):
        if hr[i] == hr[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    if max_run > 15:
        issues.append(f"HR_FLATLINE:{max_run}")
    
    # 5. Impossible running speed (>25 km/h sustained)
    if np.any(speed > 25):
        issues.append(f"SPEED_HIGH:{np.sum(speed > 25)}")
    
    # 6. Speed-HR correlation
    if len(hr) > 50:
        valid_mask = (hr > 50) & (hr < 200) & (speed > 0)
        if np.sum(valid_mask) > 30:
            corr = np.corrcoef(speed[valid_mask], hr[valid_mask])[0, 1]
            if np.isnan(corr):
                issues.append("CORR_NAN")
            elif corr < 0:
                issues.append(f"CORR_NEG:{corr:.2f}")
            elif corr < 0.1:
                issues.append(f"CORR_LOW:{corr:.2f}")
        else:
            corr = np.nan
    else:
        corr = np.nan
    
    return {
        'issues': issues,
        'hr_mean': np.mean(hr),
        'hr_std': np.std(hr),
        'hr_min': np.min(hr),
        'hr_max': np.max(hr),
        'speed_mean': np.mean(speed),
        'speed_max': np.max(speed),
        'correlation': corr if not np.isnan(corr) else None
    }


def main():
    print("Finding running workouts with speed data...")
    lines = get_running_lines_with_speed(limit=50000)
    print(f"Found {len(lines)} running workouts with speed")
    
    # Sample 500 for analysis
    sample_size = min(500, len(lines))
    sample = random.sample(lines, sample_size)
    print(f"Analyzing {sample_size} random samples...\n")
    
    issue_counts = defaultdict(int)
    correlations = []
    problem_workouts = []
    
    for i, line_num in enumerate(sample):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{sample_size}...")
        
        try:
            workout = get_workout(line_num)
            analysis = analyze_workout(workout)
            
            for issue in analysis['issues']:
                issue_type = issue.split(':')[0]
                issue_counts[issue_type] += 1
            
            if analysis['correlation'] is not None:
                correlations.append(analysis['correlation'])
            
            if len(analysis['issues']) > 0:
                problem_workouts.append({
                    'line': line_num,
                    'issues': analysis['issues'],
                    'hr_range': f"{analysis['hr_min']:.0f}-{analysis['hr_max']:.0f}",
                    'speed_max': analysis['speed_max']
                })
        except Exception as e:
            print(f"  Error at line {line_num}: {e}")
    
    # Report
    print("\n" + "="*60)
    print("QUALITY ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nSample size: {sample_size} running workouts")
    print(f"Total with issues: {len(problem_workouts)} ({len(problem_workouts)/sample_size*100:.1f}%)")
    
    print("\n--- Issue Breakdown ---")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count} ({count/sample_size*100:.1f}%)")
    
    print("\n--- Correlation Statistics ---")
    if correlations:
        corr_arr = np.array(correlations)
        print(f"  Mean: {np.mean(corr_arr):.3f}")
        print(f"  Std:  {np.std(corr_arr):.3f}")
        print(f"  Min:  {np.min(corr_arr):.3f}")
        print(f"  Max:  {np.max(corr_arr):.3f}")
        print(f"  <0.1: {np.sum(corr_arr < 0.1)} ({np.sum(corr_arr < 0.1)/len(corr_arr)*100:.1f}%)")
        print(f"  <0.2: {np.sum(corr_arr < 0.2)} ({np.sum(corr_arr < 0.2)/len(corr_arr)*100:.1f}%)")
        print(f"  >0.5: {np.sum(corr_arr > 0.5)} ({np.sum(corr_arr > 0.5)/len(corr_arr)*100:.1f}%)")
    
    print("\n--- Sample Problem Workouts ---")
    for pw in problem_workouts[:10]:
        print(f"  Line {pw['line']}: {pw['issues']} (HR: {pw['hr_range']}, MaxSpeed: {pw['speed_max']:.1f})")
    
    # Clean workouts percentage
    clean_count = sample_size - len(problem_workouts)
    print(f"\n--- Summary ---")
    print(f"Clean workouts: {clean_count} ({clean_count/sample_size*100:.1f}%)")
    print(f"Problematic: {len(problem_workouts)} ({len(problem_workouts)/sample_size*100:.1f}%)")


if __name__ == "__main__":
    main()
