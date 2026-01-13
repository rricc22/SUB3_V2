#!/usr/bin/env python3
"""
Apply moving average smoothing to workout time series data.

Smooths heart_rate, speed, and altitude arrays using a moving average
with a specified window size (default: 6 timestamps).

Author: Claude Code
Date: 2026-01-13
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse


def moving_average(data: list, window_size: int) -> list:
    """
    Apply moving average smoothing to a 1D array.

    Uses 'valid' convolution which returns smoothed values only where
    the full window fits. Edges are handled by padding with first/last values.

    Args:
        data: Input array as list
        window_size: Size of the moving average window

    Returns:
        Smoothed array as list
    """
    if len(data) == 0:
        return data

    if len(data) < window_size:
        # If data shorter than window, just return the mean
        return [np.mean(data)] * len(data)

    arr = np.array(data, dtype=np.float32)

    # Use convolution for efficient moving average
    kernel = np.ones(window_size) / window_size

    # 'same' mode keeps the same length as input
    smoothed = np.convolve(arr, kernel, mode='same')

    # Fix edges: convolution at edges uses partial windows
    # Replace edge values with proper partial averages
    half_window = window_size // 2

    # Left edge
    for i in range(half_window):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        smoothed[i] = np.mean(arr[start:end])

    # Right edge
    for i in range(len(arr) - half_window, len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        smoothed[i] = np.mean(arr[start:end])

    return smoothed.tolist()


def smooth_workout(workout: dict, window_size: int) -> dict:
    """
    Apply moving average to heart_rate, speed, and altitude in a workout.

    Args:
        workout: Workout dictionary
        window_size: Size of the moving average window

    Returns:
        Workout dictionary with smoothed arrays
    """
    # Create a copy to avoid modifying original
    smoothed = workout.copy()

    # Smooth heart_rate (always present)
    if 'heart_rate' in smoothed and len(smoothed['heart_rate']) > 0:
        smoothed['heart_rate'] = moving_average(smoothed['heart_rate'], window_size)

    # Smooth speed (if present)
    if 'speed' in smoothed and len(smoothed['speed']) > 0:
        smoothed['speed'] = moving_average(smoothed['speed'], window_size)

    # Smooth altitude (if present)
    if 'altitude' in smoothed and len(smoothed['altitude']) > 0:
        smoothed['altitude'] = moving_average(smoothed['altitude'], window_size)

    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description='Apply moving average smoothing to workout data'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='Preprocessing/clean_dataset_merged.json',
        help='Input JSON file (default: Preprocessing/clean_dataset_merged.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='Preprocessing/clean_dataset_smoothed.json',
        help='Output JSON file (default: Preprocessing/clean_dataset_smoothed.json)'
    )
    parser.add_argument(
        '--window', '-w',
        type=int,
        default=6,
        help='Moving average window size (default: 6)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of workouts to process (for testing)'
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / args.input
    output_file = project_root / args.output

    print("=" * 70)
    print("MOVING AVERAGE SMOOTHING")
    print("=" * 70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Window size: {args.window} timestamps")
    if args.limit:
        print(f"Limit: {args.limit} workouts (testing mode)")
    print()

    # Load input data
    print("Loading input data...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    original_count = len(data['workouts'])
    print(f"Loaded {original_count:,} workouts")
    print()

    # Apply limit if specified
    if args.limit:
        data['workouts'] = data['workouts'][:args.limit]
        print(f"Processing first {args.limit} workouts (testing mode)...")
    else:
        print("Processing all workouts...")

    # Process workouts
    smoothed_workouts = []

    for i, workout in enumerate(data['workouts']):
        if (i + 1) % 1000 == 0:
            progress = (i + 1) / len(data['workouts']) * 100
            print(f"  Progress: {i+1:,}/{len(data['workouts']):,} ({progress:.1f}%)")

        smoothed_workout = smooth_workout(workout, args.window)
        smoothed_workouts.append(smoothed_workout)

    print(f"  Completed: {len(smoothed_workouts):,} workouts smoothed")
    print()

    # Update metadata
    data['workouts'] = smoothed_workouts

    if 'metadata' not in data:
        data['metadata'] = {}

    data['metadata']['smoothing_applied'] = {
        'timestamp': datetime.now().isoformat(),
        'window_size': args.window,
        'fields_smoothed': ['heart_rate', 'speed', 'altitude'],
        'original_file': str(input_file.name)
    }

    # Save output
    print("Saving smoothed data...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Print file sizes
    input_size_mb = input_file.stat().st_size / (1024 * 1024)
    output_size_mb = output_file.stat().st_size / (1024 * 1024)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total workouts processed: {len(smoothed_workouts):,}")
    print(f"Window size: {args.window} timestamps")
    print(f"Input file size:  {input_size_mb:.1f} MB")
    print(f"Output file size: {output_size_mb:.1f} MB")
    print()
    print(f"Smoothed data saved to: {output_file}")
    print()
    print("Fields smoothed:")
    print("  - heart_rate")
    print("  - speed")
    print("  - altitude")


if __name__ == "__main__":
    main()
