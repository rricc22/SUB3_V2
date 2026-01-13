#!/usr/bin/env python3
"""
Quick viewer for smoothed workout data with before/after comparison.

Usage:
    python3 Preprocessing/view_smoothed.py --index 0
    python3 Preprocessing/view_smoothed.py --index 5 --compare

Author: Claude Code
Date: 2026-01-13
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_workout(workout: dict, ax_hr, ax_speed, ax_alt, title_prefix=""):
    """Plot a single workout's time series data."""

    hr = workout.get('heart_rate', [])
    speed = workout.get('speed', [])
    altitude = workout.get('altitude', [])
    timestamps = workout.get('timestamp', [])

    # Convert timestamps to minutes from start
    if len(timestamps) > 0:
        time_minutes = [(t - timestamps[0]) / 60 for t in timestamps]
    else:
        time_minutes = list(range(len(hr)))

    # Plot HR
    if len(hr) > 0:
        ax_hr.plot(time_minutes[:len(hr)], hr, linewidth=1.5, alpha=0.8)
        ax_hr.set_ylabel('Heart Rate (BPM)', fontsize=10)
        ax_hr.grid(True, alpha=0.3)
        ax_hr.set_title(f"{title_prefix}Heart Rate", fontsize=11, fontweight='bold')

    # Plot Speed
    if len(speed) > 0:
        ax_speed.plot(time_minutes[:len(speed)], speed, linewidth=1.5, alpha=0.8, color='orange')
        ax_speed.set_ylabel('Speed (km/h)', fontsize=10)
        ax_speed.grid(True, alpha=0.3)
        ax_speed.set_title(f"{title_prefix}Speed", fontsize=11, fontweight='bold')

    # Plot Altitude
    if len(altitude) > 0:
        ax_alt.plot(time_minutes[:len(altitude)], altitude, linewidth=1.5, alpha=0.8, color='green')
        ax_alt.set_ylabel('Altitude (m)', fontsize=10)
        ax_alt.set_xlabel('Time (minutes)', fontsize=10)
        ax_alt.grid(True, alpha=0.3)
        ax_alt.set_title(f"{title_prefix}Altitude", fontsize=11, fontweight='bold')


def main():
    parser = argparse.ArgumentParser(description='View smoothed workout data')
    parser.add_argument('--index', '-i', type=int, default=0,
                        help='Workout index to view (default: 0)')
    parser.add_argument('--smoothed', '-s', type=str,
                        default='Preprocessing/clean_dataset_smoothed.json',
                        help='Smoothed data file')
    parser.add_argument('--original', type=str,
                        default='Preprocessing/complete_dataset.json',
                        help='Original data file (for comparison)')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Show before/after comparison')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of displaying')

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    smoothed_file = project_root / args.smoothed
    original_file = project_root / args.original

    # Load smoothed data
    print(f"Loading smoothed data from {smoothed_file.name}...")
    with open(smoothed_file, 'r') as f:
        smoothed_data = json.load(f)

    if args.index >= len(smoothed_data['workouts']):
        print(f"Error: Index {args.index} out of range (max: {len(smoothed_data['workouts'])-1})")
        return

    smoothed_workout = smoothed_data['workouts'][args.index]

    # Print workout info
    print()
    print("=" * 70)
    print(f"WORKOUT #{args.index}")
    print("=" * 70)
    print(f"ID: {smoothed_workout.get('workout_id', 'N/A')}")
    print(f"User ID: {smoothed_workout.get('user_id', 'N/A')}")
    print(f"Type: {smoothed_workout.get('workout_type', 'N/A')}")
    print(f"Duration: {smoothed_workout.get('duration_min', 0):.1f} min")
    print(f"Data points: {len(smoothed_workout.get('heart_rate', []))}")

    if 'heart_rate' in smoothed_workout:
        hr = smoothed_workout['heart_rate']
        print(f"HR: {np.mean(hr):.1f} ± {np.std(hr):.1f} BPM (range: {np.min(hr):.0f}-{np.max(hr):.0f})")

    if 'speed' in smoothed_workout and len(smoothed_workout['speed']) > 0:
        speed = smoothed_workout['speed']
        print(f"Speed: {np.mean(speed):.1f} ± {np.std(speed):.1f} km/h (max: {np.max(speed):.1f})")

    print()

    # Create plot
    if args.compare and original_file.exists():
        # Load original data for comparison
        print(f"Loading original data from {original_file.name}...")
        with open(original_file, 'r') as f:
            original_data = json.load(f)

        if args.index < len(original_data['workouts']):
            original_workout = original_data['workouts'][args.index]

            # Create side-by-side comparison
            fig, axes = plt.subplots(3, 2, figsize=(16, 10))
            fig.suptitle(f'Workout #{args.index} - Before/After Smoothing Comparison',
                        fontsize=14, fontweight='bold')

            # Original (left column)
            plot_workout(original_workout, axes[0, 0], axes[1, 0], axes[2, 0],
                        title_prefix="ORIGINAL - ")

            # Smoothed (right column)
            plot_workout(smoothed_workout, axes[0, 1], axes[1, 1], axes[2, 1],
                        title_prefix="SMOOTHED - ")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
        else:
            print("Warning: Original data doesn't have this workout. Showing smoothed only.")
            args.compare = False

    if not args.compare:
        # Single plot (smoothed only)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Workout #{args.index} - Smoothed Data (Window=6)',
                    fontsize=14, fontweight='bold')

        plot_workout(smoothed_workout, axes[0], axes[1], axes[2])

        plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save or show
    if args.save:
        save_path = project_root / args.save
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        print("Displaying plot... (close window to exit)")
        plt.show()


if __name__ == "__main__":
    main()
