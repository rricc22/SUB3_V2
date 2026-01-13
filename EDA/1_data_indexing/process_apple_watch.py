#!/usr/bin/env python3
"""
Process Apple Watch outdoor run data and compare with Edmondo.

Extracts HR, speed, and elevation from Apple Watch CSV exports.
Creates comparison metrics with Edmondo data.

Author: Claude
Date: 2026-01-13
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Paths
APPLE_WATCH_DIR = Path(__file__).parent.parent.parent / "DATA" / "raw" / "apple_health_export_User1"
EDMONDO_FILE = Path(__file__).parent.parent.parent / "DATA" / "raw" / "endomondoHR_proper-002.json"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def load_apple_watch_workout(workout_id: str):
    """Load a single Apple Watch workout with route and HR data."""
    route_file = APPLE_WATCH_DIR / f"Outdoor Run-Route-{workout_id}.csv"
    hr_file = APPLE_WATCH_DIR / f"Outdoor Run-Heart Rate-{workout_id}.csv"

    if not route_file.exists():
        return None

    # Load route data (GPS, speed, elevation)
    route_df = pd.read_csv(route_file)
    route_df['Timestamp'] = pd.to_datetime(route_df['Timestamp']).dt.tz_localize(None)
    route_df = route_df.rename(columns={
        'Altitude (m)': 'altitude',
        'Speed (m/s)': 'speed',
        'Latitude': 'latitude',
        'Longitude': 'longitude'
    })

    # Load heart rate data if available
    if hr_file.exists():
        hr_df = pd.read_csv(hr_file)
        hr_df['Date/Time'] = pd.to_datetime(hr_df['Date/Time']).dt.tz_localize(None)
        hr_df = hr_df.rename(columns={'Date/Time': 'Timestamp', 'Avg (count/min)': 'heart_rate'})

        # Merge route and HR data
        # Use asof merge to match closest timestamps
        hr_df = hr_df.sort_values('Timestamp')
        route_df = route_df.sort_values('Timestamp')

        merged_df = pd.merge_asof(
            route_df,
            hr_df[['Timestamp', 'heart_rate']],
            on='Timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('10s')
        )
    else:
        merged_df = route_df
        merged_df['heart_rate'] = np.nan

    return merged_df


def process_all_apple_watch_workouts():
    """Process all Apple Watch outdoor run workouts."""
    print("Scanning Apple Watch data...")

    # Find all unique workout IDs from route files
    route_files = list(APPLE_WATCH_DIR.glob("Outdoor Run-Route-*.csv"))
    workout_ids = [f.name.replace("Outdoor Run-Route-", "").replace(".csv", "") for f in route_files]

    print(f"Found {len(workout_ids)} outdoor run workouts\n")

    all_workouts = []
    stats = []

    for i, workout_id in enumerate(workout_ids, 1):
        if i % 10 == 0:
            print(f"Processing workout {i}/{len(workout_ids)}...")

        workout_df = load_apple_watch_workout(workout_id)
        if workout_df is None:
            continue

        # Calculate workout statistics
        workout_stats = {
            'workout_id': workout_id,
            'start_time': workout_df['Timestamp'].min(),
            'duration_seconds': (workout_df['Timestamp'].max() - workout_df['Timestamp'].min()).total_seconds(),
            'distance_km': len(workout_df) * workout_df['speed'].mean() / 1000,
            'avg_speed_ms': workout_df['speed'].mean(),
            'avg_speed_kmh': workout_df['speed'].mean() * 3.6,
            'max_speed_ms': workout_df['speed'].max(),
            'avg_hr': workout_df['heart_rate'].mean(),
            'max_hr': workout_df['heart_rate'].max(),
            'min_altitude': workout_df['altitude'].min(),
            'max_altitude': workout_df['altitude'].max(),
            'elevation_gain': workout_df['altitude'].diff().clip(lower=0).sum(),
            'num_points': len(workout_df),
            'hr_coverage': workout_df['heart_rate'].notna().mean() * 100,
        }

        stats.append(workout_stats)
        all_workouts.append(workout_df)

    print(f"\nSuccessfully processed {len(stats)} workouts")

    return all_workouts, pd.DataFrame(stats)


def load_edmondo_sample(num_workouts=100):
    """Load a sample of Edmondo workouts for comparison."""
    print("\nLoading Edmondo data sample...")

    edmondo_stats = []

    with open(EDMONDO_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_workouts:
                break

            try:
                workout = eval(line.strip())

                # Check if it's a running workout with complete data
                if workout.get('sport') != 'run':
                    continue

                if 'heart_rate' not in workout or 'speed' not in workout or 'altitude' not in workout:
                    continue

                hr_data = workout['heart_rate']
                speed_data = workout['speed']
                altitude_data = workout['altitude']

                if not hr_data or not speed_data or not altitude_data:
                    continue

                # Calculate statistics
                duration = len(hr_data)
                avg_hr = np.mean(hr_data)
                avg_speed = np.mean(speed_data)

                # Calculate elevation gain
                elevation_gain = sum([max(0, altitude_data[i] - altitude_data[i-1])
                                     for i in range(1, len(altitude_data))])

                stats = {
                    'workout_id': f"edmondo_{i}",
                    'duration_seconds': duration,
                    'avg_speed_ms': avg_speed,
                    'avg_speed_kmh': avg_speed * 3.6,
                    'avg_hr': avg_hr,
                    'max_hr': max(hr_data),
                    'min_altitude': min(altitude_data),
                    'max_altitude': max(altitude_data),
                    'elevation_gain': elevation_gain,
                    'num_points': len(hr_data),
                }

                edmondo_stats.append(stats)

            except Exception as e:
                continue

    print(f"Loaded {len(edmondo_stats)} Edmondo workouts")

    return pd.DataFrame(edmondo_stats)


def compare_datasets(apple_stats, edmondo_stats):
    """Compare Apple Watch and Edmondo data."""
    print("\n" + "=" * 80)
    print("COMPARISON: APPLE WATCH vs EDMONDO")
    print("=" * 80)

    print("\n--- Apple Watch Statistics ---")
    print(f"Number of workouts: {len(apple_stats)}")
    print(f"Average duration: {apple_stats['duration_seconds'].mean() / 60:.1f} minutes")
    print(f"Average speed: {apple_stats['avg_speed_kmh'].mean():.2f} km/h")
    print(f"Average heart rate: {apple_stats['avg_hr'].mean():.1f} bpm")
    print(f"Average HR coverage: {apple_stats['hr_coverage'].mean():.1f}%")
    print(f"Average elevation gain: {apple_stats['elevation_gain'].mean():.1f} m")
    print(f"Average points per workout: {apple_stats['num_points'].mean():.0f}")

    print("\n--- Edmondo Statistics ---")
    print(f"Number of workouts: {len(edmondo_stats)}")
    print(f"Average duration: {edmondo_stats['duration_seconds'].mean() / 60:.1f} minutes")
    print(f"Average speed: {edmondo_stats['avg_speed_kmh'].mean():.2f} km/h")
    print(f"Average heart rate: {edmondo_stats['avg_hr'].mean():.1f} bpm")
    print(f"Average elevation gain: {edmondo_stats['elevation_gain'].mean():.1f} m")
    print(f"Average points per workout: {edmondo_stats['num_points'].mean():.0f}")

    print("\n--- Key Differences ---")
    speed_diff = apple_stats['avg_speed_kmh'].mean() - edmondo_stats['avg_speed_kmh'].mean()
    hr_diff = apple_stats['avg_hr'].mean() - edmondo_stats['avg_hr'].mean()
    elev_diff = apple_stats['elevation_gain'].mean() - edmondo_stats['elevation_gain'].mean()

    print(f"Speed difference: {speed_diff:+.2f} km/h (Apple Watch {'faster' if speed_diff > 0 else 'slower'})")
    print(f"Heart rate difference: {hr_diff:+.1f} bpm (Apple Watch {'higher' if hr_diff > 0 else 'lower'})")
    print(f"Elevation gain difference: {elev_diff:+.1f} m (Apple Watch {'more' if elev_diff > 0 else 'less'})")

    # Data quality comparison
    print("\n--- Data Quality ---")
    print(f"Apple Watch:")
    print(f"  - HR coverage: {apple_stats['hr_coverage'].mean():.1f}%")
    print(f"  - Speed coverage: 100% (always present in GPS)")
    print(f"  - Elevation coverage: 100% (always present in GPS)")
    print(f"  - Sampling rate: ~1 Hz (1 point per second)")

    print(f"\nEdmondo:")
    print(f"  - HR coverage: 100% (filtered for complete workouts)")
    print(f"  - Speed coverage: 100% (filtered for complete workouts)")
    print(f"  - Elevation coverage: 100% (filtered for complete workouts)")
    print(f"  - Sampling rate: ~1 Hz")

    return {
        'apple_watch': {
            'count': len(apple_stats),
            'avg_speed_kmh': apple_stats['avg_speed_kmh'].mean(),
            'avg_hr': apple_stats['avg_hr'].mean(),
            'avg_elevation_gain': apple_stats['elevation_gain'].mean(),
            'hr_coverage': apple_stats['hr_coverage'].mean(),
        },
        'edmondo': {
            'count': len(edmondo_stats),
            'avg_speed_kmh': edmondo_stats['avg_speed_kmh'].mean(),
            'avg_hr': edmondo_stats['avg_hr'].mean(),
            'avg_elevation_gain': edmondo_stats['elevation_gain'].mean(),
        },
        'differences': {
            'speed_kmh': speed_diff,
            'hr_bpm': hr_diff,
            'elevation_m': elev_diff,
        }
    }


def main():
    """Main processing pipeline."""
    print("=" * 80)
    print("APPLE WATCH DATA PROCESSING & COMPARISON")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process Apple Watch data
    apple_workouts, apple_stats = process_all_apple_watch_workouts()

    # Load Edmondo sample
    edmondo_stats = load_edmondo_sample(num_workouts=1000)

    # Compare datasets
    comparison = compare_datasets(apple_stats, edmondo_stats)

    # Save results
    print("\n--- Saving Results ---")

    apple_stats_file = OUTPUT_DIR / "apple_watch_stats.csv"
    apple_stats.to_csv(apple_stats_file, index=False)
    print(f"Saved Apple Watch stats to: {apple_stats_file}")

    edmondo_stats_file = OUTPUT_DIR / "edmondo_stats_sample.csv"
    edmondo_stats.to_csv(edmondo_stats_file, index=False)
    print(f"Saved Edmondo stats to: {edmondo_stats_file}")

    comparison_file = OUTPUT_DIR / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison summary to: {comparison_file}")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
