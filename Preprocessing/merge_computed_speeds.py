#!/usr/bin/env python3
"""
Merge GPS-computed speeds into the clean dataset.

Takes workouts from clean_dataset.json and adds computed speeds from
running_computed_speed.jsonl where speed data is missing or incomplete.

Author: Claude Code
Date: 2026-01-13
"""

import json
from pathlib import Path
from datetime import datetime
import argparse


def load_computed_speeds(jsonl_file: Path) -> dict:
    """
    Load computed speeds from JSONL file into a dictionary.

    Args:
        jsonl_file: Path to running_computed_speed.jsonl

    Returns:
        Dictionary mapping line_number -> computed speed data
    """
    print(f"Loading computed speeds from {jsonl_file.name}...")
    computed_speeds = {}

    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if (i + 1) % 10000 == 0:
                print(f"  Loaded {i+1:,} computed speed records...")

            data = json.loads(line.strip())
            line_num = data['line_num']
            computed_speeds[line_num] = {
                'speed': data['speed'],
                'metrics': data.get('metrics', {})
            }

    print(f"  Total computed speeds loaded: {len(computed_speeds):,}")
    return computed_speeds


def merge_speeds(workout: dict, computed_speeds: dict) -> dict:
    """
    Merge computed speed into a workout if needed.

    Args:
        workout: Workout dictionary from clean_dataset.json
        computed_speeds: Dictionary of computed speeds by line_number

    Returns:
        Updated workout dictionary with speed data
    """
    line_num = workout.get('line_number')

    # Check if workout needs computed speed
    needs_speed = False

    if 'speed' not in workout:
        needs_speed = True
    elif workout['speed'] is None or len(workout['speed']) == 0:
        needs_speed = True

    # If needs speed and we have computed speed for this line
    if needs_speed and line_num in computed_speeds:
        workout['speed'] = computed_speeds[line_num]['speed']
        workout['speed_source'] = 'GPS_computed'
        workout['speed_metrics'] = computed_speeds[line_num]['metrics']
        return workout, True

    # If has speed already, just mark the source
    if not needs_speed and 'speed_source' not in workout:
        workout['speed_source'] = 'original'

    return workout, False


def main():
    parser = argparse.ArgumentParser(
        description='Merge GPS-computed speeds into clean dataset'
    )
    parser.add_argument(
        '--clean', '-c',
        type=str,
        default='Preprocessing/clean_dataset.json',
        help='Input clean dataset (default: Preprocessing/clean_dataset.json)'
    )
    parser.add_argument(
        '--computed', '-s',
        type=str,
        default='DATA/processed/running_computed_speed.jsonl',
        help='Computed speeds JSONL file (default: DATA/processed/running_computed_speed.jsonl)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='Preprocessing/clean_dataset_merged.json',
        help='Output merged dataset (default: Preprocessing/clean_dataset_merged.json)'
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    clean_file = project_root / args.clean
    computed_file = project_root / args.computed
    output_file = project_root / args.output

    print("=" * 70)
    print("MERGE COMPUTED SPEEDS INTO CLEAN DATASET")
    print("=" * 70)
    print(f"Clean dataset:    {clean_file}")
    print(f"Computed speeds:  {computed_file}")
    print(f"Output:           {output_file}")
    print()

    # Load computed speeds
    computed_speeds = load_computed_speeds(computed_file)
    print()

    # Load clean dataset
    print(f"Loading clean dataset from {clean_file.name}...")
    with open(clean_file, 'r') as f:
        data = json.load(f)

    total_workouts = len(data['workouts'])
    print(f"  Loaded {total_workouts:,} workouts")
    print()

    # Merge speeds
    print("Merging computed speeds...")
    merged_count = 0
    already_had_speed = 0
    no_computed_speed = 0

    for i, workout in enumerate(data['workouts']):
        if (i + 1) % 1000 == 0:
            progress = (i + 1) / total_workouts * 100
            print(f"  Progress: {i+1:,}/{total_workouts:,} ({progress:.1f}%)")

        workout, was_merged = merge_speeds(workout, computed_speeds)

        if was_merged:
            merged_count += 1
        elif 'speed' in workout and workout['speed'] and len(workout['speed']) > 0:
            already_had_speed += 1
        else:
            no_computed_speed += 1

    print(f"  Completed: {total_workouts:,} workouts processed")
    print()

    # Update metadata
    if 'metadata' not in data:
        data['metadata'] = {}

    data['metadata']['merge_info'] = {
        'timestamp': datetime.now().isoformat(),
        'computed_speeds_added': merged_count,
        'original_speeds': already_had_speed,
        'no_speed_available': no_computed_speed,
        'computed_speeds_file': str(computed_file.name)
    }

    # Save merged dataset
    print("Saving merged dataset...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Print file sizes
    clean_size_mb = clean_file.stat().st_size / (1024 * 1024)
    output_size_mb = output_file.stat().st_size / (1024 * 1024)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total workouts:           {total_workouts:,}")
    print(f"  Already had speed:      {already_had_speed:,} ({already_had_speed/total_workouts*100:.1f}%)")
    print(f"  Computed speed added:   {merged_count:,} ({merged_count/total_workouts*100:.1f}%)")
    print(f"  No speed available:     {no_computed_speed:,} ({no_computed_speed/total_workouts*100:.1f}%)")
    print()
    print(f"Input file size:  {clean_size_mb:.1f} MB")
    print(f"Output file size: {output_size_mb:.1f} MB")
    print()
    print(f"Merged dataset saved to: {output_file}")
    print()
    print("All workouts now have 'speed_source' field:")
    print("  - 'original': Speed was already in clean dataset")
    print("  - 'GPS_computed': Speed was computed from GPS coordinates")


if __name__ == "__main__":
    main()
