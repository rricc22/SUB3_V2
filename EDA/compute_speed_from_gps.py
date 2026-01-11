#!/usr/bin/env python3
"""
Compute speed from GPS coordinates for workouts missing speed data.

Uses Haversine formula for distance calculation with robust validation:
- Outlier detection and capping
- Smoothing to reduce GPS noise
- Sanity checks for running speeds

Author: OpenCode
Date: 2025-01-11
"""

import ast
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from time import time
import json

# Constants
DATA_FILE = Path(__file__).parent.parent / "DATA" / "raw" / "endomondoHR_proper.json"
INDEX_DIR = Path(__file__).parent.parent / "DATA" / "indices"
OUTPUT_DIR = Path(__file__).parent.parent / "DATA" / "processed"

# Speed limits for running (km/h)
MIN_SPEED = 0.0          # Can be standing still
MAX_RUNNING_SPEED = 25.0  # World record marathon pace is ~21 km/h, give some margin
MAX_REASONABLE_SPEED = 30.0  # Above this is definitely GPS error

# Earth radius in km
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def haversine_distance_vectorized(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate distances between consecutive GPS points.
    
    Args:
        lat: Array of latitudes in degrees
        lon: Array of longitudes in degrees
    
    Returns:
        Array of distances in kilometers (length = len(lat) - 1)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    
    a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical issues
    
    return EARTH_RADIUS_KM * c


def compute_speed_from_gps(
    lat: np.ndarray, 
    lon: np.ndarray, 
    timestamps: np.ndarray,
    smooth_window: int = 3
) -> Tuple[np.ndarray, Dict]:
    """
    Compute speed from GPS coordinates with validation and smoothing.
    
    Args:
        lat: Latitude array in degrees
        lon: Longitude array in degrees
        timestamps: Unix timestamps
        smooth_window: Window size for moving average smoothing
    
    Returns:
        Tuple of (speed array in km/h, quality metrics dict)
    """
    n = len(lat)
    
    # Calculate distances between consecutive points
    distances = haversine_distance_vectorized(lat, lon)  # km
    
    # Calculate time differences
    dt = np.diff(timestamps).astype(float)  # seconds
    
    # Avoid division by zero
    dt = np.maximum(dt, 0.1)
    
    # Raw speed in km/h
    raw_speed = (distances / dt) * 3600
    
    # Pad to original length (first point gets speed of second point)
    raw_speed = np.concatenate([[raw_speed[0]], raw_speed])
    
    # Quality metrics before cleaning
    metrics = {
        'raw_max': float(np.max(raw_speed)),
        'raw_mean': float(np.mean(raw_speed)),
        'outliers_above_30': int(np.sum(raw_speed > MAX_REASONABLE_SPEED)),
        'outliers_above_25': int(np.sum(raw_speed > MAX_RUNNING_SPEED)),
        'zeros': int(np.sum(raw_speed == 0)),
    }
    
    # Step 1: Cap extreme outliers (GPS jumps)
    speed = np.clip(raw_speed, MIN_SPEED, MAX_REASONABLE_SPEED)
    
    # Step 2: Apply moving average smoothing to reduce GPS noise
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        # Pad edges to avoid boundary effects
        padded = np.pad(speed, (smooth_window//2, smooth_window//2), mode='edge')
        speed = np.convolve(padded, kernel, mode='valid')[:n]
    
    # Step 3: Final sanity cap at max running speed
    speed = np.clip(speed, MIN_SPEED, MAX_RUNNING_SPEED)
    
    # Update metrics after cleaning
    metrics['clean_max'] = float(np.max(speed))
    metrics['clean_mean'] = float(np.mean(speed))
    metrics['clean_std'] = float(np.std(speed))
    
    return speed, metrics


def validate_workout_for_speed_computation(workout: Dict) -> Tuple[bool, str]:
    """
    Check if a workout has valid GPS data for speed computation.
    
    Returns:
        Tuple of (is_valid, reason)
    """
    # Must have required fields
    required = ['latitude', 'longitude', 'timestamp', 'heart_rate']
    for field in required:
        if field not in workout:
            return False, f"Missing {field}"
    
    lat = np.array(workout['latitude'])
    lon = np.array(workout['longitude'])
    ts = np.array(workout['timestamp'])
    
    # Must have enough points
    if len(lat) < 10:
        return False, "Too few points"
    
    # Check for constant GPS (device error)
    if np.std(lat) < 0.0001 or np.std(lon) < 0.0001:
        return False, "GPS appears stuck"
    
    # Check timestamps are monotonic
    if not np.all(np.diff(ts) > 0):
        return False, "Timestamps not monotonic"
    
    # Check for reasonable duration (at least 5 minutes)
    duration_min = (ts[-1] - ts[0]) / 60
    if duration_min < 5:
        return False, "Duration too short"
    
    return True, "OK"


def process_workout(line: str, line_num: int) -> Optional[Dict]:
    """
    Process a single workout line and compute speed.
    
    Returns:
        Dict with line_num and computed speed, or None if invalid
    """
    try:
        workout = ast.literal_eval(line.strip())
    except:
        return None
    
    # Skip if already has speed
    if 'speed' in workout:
        return None
    
    # Validate
    is_valid, reason = validate_workout_for_speed_computation(workout)
    if not is_valid:
        return {'line_num': line_num, 'status': 'invalid', 'reason': reason}
    
    # Compute speed
    lat = np.array(workout['latitude'])
    lon = np.array(workout['longitude'])
    ts = np.array(workout['timestamp'])
    
    speed, metrics = compute_speed_from_gps(lat, lon, ts, smooth_window=3)
    
    # Quality check on computed speed
    if metrics['clean_mean'] < 2.0:  # Less than 2 km/h average - probably not running
        return {'line_num': line_num, 'status': 'invalid', 'reason': 'Speed too low for running'}
    
    if metrics['outliers_above_30'] > len(speed) * 0.1:  # More than 10% outliers
        return {'line_num': line_num, 'status': 'invalid', 'reason': 'Too many GPS outliers'}
    
    return {
        'line_num': line_num,
        'status': 'valid',
        'speed': speed.tolist(),
        'metrics': metrics
    }


def main():
    """Process all running workouts without speed."""
    
    # Load index of running workouts without speed
    index_file = INDEX_DIR / "running_no_speed.txt"
    with open(index_file, 'r') as f:
        line_numbers = [int(l.strip()) for l in f.readlines()]
    
    print(f"Processing {len(line_numbers):,} running workouts without speed...")
    print(f"Data file: {DATA_FILE}")
    print()
    
    # Output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_output = OUTPUT_DIR / "running_computed_speed.jsonl"
    invalid_output = OUTPUT_DIR / "running_invalid_gps.jsonl"
    new_index = INDEX_DIR / "running_computed_speed.txt"
    
    start_time = time()
    
    valid_count = 0
    invalid_count = 0
    invalid_reasons = {}
    
    # Process line by line (memory efficient)
    with open(DATA_FILE, 'r') as f_in, \
         open(valid_output, 'w') as f_valid, \
         open(invalid_output, 'w') as f_invalid, \
         open(new_index, 'w') as f_index:
        
        current_line = 0
        line_idx = 0  # Index into line_numbers
        
        for line in f_in:
            current_line += 1
            
            # Skip lines not in our index
            if line_idx >= len(line_numbers):
                break
            if current_line != line_numbers[line_idx]:
                continue
            
            line_idx += 1
            
            # Process this workout
            result = process_workout(line, current_line)
            
            if result is None:
                continue
            
            if result['status'] == 'valid':
                # Save computed speed
                f_valid.write(json.dumps({
                    'line_num': result['line_num'],
                    'speed': result['speed'],
                    'metrics': result['metrics']
                }) + '\n')
                f_index.write(f"{result['line_num']}\n")
                valid_count += 1
            else:
                # Save invalid reason
                f_invalid.write(json.dumps({
                    'line_num': result['line_num'],
                    'reason': result['reason']
                }) + '\n')
                invalid_count += 1
                reason = result['reason']
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            
            # Progress
            if line_idx % 5000 == 0:
                elapsed = time() - start_time
                rate = line_idx / elapsed
                print(f"  Processed {line_idx:,}/{len(line_numbers):,} "
                      f"({rate:.0f}/sec) - Valid: {valid_count:,}, Invalid: {invalid_count:,}")
    
    elapsed = time() - start_time
    
    # Summary
    print()
    print("=" * 60)
    print("SPEED COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nProcessed: {line_idx:,} workouts in {elapsed:.1f} seconds")
    print(f"Valid (speed computed): {valid_count:,} ({valid_count/line_idx*100:.1f}%)")
    print(f"Invalid (bad GPS): {invalid_count:,} ({invalid_count/line_idx*100:.1f}%)")
    
    print("\n--- Invalid Reasons ---")
    for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}")
    
    print(f"\n--- Output Files ---")
    print(f"  Valid speeds: {valid_output}")
    print(f"  Invalid list: {invalid_output}")
    print(f"  New index: {new_index}")
    
    # Update summary
    summary_path = INDEX_DIR / "summary.txt"
    with open(summary_path, 'a') as f:
        f.write(f"\n--- Speed Computation (computed from GPS) ---\n")
        f.write(f"Running (computed speed): {valid_count}\n")
        f.write(f"Running (invalid GPS): {invalid_count}\n")
    
    print(f"\nTotal running workouts with speed: {11532 + valid_count:,}")
    print(f"  - Original (had speed): 11,532")
    print(f"  - Computed from GPS: {valid_count:,}")


if __name__ == "__main__":
    main()
