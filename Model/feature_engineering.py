"""
Feature Engineering Module for SUB3_V2

Implements 11 features for heart rate prediction:
- Base (3): speed, altitude, gender
- Lag (3): speed_lag_2, speed_lag_5, altitude_lag_30
- Derivatives (2): speed_derivative, altitude_derivative
- Rolling (2): rolling_speed_10, rolling_speed_30
- Cumulative (1): cumulative_elevation_gain

Author: Riccardo
Date: 2026-01-13
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def create_lag_feature(data: np.ndarray, lag: int) -> np.ndarray:
    """
    Create a lag feature by shifting data forward.

    Args:
        data: 1D array of values
        lag: Number of timesteps to lag

    Returns:
        Lagged array with first 'lag' values forward-filled from first valid value
    """
    lagged = np.empty_like(data)
    lagged[:lag] = data[0]  # Forward fill with first value
    lagged[lag:] = data[:-lag]
    return lagged


def compute_derivative(data: np.ndarray) -> np.ndarray:
    """
    Compute derivative (rate of change) of a signal.

    Args:
        data: 1D array of values

    Returns:
        Array of differences, first value is 0
    """
    derivative = np.zeros_like(data)
    derivative[1:] = np.diff(data)
    return derivative


def compute_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean using pandas for efficiency.

    Args:
        data: 1D array of values
        window: Window size for rolling mean

    Returns:
        Array with rolling mean, first 'window-1' values forward-filled
    """
    if len(data) < window:
        # If sequence is shorter than window, return mean of all data
        return np.full_like(data, data.mean(), dtype=float)

    series = pd.Series(data)
    rolling = series.rolling(window=window, min_periods=1).mean()
    return rolling.values


def compute_cumulative_elevation_gain(altitude: np.ndarray) -> np.ndarray:
    """
    Compute cumulative elevation gain (only positive altitude changes).

    Args:
        altitude: 1D array of altitude values in meters

    Returns:
        Cumulative elevation gain array
    """
    altitude_diff = np.zeros_like(altitude)
    altitude_diff[1:] = np.diff(altitude)

    # Only keep positive changes (elevation gain)
    elevation_gain = np.maximum(altitude_diff, 0)

    # Cumulative sum
    cumulative_gain = np.cumsum(elevation_gain)
    return cumulative_gain


def engineer_features(workout: Dict) -> np.ndarray:
    """
    Engineer all 11 features from a workout dictionary.

    Args:
        workout: Dictionary containing:
            - 'speed': List of speed values in km/h
            - 'altitude': List of altitude values in meters
            - 'gender': Binary (1.0 for male, 0.0 for female)

    Returns:
        Array of shape [seq_len, 11] with all features:
            0: speed
            1: altitude
            2: gender (repeated for all timesteps)
            3: speed_lag_2
            4: speed_lag_5
            5: altitude_lag_30
            6: speed_derivative
            7: altitude_derivative
            8: rolling_speed_10
            9: rolling_speed_30
            10: cumulative_elevation_gain
    """
    # Extract base features
    speed = np.array(workout['speed'], dtype=np.float32)
    altitude = np.array(workout['altitude'], dtype=np.float32)
    gender = float(workout.get('gender', 1.0))  # Default to male if missing

    seq_len = len(speed)

    # Validate input
    assert len(altitude) == seq_len, "Speed and altitude must have same length"

    # Initialize feature array
    features = np.zeros((seq_len, 11), dtype=np.float32)

    # Base features
    features[:, 0] = speed
    features[:, 1] = altitude
    features[:, 2] = gender  # Broadcast to all timesteps

    # Lag features
    features[:, 3] = create_lag_feature(speed, lag=2)
    features[:, 4] = create_lag_feature(speed, lag=5)
    features[:, 5] = create_lag_feature(altitude, lag=30)

    # Derivative features
    features[:, 6] = compute_derivative(speed)
    features[:, 7] = compute_derivative(altitude)

    # Rolling mean features
    features[:, 8] = compute_rolling_mean(speed, window=10)
    features[:, 9] = compute_rolling_mean(speed, window=30)

    # Cumulative elevation gain
    features[:, 10] = compute_cumulative_elevation_gain(altitude)

    return features


def get_feature_names() -> List[str]:
    """
    Get names of all 11 features in order.

    Returns:
        List of feature names
    """
    return [
        'speed',
        'altitude',
        'gender',
        'speed_lag_2',
        'speed_lag_5',
        'altitude_lag_30',
        'speed_derivative',
        'altitude_derivative',
        'rolling_speed_10',
        'rolling_speed_30',
        'cumulative_elevation_gain'
    ]


def validate_features(features: np.ndarray) -> bool:
    """
    Validate engineered features for common issues.

    Args:
        features: Array of shape [seq_len, 11]

    Returns:
        True if valid, raises AssertionError otherwise
    """
    assert features.shape[1] == 11, f"Expected 11 features, got {features.shape[1]}"
    assert not np.any(np.isnan(features)), "Features contain NaN values"
    assert not np.any(np.isinf(features)), "Features contain Inf values"

    # Check reasonable ranges
    speed = features[:, 0]
    altitude = features[:, 1]
    gender = features[:, 2]

    assert np.all(speed >= 0) and np.all(speed < 50), "Speed out of reasonable range [0, 50] km/h"
    assert np.all(altitude >= -500) and np.all(altitude < 10000), "Altitude out of reasonable range [-500, 10000] m"
    assert np.all((gender == 0) | (gender == 1)), "Gender must be binary (0 or 1)"

    return True


if __name__ == "__main__":
    # Test on a sample workout
    print("Testing feature engineering...")

    # Create synthetic workout
    test_workout = {
        'speed': [10.0] * 100 + [12.0] * 100 + [8.0] * 100,  # 300 timesteps
        'altitude': list(range(100, 400)),  # Increasing altitude
        'gender': 1.0,
        'heart_rate': [150] * 300  # Not used in feature engineering
    }

    # Engineer features
    features = engineer_features(test_workout)

    print(f"✓ Feature shape: {features.shape}")
    print(f"✓ Expected: (300, 11)")
    print(f"\nFeature statistics:")

    for i, name in enumerate(get_feature_names()):
        values = features[:, i]
        print(f"  {name:25s}: min={values.min():8.2f}, max={values.max():8.2f}, "
              f"mean={values.mean():8.2f}, std={values.std():8.2f}")

    # Validate
    try:
        validate_features(features)
        print("\n✓ All validation checks passed!")
    except AssertionError as e:
        print(f"\n✗ Validation failed: {e}")
