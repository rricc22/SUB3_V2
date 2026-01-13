"""
Prepare Sequences for SUB3_V2 Training

Implements the V1 methodology with V2 improvements:
1. Load clean_dataset_smoothed.json
2. Load gender from raw data using line_number
3. Engineer 11 features (vs 3 in V1)
4. Pad/truncate to 500 timesteps with mask generation (V2 improvement)
5. User-based stratified split (70/15/15)
6. Normalize features (fit on train only, HR unnormalized)
7. Save tensors: train.pt, val.pt, test.pt

Author: Riccardo
Date: 2026-01-13
"""

import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import argparse
from tqdm import tqdm
import linecache
import ast
from typing import Dict, List, Tuple

# Import feature engineering module
from feature_engineering import engineer_features, get_feature_names


# Configuration
RANDOM_SEED = 42
SEQ_LENGTH = 500
MIN_SEQ_LENGTH = 50


def load_gender_from_raw(line_number: int, raw_data_path: str) -> float:
    """
    Load gender from raw data using line number.

    Args:
        line_number: Line number in raw data file (1-indexed)
        raw_data_path: Path to raw data file

    Returns:
        1.0 for male, 0.0 for female, 1.0 if missing
    """
    try:
        line = linecache.getline(raw_data_path, line_number)
        if not line.strip():
            return 1.0  # Default to male if line not found

        workout = ast.literal_eval(line.strip())
        gender_str = workout.get('gender', 'male').lower()

        return 1.0 if gender_str == 'male' else 0.0

    except Exception as e:
        # If any error, default to male
        return 1.0


def pad_or_truncate_with_mask(
    features: np.ndarray,
    heart_rate: np.ndarray,
    target_length: int = SEQ_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Pad or truncate sequences to target length and generate mask.

    Args:
        features: Array of shape [seq_len, 11]
        heart_rate: Array of shape [seq_len, 1]
        target_length: Target sequence length

    Returns:
        Tuple of (features, heart_rate, mask, original_length)
        - features: [target_length, 11]
        - heart_rate: [target_length, 1]
        - mask: [target_length, 1] (1=valid, 0=padded)
        - original_length: Length before padding
    """
    seq_len = len(features)
    original_length = seq_len

    # Create mask
    mask = np.zeros((target_length, 1), dtype=np.float32)

    if seq_len >= target_length:
        # Truncate
        features = features[:target_length]
        heart_rate = heart_rate[:target_length]
        mask[:] = 1.0  # All valid
        original_length = target_length
    else:
        # Pad with last value
        padding_features = np.tile(features[-1:], (target_length - seq_len, 1))
        features = np.vstack([features, padding_features])

        padding_hr = np.tile(heart_rate[-1:], (target_length - seq_len, 1))
        heart_rate = np.vstack([heart_rate, padding_hr])

        # Mask: 1 for valid, 0 for padded
        mask[:seq_len] = 1.0
        mask[seq_len:] = 0.0

    return features, heart_rate, mask, original_length


def prepare_sequences(
    input_file: str,
    raw_data_path: str,
    output_dir: str,
    verbose: bool = True
):
    """
    Main preprocessing pipeline.

    Args:
        input_file: Path to clean_dataset_smoothed.json
        raw_data_path: Path to raw Endomondo data (for gender)
        output_dir: Directory to save processed tensors
        verbose: Print progress
    """
    np.random.seed(RANDOM_SEED)

    # Load cleaned data
    if verbose:
        print(f"Loading {input_file}...")

    with open(input_file, 'r') as f:
        data = json.load(f)

    workouts = data['workouts']
    total_workouts = len(workouts)

    if verbose:
        print(f"Total workouts: {total_workouts}")
        print()
        print("Step 1: Loading gender and engineering features...")

    # Process all workouts
    all_features = []
    all_heart_rate = []
    all_masks = []
    all_original_lengths = []
    all_user_ids = []
    skipped = 0

    for workout in tqdm(workouts, disable=not verbose):
        try:
            # Check minimum length
            if workout['data_points'] < MIN_SEQ_LENGTH:
                skipped += 1
                continue

            # Load gender from raw data
            line_number = workout['line_number']
            gender = load_gender_from_raw(line_number, raw_data_path)

            # Prepare workout data for feature engineering
            workout_data = {
                'speed': workout['speed'],
                'altitude': workout['altitude'],
                'gender': gender
            }

            # Engineer features
            features = engineer_features(workout_data)

            # Extract heart rate
            heart_rate = np.array(workout['heart_rate'], dtype=np.float32).reshape(-1, 1)

            # Validate
            assert len(features) == len(heart_rate), "Feature and HR length mismatch"

            # Pad/truncate with mask
            features_padded, hr_padded, mask, orig_len = pad_or_truncate_with_mask(
                features, heart_rate, target_length=SEQ_LENGTH
            )

            # Store
            all_features.append(features_padded)
            all_heart_rate.append(hr_padded)
            all_masks.append(mask)
            all_original_lengths.append(orig_len)
            all_user_ids.append(workout['user_id'])

        except Exception as e:
            if verbose:
                print(f"Warning: Skipped workout {workout.get('workout_id', 'unknown')}: {e}")
            skipped += 1
            continue

    if verbose:
        print(f"\nProcessed {len(all_features)} workouts (skipped {skipped})")
        print()

    # Convert to numpy arrays
    all_features = np.array(all_features, dtype=np.float32)  # [N, 500, 11]
    all_heart_rate = np.array(all_heart_rate, dtype=np.float32)  # [N, 500, 1]
    all_masks = np.array(all_masks, dtype=np.float32)  # [N, 500, 1]
    all_original_lengths = np.array(all_original_lengths, dtype=np.int32)
    all_user_ids = np.array(all_user_ids, dtype=np.int32)

    if verbose:
        print(f"Feature array shape: {all_features.shape}")
        print(f"Heart rate array shape: {all_heart_rate.shape}")
        print(f"Mask array shape: {all_masks.shape}")
        print()

    # Step 2: User-based stratified split
    if verbose:
        print("Step 2: User-based stratified split (70/15/15)...")

    # Group workouts by user
    user_to_indices = defaultdict(list)
    for idx, user_id in enumerate(all_user_ids):
        user_to_indices[user_id].append(idx)

    unique_users = list(user_to_indices.keys())
    num_users = len(unique_users)

    if verbose:
        print(f"Total unique users: {num_users}")

    # Split users (not workouts)
    # First split: 85% train+val, 15% test
    train_val_users, test_users = train_test_split(
        unique_users,
        test_size=0.15,
        random_state=RANDOM_SEED
    )

    # Second split: from train_val, split into train and val
    # val_size = 0.15 / 0.85 ≈ 0.1765
    train_users, val_users = train_test_split(
        train_val_users,
        test_size=0.1765,
        random_state=RANDOM_SEED
    )

    if verbose:
        print(f"Train users: {len(train_users)}")
        print(f"Val users: {len(val_users)}")
        print(f"Test users: {len(test_users)}")

    # Get indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    for user_id in train_users:
        train_indices.extend(user_to_indices[user_id])
    for user_id in val_users:
        val_indices.extend(user_to_indices[user_id])
    for user_id in test_users:
        test_indices.extend(user_to_indices[user_id])

    if verbose:
        print(f"Train workouts: {len(train_indices)}")
        print(f"Val workouts: {len(val_indices)}")
        print(f"Test workouts: {len(test_indices)}")
        print()

    # Split data
    train_features = all_features[train_indices]
    train_hr = all_heart_rate[train_indices]
    train_masks = all_masks[train_indices]
    train_lengths = all_original_lengths[train_indices]
    train_user_ids = all_user_ids[train_indices]

    val_features = all_features[val_indices]
    val_hr = all_heart_rate[val_indices]
    val_masks = all_masks[val_indices]
    val_lengths = all_original_lengths[val_indices]
    val_user_ids = all_user_ids[val_indices]

    test_features = all_features[test_indices]
    test_hr = all_heart_rate[test_indices]
    test_masks = all_masks[test_indices]
    test_lengths = all_original_lengths[test_indices]
    test_user_ids = all_user_ids[test_indices]

    # Step 3: Normalize features (fit on train only)
    if verbose:
        print("Step 3: Normalizing features (fit on train only)...")

    # We'll normalize all features except gender (column 2)
    # Features to normalize: 0-1 (speed, altitude), 3-10 (all temporal features)
    # Gender (column 2) stays as 0/1

    scalers = {}
    feature_names = get_feature_names()

    # Prepare normalized features
    train_features_norm = train_features.copy()
    val_features_norm = val_features.copy()
    test_features_norm = test_features.copy()

    for i, name in enumerate(feature_names):
        if name == 'gender':
            # Don't normalize gender
            continue

        if verbose:
            print(f"  Normalizing {name}...")

        # Fit scaler on training data only (flatten valid regions only)
        train_feature_flat = train_features[:, :, i].reshape(-1, 1)
        train_mask_flat = train_masks[:, :, 0].reshape(-1, 1)

        # Only fit on valid (non-padded) regions
        valid_indices = train_mask_flat.squeeze() == 1.0
        train_feature_valid = train_feature_flat[valid_indices]

        scaler = StandardScaler()
        scaler.fit(train_feature_valid.reshape(-1, 1))

        # Transform all splits
        train_features_norm[:, :, i] = scaler.transform(
            train_features[:, :, i].reshape(-1, 1)
        ).reshape(train_features.shape[0], train_features.shape[1])

        val_features_norm[:, :, i] = scaler.transform(
            val_features[:, :, i].reshape(-1, 1)
        ).reshape(val_features.shape[0], val_features.shape[1])

        test_features_norm[:, :, i] = scaler.transform(
            test_features[:, :, i].reshape(-1, 1)
        ).reshape(test_features.shape[0], test_features.shape[1])

        # Store scaler parameters
        scalers[name] = {
            'mean': float(scaler.mean_[0]),
            'std': float(scaler.scale_[0])
        }

    if verbose:
        print()

    # Step 4: Convert to PyTorch tensors
    if verbose:
        print("Step 4: Converting to PyTorch tensors...")

    train_data = {
        'features': torch.from_numpy(train_features_norm),
        'heart_rate': torch.from_numpy(train_hr),
        'mask': torch.from_numpy(train_masks),
        'original_lengths': torch.from_numpy(train_lengths),
        'user_ids': torch.from_numpy(train_user_ids)
    }

    val_data = {
        'features': torch.from_numpy(val_features_norm),
        'heart_rate': torch.from_numpy(val_hr),
        'mask': torch.from_numpy(val_masks),
        'original_lengths': torch.from_numpy(val_lengths),
        'user_ids': torch.from_numpy(val_user_ids)
    }

    test_data = {
        'features': torch.from_numpy(test_features_norm),
        'heart_rate': torch.from_numpy(test_hr),
        'mask': torch.from_numpy(test_masks),
        'original_lengths': torch.from_numpy(test_lengths),
        'user_ids': torch.from_numpy(test_user_ids)
    }

    if verbose:
        print(f"  Train: {train_data['features'].shape}")
        print(f"  Val: {val_data['features'].shape}")
        print(f"  Test: {test_data['features'].shape}")
        print()

    # Step 5: Save tensors
    if verbose:
        print(f"Step 5: Saving tensors to {output_dir}...")

    import os
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_data, f"{output_dir}/train.pt")
    torch.save(val_data, f"{output_dir}/val.pt")
    torch.save(test_data, f"{output_dir}/test.pt")

    if verbose:
        print(f"  ✓ Saved train.pt ({train_data['features'].shape[0]} samples)")
        print(f"  ✓ Saved val.pt ({val_data['features'].shape[0]} samples)")
        print(f"  ✓ Saved test.pt ({test_data['features'].shape[0]} samples)")
        print()

    # Save scaler parameters
    scaler_path = f"{output_dir}/scaler_params.json"
    with open(scaler_path, 'w') as f:
        json.dump(scalers, f, indent=2)

    if verbose:
        print(f"  ✓ Saved scaler_params.json")
        print()

    # Save metadata
    metadata = {
        'random_seed': RANDOM_SEED,
        'seq_length': SEQ_LENGTH,
        'min_seq_length': MIN_SEQ_LENGTH,
        'num_features': 11,
        'feature_names': feature_names,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
        'train_users': len(train_users),
        'val_users': len(val_users),
        'test_users': len(test_users),
        'total_workouts_processed': len(all_features),
        'total_workouts_skipped': skipped,
        'masking_enabled': True,
        'hr_normalized': False,
        'input_file': input_file,
        'raw_data_path': raw_data_path
    }

    metadata_path = f"{output_dir}/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  ✓ Saved metadata.json")
        print()
        print("=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)
        print(f"Train: {len(train_indices)} workouts from {len(train_users)} users")
        print(f"Val:   {len(val_indices)} workouts from {len(val_users)} users")
        print(f"Test:  {len(test_indices)} workouts from {len(test_users)} users")
        print()
        print(f"Avg padding ratio: {1 - (all_original_lengths.mean() / SEQ_LENGTH):.1%}")
        print()
        print("Next steps:")
        print("  1. Run Model/verify_tensors.py to validate output")
        print("  2. Start model training with Model/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sequences for SUB3_V2 training")
    parser.add_argument(
        '--input',
        default='Preprocessing/clean_dataset_smoothed.json',
        help='Path to clean_dataset_smoothed.json'
    )
    parser.add_argument(
        '--raw-data',
        default='DATA/raw/endomondoHR_proper-002.json',
        help='Path to raw Endomondo data (for gender)'
    )
    parser.add_argument(
        '--output',
        default='DATA/processed',
        help='Output directory for tensors'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print progress'
    )

    args = parser.parse_args()

    prepare_sequences(
        input_file=args.input,
        raw_data_path=args.raw_data,
        output_dir=args.output,
        verbose=args.verbose
    )
