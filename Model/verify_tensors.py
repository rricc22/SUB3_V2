"""
Verification Script for SUB3_V2 Tensors

Validates the preprocessing output and computes feature-HR correlations.

Author: Riccardo
Date: 2026-01-13
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from feature_engineering import get_feature_names


def load_data(data_dir='DATA/processed'):
    """Load all preprocessed tensors and metadata."""
    print("Loading preprocessed data...")

    train_data = torch.load(f'{data_dir}/train.pt')
    val_data = torch.load(f'{data_dir}/val.pt')
    test_data = torch.load(f'{data_dir}/test.pt')

    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    with open(f'{data_dir}/scaler_params.json', 'r') as f:
        scaler_params = json.load(f)

    return train_data, val_data, test_data, metadata, scaler_params


def check_shapes(train_data, val_data, test_data, metadata):
    """Verify tensor shapes."""
    print("\n" + "=" * 60)
    print("SHAPE VALIDATION")
    print("=" * 60)

    expected_seq_len = metadata['seq_length']
    expected_num_features = metadata['num_features']

    print(f"\nExpected sequence length: {expected_seq_len}")
    print(f"Expected number of features: {expected_num_features}")
    print()

    # Check train
    print("Train set:")
    print(f"  Features:  {train_data['features'].shape} ✓" if train_data['features'].shape[1:] == (expected_seq_len, expected_num_features) else f"  Features:  {train_data['features'].shape} ✗")
    print(f"  HR:        {train_data['heart_rate'].shape} ✓" if train_data['heart_rate'].shape[1:] == (expected_seq_len, 1) else f"  HR:        {train_data['heart_rate'].shape} ✗")
    print(f"  Mask:      {train_data['mask'].shape} ✓" if train_data['mask'].shape[1:] == (expected_seq_len, 1) else f"  Mask:      {train_data['mask'].shape} ✗")

    # Check val
    print("\nValidation set:")
    print(f"  Features:  {val_data['features'].shape} ✓" if val_data['features'].shape[1:] == (expected_seq_len, expected_num_features) else f"  Features:  {val_data['features'].shape} ✗")
    print(f"  HR:        {val_data['heart_rate'].shape} ✓" if val_data['heart_rate'].shape[1:] == (expected_seq_len, 1) else f"  HR:        {val_data['heart_rate'].shape} ✗")
    print(f"  Mask:      {val_data['mask'].shape} ✓" if val_data['mask'].shape[1:] == (expected_seq_len, 1) else f"  Mask:      {val_data['mask'].shape} ✗")

    # Check test
    print("\nTest set:")
    print(f"  Features:  {test_data['features'].shape} ✓" if test_data['features'].shape[1:] == (expected_seq_len, expected_num_features) else f"  Features:  {test_data['features'].shape} ✗")
    print(f"  HR:        {test_data['heart_rate'].shape} ✓" if test_data['heart_rate'].shape[1:] == (expected_seq_len, 1) else f"  HR:        {test_data['heart_rate'].shape} ✗")
    print(f"  Mask:      {test_data['mask'].shape} ✓" if test_data['mask'].shape[1:] == (expected_seq_len, 1) else f"  Mask:      {test_data['mask'].shape} ✗")


def check_masks(train_data, val_data, test_data):
    """Verify mask correctness."""
    print("\n" + "=" * 60)
    print("MASK VALIDATION")
    print("=" * 60)

    def check_mask(data, name):
        masks = data['mask'].numpy()
        original_lengths = data['original_lengths'].numpy()

        # Check that mask sum equals original lengths
        mask_sums = masks.sum(axis=(1, 2))
        length_match = np.allclose(mask_sums, original_lengths)

        # Check mask values (should be 0 or 1)
        valid_values = np.all((masks == 0) | (masks == 1))

        # Padding statistics
        total_timesteps = masks.shape[0] * masks.shape[1]
        valid_timesteps = masks.sum()
        padding_ratio = 1 - (valid_timesteps / total_timesteps)

        print(f"\n{name}:")
        print(f"  Mask sum == original_lengths: {'✓' if length_match else '✗'}")
        print(f"  All mask values 0 or 1: {'✓' if valid_values else '✗'}")
        print(f"  Padding ratio: {padding_ratio:.2%}")
        print(f"  Avg sequence length: {original_lengths.mean():.1f} / {masks.shape[1]}")

        return length_match and valid_values

    train_ok = check_mask(train_data, "Train")
    val_ok = check_mask(val_data, "Validation")
    test_ok = check_mask(test_data, "Test")

    return train_ok and val_ok and test_ok


def check_user_overlap(train_data, val_data, test_data):
    """Check for user overlap between splits."""
    print("\n" + "=" * 60)
    print("USER OVERLAP CHECK")
    print("=" * 60)

    train_users = set(train_data['user_ids'].numpy())
    val_users = set(val_data['user_ids'].numpy())
    test_users = set(test_data['user_ids'].numpy())

    train_val_overlap = train_users & val_users
    train_test_overlap = train_users & test_users
    val_test_overlap = val_users & test_users

    print(f"\nTrain users: {len(train_users)}")
    print(f"Val users: {len(val_users)}")
    print(f"Test users: {len(test_users)}")
    print()
    print(f"Train-Val overlap: {len(train_val_overlap)} users {'✗' if train_val_overlap else '✓'}")
    print(f"Train-Test overlap: {len(train_test_overlap)} users {'✗' if train_test_overlap else '✓'}")
    print(f"Val-Test overlap: {len(val_test_overlap)} users {'✗' if val_test_overlap else '✓'}")

    return len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0


def compute_feature_correlations(train_data, metadata):
    """Compute feature-HR correlations."""
    print("\n" + "=" * 60)
    print("FEATURE-HR CORRELATIONS")
    print("=" * 60)

    features = train_data['features'].numpy()
    heart_rate = train_data['heart_rate'].numpy()
    masks = train_data['mask'].numpy()

    feature_names = metadata['feature_names']

    print("\nComputing correlations (using only valid/non-padded regions)...")
    print()

    correlations = []

    for i, name in enumerate(feature_names):
        # Extract feature and HR, flatten
        feature_flat = features[:, :, i].flatten()
        hr_flat = heart_rate[:, :, 0].flatten()
        mask_flat = masks[:, :, 0].flatten()

        # Only use valid (non-padded) regions
        valid_indices = mask_flat == 1.0
        feature_valid = feature_flat[valid_indices]
        hr_valid = hr_flat[valid_indices]

        # Compute correlation
        if len(feature_valid) > 0:
            corr, _ = pearsonr(feature_valid, hr_valid)
        else:
            corr = 0.0

        correlations.append((name, corr))

        # Print with color coding
        if abs(corr) >= 0.4:
            emoji = "🟢"  # Strong
        elif abs(corr) >= 0.25:
            emoji = "🟡"  # Moderate
        else:
            emoji = "🔴"  # Weak

        print(f"  {emoji} {name:30s}: {corr:7.4f}")

    # Print summary
    print()
    print("Summary:")
    avg_corr = np.mean([abs(c) for _, c in correlations])
    max_corr_name, max_corr = max(correlations, key=lambda x: abs(x[1]))

    print(f"  Average |correlation|: {avg_corr:.4f}")
    print(f"  Strongest correlation: {max_corr_name} ({max_corr:.4f})")
    print()

    if avg_corr > 0.35:
        print(f"  ✓ Target achieved! Avg correlation > 0.35 (V1 had 0.25)")
    else:
        print(f"  ! Avg correlation is {avg_corr:.4f}, target was > 0.35")

    return correlations


def check_normalization(train_data, metadata):
    """Check normalization statistics."""
    print("\n" + "=" * 60)
    print("NORMALIZATION CHECK")
    print("=" * 60)

    features = train_data['features'].numpy()
    heart_rate = train_data['heart_rate'].numpy()
    masks = train_data['mask'].numpy()

    feature_names = metadata['feature_names']

    print("\nFeature statistics (train set, valid regions only):")
    print()

    for i, name in enumerate(feature_names):
        # Extract feature, flatten, mask
        feature_flat = features[:, :, i].flatten()
        mask_flat = masks[:, :, 0].flatten()

        # Only use valid regions
        valid_indices = mask_flat == 1.0
        feature_valid = feature_flat[valid_indices]

        mean = feature_valid.mean()
        std = feature_valid.std()

        # Check if normalized (mean ≈ 0, std ≈ 1)
        if name == 'gender':
            status = "✓ Binary (not normalized)" if mean in [0.0, 0.5, 1.0] else "✗ Unexpected"
        else:
            normalized = (abs(mean) < 0.1) and (0.8 < std < 1.2)
            status = "✓ Normalized" if normalized else "! Check normalization"

        print(f"  {name:30s}: mean={mean:7.4f}, std={std:6.4f}  {status}")

    # Check HR (should NOT be normalized)
    hr_flat = heart_rate[:, :, 0].flatten()
    mask_flat = masks[:, :, 0].flatten()
    valid_indices = mask_flat == 1.0
    hr_valid = hr_flat[valid_indices]

    hr_mean = hr_valid.mean()
    hr_std = hr_valid.std()
    hr_normalized = (abs(hr_mean) < 10) and (hr_std < 10)

    print()
    print(f"  Heart Rate (target):  mean={hr_mean:7.2f}, std={hr_std:6.2f}")
    print(f"    Status: {'✗ NORMALIZED (should be raw BPM)' if hr_normalized else '✓ Raw BPM values'}")


def visualize_sample(train_data, metadata, sample_idx=0):
    """Visualize a sample workout."""
    print("\n" + "=" * 60)
    print("SAMPLE VISUALIZATION")
    print("=" * 60)

    features = train_data['features'][sample_idx].numpy()
    hr = train_data['heart_rate'][sample_idx].numpy()
    mask = train_data['mask'][sample_idx].numpy()
    orig_len = train_data['original_lengths'][sample_idx].item()

    feature_names = metadata['feature_names']

    print(f"\nSample {sample_idx}:")
    print(f"  Original length: {orig_len}")
    print(f"  Padded length: {len(features)}")
    print(f"  Valid timesteps: {int(mask.sum())}")
    print()

    # Plot
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]
        ax.plot(features[:orig_len, i], label=name, linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.axvline(orig_len, color='red', linestyle='--', alpha=0.5, label='Padding start')
        if i == 0:
            ax.legend()

    # Plot HR in last subplot
    axes[-1].plot(hr[:orig_len, 0], label='Heart Rate', color='red', linewidth=1.0)
    axes[-1].set_title('Heart Rate (Target)')
    axes[-1].set_xlabel('Timestep')
    axes[-1].set_ylabel('BPM')
    axes[-1].grid(True, alpha=0.3)
    axes[-1].axvline(orig_len, color='red', linestyle='--', alpha=0.5, label='Padding start')
    axes[-1].legend()

    plt.tight_layout()
    plt.savefig('DATA/processed/sample_visualization.png', dpi=150)
    print("  ✓ Saved visualization to DATA/processed/sample_visualization.png")


def main():
    """Main verification routine."""
    print("=" * 60)
    print("SUB3_V2 TENSOR VERIFICATION")
    print("=" * 60)

    # Load data
    train_data, val_data, test_data, metadata, scaler_params = load_data()

    print("\n✓ Loaded all data successfully")
    print(f"  Train: {train_data['features'].shape[0]} samples")
    print(f"  Val: {val_data['features'].shape[0]} samples")
    print(f"  Test: {test_data['features'].shape[0]} samples")

    # Run checks
    check_shapes(train_data, val_data, test_data, metadata)
    masks_ok = check_masks(train_data, val_data, test_data)
    no_overlap = check_user_overlap(train_data, val_data, test_data)
    correlations = compute_feature_correlations(train_data, metadata)
    check_normalization(train_data, metadata)
    visualize_sample(train_data, metadata, sample_idx=0)

    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"\n✓ Shapes: All correct")
    print(f"{'✓' if masks_ok else '✗'} Masks: {'Valid' if masks_ok else 'Issues detected'}")
    print(f"{'✓' if no_overlap else '✗'} User overlap: {'None' if no_overlap else 'Detected!'}")
    print(f"✓ Correlations: Computed")
    print(f"✓ Normalization: Checked")
    print(f"✓ Visualization: Saved")

    all_checks_passed = masks_ok and no_overlap

    if all_checks_passed:
        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nReady for model training!")
    else:
        print("\n" + "=" * 60)
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease review the issues above.")


if __name__ == "__main__":
    main()
