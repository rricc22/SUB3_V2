"""
Evaluation Script for SUB3_V2

Evaluates trained model on test set and generates visualizations.

Author: Riccardo
Date: 2026-01-13
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os

from lstm import HeartRateLSTM_V2
from loss import compute_masked_mae, compute_masked_rmse


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model_config = checkpoint['model_config']
    model = HeartRateLSTM_V2(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        bidirectional=model_config['bidirectional']
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  ✓ Best Val MAE: {checkpoint['val_mae']:.2f} BPM")

    return model, checkpoint


def evaluate_test_set(model, test_data, device):
    """Evaluate model on test set."""
    print("\nEvaluating on test set...")

    features = test_data['features'].to(device)
    heart_rate = test_data['heart_rate'].to(device)
    masks = test_data['mask'].to(device)
    original_lengths = test_data['original_lengths']

    total_mae = 0.0
    total_rmse = 0.0
    num_samples = features.shape[0]

    all_predictions = []
    all_targets = []
    all_errors = []

    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 32

        for i in tqdm(range(0, num_samples, batch_size)):
            end_idx = min(i + batch_size, num_samples)

            batch_features = features[i:end_idx]
            batch_hr = heart_rate[i:end_idx]
            batch_mask = masks[i:end_idx]

            # Forward pass
            batch_predictions = model(batch_features)

            # Compute metrics
            mae = compute_masked_mae(batch_predictions, batch_hr, batch_mask)
            rmse = compute_masked_rmse(batch_predictions, batch_hr, batch_mask)

            total_mae += mae * (end_idx - i)
            total_rmse += rmse * (end_idx - i)

            # Store predictions and targets
            for j in range(batch_predictions.shape[0]):
                length = int(original_lengths[i + j])
                pred = batch_predictions[j, :length, 0].cpu().numpy()
                target = batch_hr[j, :length, 0].cpu().numpy()

                all_predictions.append(pred)
                all_targets.append(target)
                all_errors.append(np.abs(pred - target))

    avg_mae = total_mae / num_samples
    avg_rmse = total_rmse / num_samples

    print(f"\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"MAE:  {avg_mae:.2f} BPM")
    print(f"RMSE: {avg_rmse:.2f} BPM")
    print("=" * 60)

    return {
        'mae': avg_mae,
        'rmse': avg_rmse,
        'predictions': all_predictions,
        'targets': all_targets,
        'errors': all_errors
    }


def visualize_results(results, output_dir, num_samples=6):
    """Generate visualizations."""
    print(f"\nGenerating visualizations...")

    os.makedirs(output_dir, exist_ok=True)

    predictions = results['predictions']
    targets = results['targets']
    errors = results['errors']

    # 1. Sample predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(num_samples, len(predictions))):
        idx = np.random.randint(0, len(predictions))
        ax = axes[i]

        pred = predictions[idx]
        target = targets[idx]
        timesteps = len(pred)

        ax.plot(target, label='Ground Truth', linewidth=1.5, alpha=0.8)
        ax.plot(pred, label='Prediction', linewidth=1.5, alpha=0.8)
        ax.set_title(f'Sample {idx} (MAE: {np.mean(errors[idx]):.2f} BPM)')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Heart Rate (BPM)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_predictions.png', dpi=150)
    print(f"  ✓ Saved {output_dir}/sample_predictions.png")

    # 2. Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Flatten all errors
    all_errors_flat = np.concatenate(errors)

    # Histogram
    axes[0].hist(all_errors_flat, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(all_errors_flat), color='red', linestyle='--', label=f'Mean: {np.mean(all_errors_flat):.2f}')
    axes[0].set_xlabel('Absolute Error (BPM)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_errors = np.sort(all_errors_flat)
    cumsum = np.cumsum(np.ones_like(sorted_errors)) / len(sorted_errors)
    axes[1].plot(sorted_errors, cumsum, linewidth=2)
    axes[1].axvline(results['mae'], color='red', linestyle='--', label=f'MAE: {results["mae"]:.2f}')
    axes[1].set_xlabel('Absolute Error (BPM)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=150)
    print(f"  ✓ Saved {output_dir}/error_distribution.png")

    # 3. Per-sample MAE
    per_sample_mae = [np.mean(err) for err in errors]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(per_sample_mae, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(per_sample_mae), color='red', linestyle='--', label=f'Mean: {np.mean(per_sample_mae):.2f}')
    ax.set_xlabel('MAE per Sample (BPM)')
    ax.set_ylabel('Frequency')
    ax.set_title('Per-Sample MAE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_sample_mae.png', dpi=150)
    print(f"  ✓ Saved {output_dir}/per_sample_mae.png")

    # 4. Scatter plot (predicted vs actual)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Downsample for plotting (too many points)
    sample_size = min(10000, len(all_errors_flat))
    indices = np.random.choice(len(all_errors_flat), sample_size, replace=False)

    # Get corresponding predictions and targets
    flat_predictions = np.concatenate(predictions)
    flat_targets = np.concatenate(targets)

    ax.scatter(flat_targets[indices], flat_predictions[indices], alpha=0.1, s=1)
    ax.plot([50, 220], [50, 220], 'r--', label='Perfect prediction', linewidth=2)
    ax.set_xlabel('Ground Truth (BPM)')
    ax.set_ylabel('Prediction (BPM)')
    ax.set_title('Predicted vs Actual Heart Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([50, 220])
    ax.set_ylim([50, 220])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_plot.png', dpi=150)
    print(f"  ✓ Saved {output_dir}/scatter_plot.png")

    plt.close('all')


def save_results(results, checkpoint_info, output_dir):
    """Save evaluation results to JSON."""
    results_dict = {
        'test_mae': results['mae'],
        'test_rmse': results['rmse'],
        'num_samples': len(results['predictions']),
        'checkpoint_epoch': checkpoint_info['epoch'],
        'val_mae_at_checkpoint': checkpoint_info['val_mae'],
        'model_config': checkpoint_info['model_config']
    }

    output_path = f'{output_dir}/test_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n  ✓ Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SUB3_V2 Model')

    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', default='DATA/processed', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')

    args = parser.parse_args()

    # Device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model, checkpoint = load_model(args.checkpoint, device)

    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    test_data = torch.load(f'{args.data_dir}/test.pt', map_location='cpu')
    print(f"  ✓ Loaded {test_data['features'].shape[0]} test samples")

    # Evaluate
    results = evaluate_test_set(model, test_data, device)

    # Visualize
    visualize_results(results, args.output_dir)

    # Save results
    save_results(results, checkpoint, args.output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
