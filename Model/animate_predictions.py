"""
Animated Visualizations for SUB3_V2 Predictions

Creates engaging animations showing:
1. Gradual reveal: Speed + HR prediction + ground truth appearing over time
2. Error evolution: Live error tracking throughout workout
3. Multi-workout comparison: 4 workouts side-by-side
4. Feature influence: How speed/altitude changes affect HR predictions
5. Prediction confidence: Showing uncertainty regions

Author: Riccardo
Date: 2026-01-14
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import json
import argparse
from tqdm import tqdm
import os

from lstm import HeartRateLSTM_V2


def load_model_and_data(checkpoint_path, data_dir, device):
    """Load model and test data."""
    print("Loading model and data...")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    model = HeartRateLSTM_V2(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        bidirectional=model_config['bidirectional']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    test_data = torch.load(f'{data_dir}/test.pt', map_location='cpu')

    # Load metadata (for denormalization)
    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load scaler params
    with open(f'{data_dir}/scaler_params.json', 'r') as f:
        scaler_params = json.load(f)

    print(f"  ✓ Model loaded (epoch {checkpoint['epoch']}, val MAE: {checkpoint['val_mae']:.2f})")
    print(f"  ✓ Test samples: {test_data['features'].shape[0]}")

    return model, test_data, metadata, scaler_params


def denormalize_features(features, scaler_params):
    """Denormalize speed and altitude for visualization."""
    speed = features[:, 0] * scaler_params['speed']['std'] + scaler_params['speed']['mean']
    altitude = features[:, 1] * scaler_params['altitude']['std'] + scaler_params['altitude']['mean']
    return speed, altitude


def select_interesting_samples(test_data, model, device, num_samples=10):
    """Select diverse, interesting samples for animation."""
    print("\nSelecting interesting samples...")

    features = test_data['features']
    heart_rate = test_data['heart_rate']
    masks = test_data['mask']
    lengths = test_data['original_lengths']

    candidates = []

    with torch.no_grad():
        # Sample every 100th workout
        for idx in range(0, len(features), 100):
            length = int(lengths[idx])

            # Skip very short sequences
            if length < 100:
                continue

            # Get prediction
            feat = features[idx:idx+1].to(device)
            pred = model(feat).cpu().squeeze().numpy()
            target = heart_rate[idx].squeeze().numpy()

            # Compute error
            mae = np.mean(np.abs(pred[:length] - target[:length]))

            candidates.append({
                'idx': idx,
                'length': length,
                'mae': mae,
                'pred': pred[:length],
                'target': target[:length]
            })

    # Sort by MAE and select diverse samples
    candidates.sort(key=lambda x: x['mae'])

    selected = []
    # Best 3
    selected.extend(candidates[:3])
    # Medium 4 (around median)
    mid = len(candidates) // 2
    selected.extend(candidates[mid-2:mid+2])
    # Worst 3
    selected.extend(candidates[-3:])

    print(f"  ✓ Selected {len(selected)} samples")
    print(f"    MAE range: {selected[0]['mae']:.1f} - {selected[-1]['mae']:.1f} BPM")

    return selected[:num_samples]


def create_gradual_reveal_animation(model, test_data, scaler_params, device,
                                    sample_idx, output_path, fps=30):
    """
    Animation 1: Gradual reveal of workout
    Shows speed profile, predicted HR, and ground truth appearing over time.
    """
    print(f"\nCreating gradual reveal animation (sample {sample_idx})...")

    # Get data
    features = test_data['features'][sample_idx:sample_idx+1].to(device)
    heart_rate = test_data['heart_rate'][sample_idx].squeeze().numpy()
    length = int(test_data['original_lengths'][sample_idx])

    # Denormalize features
    speed, altitude = denormalize_features(
        test_data['features'][sample_idx, :length].numpy(),
        scaler_params
    )

    # Get prediction
    with torch.no_grad():
        prediction = model(features).cpu().squeeze().numpy()[:length]

    # Setup figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Heart Rate Prediction: Gradual Reveal', fontsize=16, fontweight='bold')

    # Initialize lines
    line_speed, = ax1.plot([], [], 'b-', linewidth=2, label='Speed')
    line_pred, = ax2.plot([], [], 'orange', linewidth=2, label='Predicted HR')
    line_true, = ax2.plot([], [], 'g-', linewidth=2, label='Ground Truth HR')
    line_error, = ax3.plot([], [], 'r-', linewidth=2, label='Absolute Error')
    fill_error = ax3.fill_between([], [], 0, alpha=0.3, color='red')

    # Configure axes
    ax1.set_xlim(0, length)
    ax1.set_ylim(0, max(speed) * 1.1)
    ax1.set_ylabel('Speed (km/h)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.set_xlim(0, length)
    ax2.set_ylim(min(heart_rate) - 10, max(heart_rate) + 10)
    ax2.set_ylabel('Heart Rate (BPM)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    ax3.set_xlim(0, length)
    error = np.abs(prediction - heart_rate[:length])
    ax3.set_ylim(0, max(error) * 1.1)
    ax3.set_xlabel('Timestep (seconds)', fontsize=12)
    ax3.set_ylabel('Absolute Error (BPM)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # MAE text
    mae_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                        fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def animate(frame):
        # Reveal more data each frame
        t = min(frame * 5, length)  # Reveal 5 timesteps per frame

        if t == 0:
            return line_speed, line_pred, line_true, line_error, mae_text

        x = np.arange(t)

        # Update speed
        line_speed.set_data(x, speed[:t])

        # Update HR predictions
        line_pred.set_data(x, prediction[:t])
        line_true.set_data(x, heart_rate[:t])

        # Update error
        line_error.set_data(x, error[:t])

        # Update MAE
        current_mae = np.mean(error[:t])
        mae_text.set_text(f'MAE: {current_mae:.2f} BPM\nTimestep: {t}/{length}')

        return line_speed, line_pred, line_true, line_error, mae_text

    # Create animation
    frames = (length // 5) + 1
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=1000/fps, blit=True, repeat=True)

    # Save
    anim.save(output_path, writer='pillow', fps=fps)
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def create_multi_workout_animation(model, test_data, scaler_params, device,
                                   sample_indices, output_path, fps=30):
    """
    Animation 2: Multi-workout comparison
    Shows 4 workouts side-by-side with predictions appearing gradually.
    """
    print(f"\nCreating multi-workout comparison animation...")

    num_workouts = len(sample_indices)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Multi-Workout Comparison', fontsize=16, fontweight='bold')

    # Prepare data for all workouts
    workout_data = []
    max_length = 0

    for idx in sample_indices:
        features = test_data['features'][idx:idx+1].to(device)
        heart_rate = test_data['heart_rate'][idx].squeeze().numpy()
        length = int(test_data['original_lengths'][idx])
        max_length = max(max_length, length)

        # Get prediction
        with torch.no_grad():
            prediction = model(features).cpu().squeeze().numpy()[:length]

        # Denormalize speed
        speed, _ = denormalize_features(
            test_data['features'][idx, :length].numpy(),
            scaler_params
        )

        mae = np.mean(np.abs(prediction - heart_rate[:length]))

        workout_data.append({
            'speed': speed,
            'prediction': prediction,
            'heart_rate': heart_rate[:length],
            'length': length,
            'mae': mae
        })

    # Initialize plots
    lines = []
    for i, (ax, data) in enumerate(zip(axes, workout_data)):
        ax.set_xlim(0, data['length'])
        ax.set_ylim(min(data['heart_rate']) - 10, max(data['heart_rate']) + 10)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Heart Rate (BPM)')
        ax.set_title(f"Workout {sample_indices[i]} (MAE: {data['mae']:.2f} BPM)")
        ax.grid(True, alpha=0.3)

        line_pred, = ax.plot([], [], 'orange', linewidth=2, label='Predicted', alpha=0.8)
        line_true, = ax.plot([], [], 'green', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.legend(loc='upper right', fontsize=8)

        lines.append({'pred': line_pred, 'true': line_true, 'data': data})

    def animate(frame):
        results = []
        for line_dict in lines:
            t = min(frame * 5, line_dict['data']['length'])
            if t > 0:
                x = np.arange(t)
                line_dict['pred'].set_data(x, line_dict['data']['prediction'][:t])
                line_dict['true'].set_data(x, line_dict['data']['heart_rate'][:t])
            results.extend([line_dict['pred'], line_dict['true']])
        return results

    frames = (max_length // 5) + 1
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=1000/fps, blit=True, repeat=True)

    anim.save(output_path, writer='pillow', fps=fps)
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def create_feature_influence_animation(model, test_data, scaler_params, device,
                                       sample_idx, output_path, fps=30):
    """
    Animation 3: Feature influence on predictions
    Shows how speed changes correlate with HR prediction changes.
    """
    print(f"\nCreating feature influence animation (sample {sample_idx})...")

    # Get data
    features = test_data['features'][sample_idx:sample_idx+1].to(device)
    heart_rate = test_data['heart_rate'][sample_idx].squeeze().numpy()
    length = int(test_data['original_lengths'][sample_idx])

    # Denormalize
    speed, altitude = denormalize_features(
        test_data['features'][sample_idx, :length].numpy(),
        scaler_params
    )

    # Get prediction
    with torch.no_grad():
        prediction = model(features).cpu().squeeze().numpy()[:length]

    # Setup figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax_speed = fig.add_subplot(gs[0, :])
    ax_altitude = fig.add_subplot(gs[1, :])
    ax_hr = fig.add_subplot(gs[2, :])

    fig.suptitle('Feature Influence on Heart Rate Prediction', fontsize=16, fontweight='bold')

    # Initialize lines
    line_speed, = ax_speed.plot([], [], 'b-', linewidth=2)
    fill_speed = ax_speed.fill_between([], [], 0, alpha=0.3, color='blue')

    line_altitude, = ax_altitude.plot([], [], 'brown', linewidth=2)

    line_pred, = ax_hr.plot([], [], 'orange', linewidth=2.5, label='Predicted HR')
    line_true, = ax_hr.plot([], [], 'g--', linewidth=2, label='Ground Truth', alpha=0.7)

    # Vertical cursor line
    cursor_speed = ax_speed.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    cursor_altitude = ax_altitude.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    cursor_hr = ax_hr.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Configure axes
    for ax in [ax_speed, ax_altitude, ax_hr]:
        ax.set_xlim(0, length)
        ax.grid(True, alpha=0.3)

    ax_speed.set_ylim(0, max(speed) * 1.1)
    ax_speed.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    ax_speed.set_title('Running Speed', fontsize=12)

    ax_altitude.set_ylim(min(altitude) - 10, max(altitude) + 10)
    ax_altitude.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    ax_altitude.set_title('Altitude Profile', fontsize=12)

    ax_hr.set_ylim(min(heart_rate) - 10, max(heart_rate) + 10)
    ax_hr.set_ylabel('Heart Rate (BPM)', fontsize=12, fontweight='bold')
    ax_hr.set_xlabel('Timestep (seconds)', fontsize=12)
    ax_hr.set_title('Heart Rate Prediction', fontsize=12)
    ax_hr.legend(loc='upper right')

    # Info text
    info_text = ax_hr.text(0.02, 0.05, '', transform=ax_hr.transAxes,
                          fontsize=11, verticalalignment='bottom',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def animate(frame):
        t = min(frame * 5, length)

        if t == 0:
            return (line_speed, line_altitude, line_pred, line_true,
                    cursor_speed, cursor_altitude, cursor_hr, info_text)

        x = np.arange(t)

        # Update lines
        line_speed.set_data(x, speed[:t])
        line_altitude.set_data(x, altitude[:t])
        line_pred.set_data(x, prediction[:t])
        line_true.set_data(x, heart_rate[:t])

        # Update cursors
        cursor_speed.set_xdata([t, t])
        cursor_altitude.set_xdata([t, t])
        cursor_hr.set_xdata([t, t])

        # Update info text
        current_speed = speed[t-1] if t > 0 else 0
        current_alt = altitude[t-1] if t > 0 else 0
        current_pred = prediction[t-1] if t > 0 else 0
        current_true = heart_rate[t-1] if t > 0 else 0
        error = abs(current_pred - current_true)

        info_text.set_text(
            f'Time: {t}s | Speed: {current_speed:.1f} km/h | Alt: {current_alt:.0f}m\n'
            f'Predicted HR: {current_pred:.0f} BPM | True HR: {current_true:.0f} BPM | Error: {error:.1f} BPM'
        )

        return (line_speed, line_altitude, line_pred, line_true,
                cursor_speed, cursor_altitude, cursor_hr, info_text)

    frames = (length // 5) + 1
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=1000/fps, blit=True, repeat=True)

    anim.save(output_path, writer='pillow', fps=fps)
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def create_error_heatmap_animation(model, test_data, device, sample_indices,
                                   output_path, fps=10):
    """
    Animation 4: Error heatmap
    Shows error evolving across multiple workouts simultaneously.
    """
    print(f"\nCreating error heatmap animation...")

    # Prepare data
    num_workouts = len(sample_indices)
    max_length = 500  # Fixed length for heatmap

    predictions_matrix = np.zeros((num_workouts, max_length))
    targets_matrix = np.zeros((num_workouts, max_length))
    masks_matrix = np.zeros((num_workouts, max_length))

    for i, idx in enumerate(sample_indices):
        features = test_data['features'][idx:idx+1].to(device)
        heart_rate = test_data['heart_rate'][idx].squeeze().numpy()
        length = int(test_data['original_lengths'][idx])
        length = min(length, max_length)

        with torch.no_grad():
            prediction = model(features).cpu().squeeze().numpy()[:length]

        predictions_matrix[i, :length] = prediction
        targets_matrix[i, :length] = heart_rate[:length]
        masks_matrix[i, :length] = 1

    # Compute errors
    errors_matrix = np.abs(predictions_matrix - targets_matrix)
    errors_matrix[masks_matrix == 0] = np.nan

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Error Evolution Across Multiple Workouts', fontsize=16, fontweight='bold')

    # Heatmap
    im1 = ax1.imshow(errors_matrix, aspect='auto', cmap='RdYlGn_r',
                     vmin=0, vmax=30, interpolation='nearest')
    ax1.set_ylabel('Workout Index', fontsize=12)
    ax1.set_xlabel('Timestep (seconds)', fontsize=12)
    ax1.set_title('Absolute Error Heatmap (BPM)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Error (BPM)', fontsize=12)

    # Average error over time
    line_avg_error, = ax2.plot([], [], 'r-', linewidth=2, label='Mean Error')
    fill_std = ax2.fill_between([], [], [], alpha=0.3, color='red', label='±1 Std Dev')

    ax2.set_xlim(0, max_length)
    ax2.set_ylim(0, 25)
    ax2.set_xlabel('Timestep (seconds)', fontsize=12)
    ax2.set_ylabel('Error (BPM)', fontsize=12)
    ax2.set_title('Average Error Across Workouts', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Vertical cursor
    cursor1 = ax1.axvline(x=0, color='white', linestyle='--', linewidth=2)
    cursor2 = ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)

    def animate(frame):
        t = min(frame * 10, max_length - 1)

        # Update heatmap cursor
        cursor1.set_xdata([t, t])
        cursor2.set_xdata([t, t])

        # Update average error line
        if t > 0:
            x = np.arange(t)
            mean_error = np.nanmean(errors_matrix[:, :t], axis=0)
            std_error = np.nanstd(errors_matrix[:, :t], axis=0)

            line_avg_error.set_data(x, mean_error)

            # Update fill (need to recreate)
            for coll in list(ax2.collections):
                coll.remove()
            ax2.fill_between(x, mean_error - std_error, mean_error + std_error,
                            alpha=0.3, color='red')

        return cursor1, cursor2, line_avg_error

    frames = (max_length // 10) + 1
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=1000/fps, blit=False, repeat=True)

    anim.save(output_path, writer='pillow', fps=fps)
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create Animated Visualizations')

    parser.add_argument('--checkpoint', default='Model/checkpoints/best_model.pt',
                       help='Model checkpoint')
    parser.add_argument('--data-dir', default='DATA/processed',
                       help='Data directory')
    parser.add_argument('--output-dir', default='Model/animations',
                       help='Output directory')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print(f"Using device: {device}")

    # Load model and data
    model, test_data, metadata, scaler_params = load_model_and_data(
        args.checkpoint, args.data_dir, device
    )

    # Select interesting samples
    samples = select_interesting_samples(test_data, model, device, num_samples=10)
    sample_indices = [s['idx'] for s in samples]

    print("\n" + "=" * 60)
    print("Creating Animations")
    print("=" * 60)

    # Animation 1: Gradual reveal (best workout)
    create_gradual_reveal_animation(
        model, test_data, scaler_params, device,
        sample_indices[0],
        f'{args.output_dir}/1_gradual_reveal_best.gif',
        fps=args.fps
    )

    # Animation 2: Gradual reveal (worst workout)
    create_gradual_reveal_animation(
        model, test_data, scaler_params, device,
        sample_indices[-1],
        f'{args.output_dir}/2_gradual_reveal_worst.gif',
        fps=args.fps
    )

    # Animation 3: Multi-workout comparison
    create_multi_workout_animation(
        model, test_data, scaler_params, device,
        sample_indices[:4],
        f'{args.output_dir}/3_multi_workout_comparison.gif',
        fps=args.fps
    )

    # Animation 4: Feature influence
    create_feature_influence_animation(
        model, test_data, scaler_params, device,
        sample_indices[0],
        f'{args.output_dir}/4_feature_influence.gif',
        fps=args.fps
    )

    # Animation 5: Error heatmap
    create_error_heatmap_animation(
        model, test_data, device,
        sample_indices[:8],
        f'{args.output_dir}/5_error_heatmap.gif',
        fps=10
    )

    print("\n" + "=" * 60)
    print("ALL ANIMATIONS COMPLETE")
    print("=" * 60)
    print(f"Saved to: {args.output_dir}/")
    print("\nAnimations created:")
    print("  1. Gradual reveal (best workout) - Shows best prediction")
    print("  2. Gradual reveal (worst workout) - Shows challenging case")
    print("  3. Multi-workout comparison - 4 workouts side-by-side")
    print("  4. Feature influence - Speed/altitude effect on HR")
    print("  5. Error heatmap - Error patterns across workouts")


if __name__ == "__main__":
    main()
