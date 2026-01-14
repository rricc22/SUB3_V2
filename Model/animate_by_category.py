"""
Animated Visualizations by Workout Category

Categorizes workouts into:
1. INTERVALS: Alternating high/low intensity (HR variability > 15 BPM std)
2. STEADY: Constant pace runs (HR variability < 8 BPM std)
3. PROGRESSIVE: Gradually increasing intensity (positive HR trend)

Author: Riccardo
Date: 2026-01-14
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import argparse
import os
from scipy import stats

from lstm import HeartRateLSTM_V2


def load_model_and_data(checkpoint_path, data_dir, device):
    """Load model and test data."""
    print("Loading model and data...")

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

    test_data = torch.load(f'{data_dir}/test.pt', map_location='cpu')

    with open(f'{data_dir}/scaler_params.json', 'r') as f:
        scaler_params = json.load(f)

    print(f"  ✓ Model loaded (epoch {checkpoint['epoch']}, val MAE: {checkpoint['val_mae']:.2f})")
    print(f"  ✓ Test samples: {test_data['features'].shape[0]}")

    return model, test_data, scaler_params


def categorize_workout(speed, heart_rate, length):
    """
    Categorize workout based on speed and HR patterns.

    Returns: ('INTERVALS' | 'STEADY' | 'PROGRESSIVE', confidence_score)
    """
    # Compute variability metrics
    hr_std = np.std(heart_rate)
    speed_std = np.std(speed)

    # Compute trend (is intensity increasing?)
    time = np.arange(length)
    hr_slope, _, _, _, _ = stats.linregress(time, heart_rate)
    speed_slope, _, _, _, _ = stats.linregress(time, speed)

    # Compute alternation (how many peaks?)
    from scipy.signal import find_peaks
    hr_peaks, _ = find_peaks(heart_rate, height=np.mean(heart_rate), distance=30)
    hr_valleys, _ = find_peaks(-heart_rate, height=-np.mean(heart_rate), distance=30)
    num_alternations = len(hr_peaks) + len(hr_valleys)

    # Decision logic
    categories = []

    # INTERVALS: High variability + multiple peaks/valleys
    if hr_std > 12 and num_alternations >= 4:
        categories.append(('INTERVALS', hr_std + num_alternations))

    # STEADY: Low variability + flat trend
    if hr_std < 8 and abs(hr_slope) < 0.02 and speed_std < 1.5:
        categories.append(('STEADY', 20 - hr_std))  # Lower std = higher confidence

    # PROGRESSIVE: Clear upward trend
    if hr_slope > 0.03 and speed_slope > 0.005:
        categories.append(('PROGRESSIVE', hr_slope * 100))

    # If no clear category, classify as STEADY if low variability, else INTERVALS
    if not categories:
        if hr_std < 10:
            categories.append(('STEADY', 10 - hr_std))
        else:
            categories.append(('INTERVALS', hr_std))

    # Return category with highest confidence
    categories.sort(key=lambda x: x[1], reverse=True)
    return categories[0][0], categories[0][1]


def select_categorized_samples(test_data, model, scaler_params, device,
                               samples_per_category=5):
    """Select samples from each workout category."""
    print("\nCategorizing workouts...")

    features = test_data['features']
    heart_rate = test_data['heart_rate']
    lengths = test_data['original_lengths']

    categories = {
        'INTERVALS': [],
        'STEADY': [],
        'PROGRESSIVE': []
    }

    with torch.no_grad():
        for idx in range(0, len(features), 50):  # Sample every 50th
            length = int(lengths[idx])

            if length < 150:  # Need sufficient length to classify
                continue

            # Get prediction
            feat = features[idx:idx+1].to(device)
            pred = model(feat).cpu().squeeze().numpy()[:length]
            target = heart_rate[idx].squeeze().numpy()[:length]

            # Denormalize speed
            feat_np = features[idx, :length].numpy()
            speed = feat_np[:, 0] * scaler_params['speed']['std'] + scaler_params['speed']['mean']

            # Compute MAE
            mae = np.mean(np.abs(pred - target))

            # Categorize
            category, confidence = categorize_workout(speed, target, length)

            categories[category].append({
                'idx': idx,
                'length': length,
                'mae': mae,
                'confidence': confidence,
                'hr_std': np.std(target),
                'speed_mean': np.mean(speed)
            })

    # Select best samples from each category
    selected = {}
    for cat in ['INTERVALS', 'STEADY', 'PROGRESSIVE']:
        # Sort by confidence
        categories[cat].sort(key=lambda x: x['confidence'], reverse=True)

        # Take top N with good MAE (<20)
        good_samples = [s for s in categories[cat] if s['mae'] < 20]
        selected[cat] = good_samples[:samples_per_category]

        print(f"\n  {cat}:")
        print(f"    Total found: {len(categories[cat])}")
        print(f"    Selected: {len(selected[cat])}")
        if selected[cat]:
            maes = [s['mae'] for s in selected[cat]]
            print(f"    MAE range: {min(maes):.1f} - {max(maes):.1f} BPM")

    return selected


def denormalize_features(features, scaler_params):
    """Denormalize speed and altitude."""
    speed = features[:, 0] * scaler_params['speed']['std'] + scaler_params['speed']['mean']
    altitude = features[:, 1] * scaler_params['altitude']['std'] + scaler_params['altitude']['mean']
    return speed, altitude


def create_category_animation(model, test_data, scaler_params, device,
                             category_name, samples, output_path, fps=20):
    """
    Create animation showing multiple workouts from same category.
    """
    print(f"\nCreating {category_name} animation...")

    if not samples:
        print(f"  ! No samples for {category_name}, skipping")
        return

    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    fig.suptitle(f'{category_name} Workouts', fontsize=18, fontweight='bold')

    # Prepare data for all samples
    workout_data = []
    max_length = 0

    for sample in samples:
        idx = sample['idx']
        length = sample['length']
        max_length = max(max_length, length)

        # Get features and target
        features_raw = test_data['features'][idx, :length].numpy()
        speed, altitude = denormalize_features(features_raw, scaler_params)
        heart_rate = test_data['heart_rate'][idx].squeeze().numpy()[:length]

        # Get prediction
        with torch.no_grad():
            feat = test_data['features'][idx:idx+1].to(device)
            prediction = model(feat).cpu().squeeze().numpy()[:length]

        workout_data.append({
            'idx': idx,
            'length': length,
            'speed': speed,
            'altitude': altitude,
            'heart_rate': heart_rate,
            'prediction': prediction,
            'mae': sample['mae']
        })

    # Initialize plots
    artists = []
    for i, (ax, data) in enumerate(zip(axes, workout_data)):
        ax.set_xlim(0, data['length'])
        ax.set_ylim(min(data['heart_rate']) - 15, max(data['heart_rate']) + 15)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Heart Rate (BPM)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Create twin axis for speed
        ax2 = ax.twinx()
        ax2.set_ylim(0, max(data['speed']) * 1.2)
        ax2.set_ylabel('Speed (km/h)', fontsize=10, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Lines
        line_speed, = ax2.plot([], [], 'b-', linewidth=1.5, alpha=0.5, label='Speed')
        line_pred, = ax.plot([], [], 'orange', linewidth=2.5, label='Predicted HR')
        line_true, = ax.plot([], [], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)

        # Title with MAE
        title = ax.text(0.5, 1.05, f'Workout #{data["idx"]} (MAE: {data["mae"]:.1f} BPM)',
                       transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)

        artists.append({
            'ax': ax,
            'ax2': ax2,
            'line_speed': line_speed,
            'line_pred': line_pred,
            'line_true': line_true,
            'title': title,
            'data': data
        })

    def animate(frame):
        results = []
        for artist in artists:
            data = artist['data']
            t = min(frame * 5, data['length'])

            if t > 0:
                x = np.arange(t)
                artist['line_speed'].set_data(x, data['speed'][:t])
                artist['line_pred'].set_data(x, data['prediction'][:t])
                artist['line_true'].set_data(x, data['heart_rate'][:t])

            results.extend([artist['line_speed'], artist['line_pred'], artist['line_true']])

        return results

    frames = (max_length // 5) + 1
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=1000/fps, blit=True, repeat=True)

    anim.save(output_path, writer='pillow', fps=fps)
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def create_comparison_animation(model, test_data, scaler_params, device,
                               all_samples, output_path, fps=20):
    """
    Create side-by-side comparison of one workout from each category.
    """
    print(f"\nCreating cross-category comparison animation...")

    # Select one representative from each category
    representatives = []
    for cat in ['INTERVALS', 'STEADY', 'PROGRESSIVE']:
        if all_samples[cat]:
            # Pick the one with median MAE
            sorted_samples = sorted(all_samples[cat], key=lambda x: x['mae'])
            representatives.append((cat, sorted_samples[len(sorted_samples)//2]))

    if len(representatives) < 3:
        print("  ! Not enough samples for comparison, skipping")
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Category Comparison: Intervals vs Steady vs Progressive',
                 fontsize=16, fontweight='bold')

    # Prepare data
    workout_data = []
    max_length = 0

    for cat, sample in representatives:
        idx = sample['idx']
        length = sample['length']
        max_length = max(max_length, length)

        features_raw = test_data['features'][idx, :length].numpy()
        speed, altitude = denormalize_features(features_raw, scaler_params)
        heart_rate = test_data['heart_rate'][idx].squeeze().numpy()[:length]

        with torch.no_grad():
            feat = test_data['features'][idx:idx+1].to(device)
            prediction = model(feat).cpu().squeeze().numpy()[:length]

        workout_data.append({
            'category': cat,
            'idx': idx,
            'length': length,
            'speed': speed,
            'heart_rate': heart_rate,
            'prediction': prediction,
            'mae': sample['mae']
        })

    # Initialize plots
    artists = []
    for i, (ax, data) in enumerate(zip(axes, workout_data)):
        ax.set_xlim(0, data['length'])
        ax.set_ylim(min(data['heart_rate']) - 15, max(data['heart_rate']) + 15)
        ax.set_ylabel('Heart Rate (BPM)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i == 2:
            ax.set_xlabel('Time (seconds)', fontsize=11)

        # Twin axis for speed
        ax2 = ax.twinx()
        ax2.set_ylim(0, max(data['speed']) * 1.2)
        ax2.set_ylabel('Speed (km/h)', fontsize=10, color='blue', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='blue')

        # Lines
        line_speed, = ax2.plot([], [], 'b-', linewidth=1.5, alpha=0.4)
        line_pred, = ax.plot([], [], 'orange', linewidth=2.5, label='Predicted')
        line_true, = ax.plot([], [], 'green', linewidth=2.5, label='Actual', alpha=0.7)

        ax.set_title(f'{data["category"]} (MAE: {data["mae"]:.1f} BPM)',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=9)

        artists.append({
            'line_speed': line_speed,
            'line_pred': line_pred,
            'line_true': line_true,
            'data': data
        })

    def animate(frame):
        results = []
        for artist in artists:
            data = artist['data']
            t = min(frame * 5, data['length'])

            if t > 0:
                x = np.arange(t)
                artist['line_speed'].set_data(x, data['speed'][:t])
                artist['line_pred'].set_data(x, data['prediction'][:t])
                artist['line_true'].set_data(x, data['heart_rate'][:t])

            results.extend([artist['line_speed'], artist['line_pred'], artist['line_true']])

        return results

    frames = (max_length // 5) + 1
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                   interval=1000/fps, blit=True, repeat=True)

    anim.save(output_path, writer='pillow', fps=fps)
    print(f"  ✓ Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create Category-Based Animations')

    parser.add_argument('--checkpoint', default='Model/checkpoints/best_model.pt')
    parser.add_argument('--data-dir', default='DATA/processed')
    parser.add_argument('--output-dir', default='Model/animations_category')
    parser.add_argument('--samples-per-category', type=int, default=5)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print(f"Using device: {device}")

    # Load model and data
    model, test_data, scaler_params = load_model_and_data(
        args.checkpoint, args.data_dir, device
    )

    # Select categorized samples
    samples = select_categorized_samples(
        test_data, model, scaler_params, device,
        samples_per_category=args.samples_per_category
    )

    print("\n" + "=" * 60)
    print("Creating Category-Based Animations")
    print("=" * 60)

    # Create animation for each category
    for category in ['INTERVALS', 'STEADY', 'PROGRESSIVE']:
        if samples[category]:
            create_category_animation(
                model, test_data, scaler_params, device,
                category, samples[category],
                f'{args.output_dir}/{category.lower()}_workouts.gif',
                fps=args.fps
            )

    # Create cross-category comparison
    create_comparison_animation(
        model, test_data, scaler_params, device,
        samples,
        f'{args.output_dir}/category_comparison.gif',
        fps=args.fps
    )

    print("\n" + "=" * 60)
    print("ANIMATIONS COMPLETE")
    print("=" * 60)
    print(f"Saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
