"""
Investigate specific workout to understand why prediction failed.

Author: Riccardo
Date: 2026-01-14
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from lstm import HeartRateLSTM_V2

# Load model
checkpoint = torch.load('Model/checkpoints/best_model.pt', map_location='cpu')
model_config = checkpoint['model_config']
model = HeartRateLSTM_V2(
    input_size=model_config['input_size'],
    hidden_size=model_config['hidden_size'],
    num_layers=model_config['num_layers'],
    dropout=model_config['dropout'],
    bidirectional=model_config['bidirectional']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
test_data = torch.load('DATA/processed/test.pt', map_location='cpu')

# Load scaler params
with open('DATA/processed/scaler_params.json', 'r') as f:
    scaler_params = json.load(f)

# Select samples (same logic as animation script)
features = test_data['features']
heart_rate = test_data['heart_rate']
lengths = test_data['original_lengths']

candidates = []
with torch.no_grad():
    for idx in range(0, len(features), 100):
        length = int(lengths[idx])
        if length < 100:
            continue

        feat = features[idx:idx+1]
        pred = model(feat).squeeze().numpy()
        target = heart_rate[idx].squeeze().numpy()
        mae = np.mean(np.abs(pred[:length] - target[:length]))

        candidates.append({'idx': idx, 'length': length, 'mae': mae})

candidates.sort(key=lambda x: x['mae'])

# Select same samples as animation (best 3, mid 4, worst 3)
selected = []
selected.extend(candidates[:3])
mid = len(candidates) // 2
selected.extend(candidates[mid-2:mid+2])
selected.extend(candidates[-3:])
selected = selected[:10]

# Workout #7 is the 8th sample (index 7)
workout_7 = selected[7]
idx = workout_7['idx']

print("=" * 60)
print("INVESTIGATING WORKOUT #7 (WORST PERFORMER IN HEATMAP)")
print("=" * 60)
print(f"Test dataset index: {idx}")
print(f"MAE: {workout_7['mae']:.2f} BPM")
print(f"Length: {workout_7['length']} seconds")

# Get full data
length = workout_7['length']
feat = features[idx:idx+1]
hr_target = heart_rate[idx].squeeze().numpy()[:length]

with torch.no_grad():
    hr_pred = model(feat).squeeze().numpy()[:length]

# Denormalize features
feat_np = features[idx, :length].numpy()
speed = feat_np[:, 0] * scaler_params['speed']['std'] + scaler_params['speed']['mean']
altitude = feat_np[:, 1] * scaler_params['altitude']['std'] + scaler_params['altitude']['mean']

# Compute statistics
print(f"\n{'Feature':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 60)
print(f"{'Speed (km/h)':<20} {speed.mean():<12.2f} {speed.std():<12.2f} {speed.min():<12.2f} {speed.max():<12.2f}")
print(f"{'Altitude (m)':<20} {altitude.mean():<12.2f} {altitude.std():<12.2f} {altitude.min():<12.2f} {altitude.max():<12.2f}")
print(f"{'True HR (BPM)':<20} {hr_target.mean():<12.2f} {hr_target.std():<12.2f} {hr_target.min():<12.2f} {hr_target.max():<12.2f}")
print(f"{'Pred HR (BPM)':<20} {hr_pred.mean():<12.2f} {hr_pred.std():<12.2f} {hr_pred.min():<12.2f} {hr_pred.max():<12.2f}")

# Compute error distribution
error = np.abs(hr_pred - hr_target)
print(f"\n{'Error Statistics':<20}")
print("-" * 60)
print(f"Mean Error (MAE):     {error.mean():.2f} BPM")
print(f"Median Error:         {np.median(error):.2f} BPM")
print(f"Std Error:            {error.std():.2f} BPM")
print(f"Max Error:            {error.max():.2f} BPM")
print(f"95th percentile:      {np.percentile(error, 95):.2f} BPM")

# Check for anomalies
print(f"\n{'Potential Issues':<20}")
print("-" * 60)

# Issue 1: HR offset
hr_diff = hr_target.mean() - hr_pred.mean()
print(f"HR Offset (True-Pred): {hr_diff:.2f} BPM")
if abs(hr_diff) > 20:
    print("  ⚠️  LARGE OFFSET DETECTED - Possible HR sensor issue")

# Issue 2: Low speed (recovery run)
if speed.mean() < 7:
    print(f"  ⚠️  LOW SPEED WORKOUT ({speed.mean():.1f} km/h) - Model struggles with recovery runs")

# Issue 3: Unusual HR range
if hr_target.mean() < 120 or hr_target.mean() > 180:
    print(f"  ⚠️  UNUSUAL HR RANGE (mean={hr_target.mean():.0f}) - Outside typical running HR")

# Issue 4: HR variability
if hr_target.std() < 5:
    print(f"  ⚠️  VERY FLAT HR (std={hr_target.std():.1f}) - Possible sensor malfunction")

# Issue 5: Speed-HR correlation
corr = np.corrcoef(speed, hr_target)[0, 1]
print(f"Speed-HR correlation:  {corr:.3f}")
if abs(corr) < 0.1:
    print("  ⚠️  WEAK CORRELATION - HR doesn't respond to speed changes")

# Create detailed visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle(f'Workout #{idx} - Detailed Analysis (MAE: {workout_7["mae"]:.2f} BPM)',
             fontsize=16, fontweight='bold')

# Plot 1: Speed
axes[0].plot(speed, 'b-', linewidth=1.5)
axes[0].fill_between(range(len(speed)), speed, alpha=0.3, color='blue')
axes[0].set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
axes[0].set_title('Running Speed Profile', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, length)

# Plot 2: HR comparison
axes[1].plot(hr_target, 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
axes[1].plot(hr_pred, 'orange', linewidth=2, label='Prediction', alpha=0.8)
axes[1].set_ylabel('Heart Rate (BPM)', fontsize=12, fontweight='bold')
axes[1].set_title('Heart Rate: Prediction vs Ground Truth', fontsize=12)
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, length)

# Plot 3: Absolute error
axes[2].plot(error, 'r-', linewidth=1.5)
axes[2].fill_between(range(len(error)), error, alpha=0.3, color='red')
axes[2].axhline(error.mean(), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {error.mean():.1f} BPM')
axes[2].set_ylabel('Absolute Error (BPM)', fontsize=12, fontweight='bold')
axes[2].set_title('Prediction Error Over Time', fontsize=12)
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, length)

# Plot 4: Speed vs HR scatter
axes[3].scatter(speed, hr_target, alpha=0.5, s=10, label='Ground Truth', color='green')
axes[3].scatter(speed, hr_pred, alpha=0.5, s=10, label='Prediction', color='orange')
axes[3].set_xlabel('Speed (km/h)', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Heart Rate (BPM)', fontsize=12, fontweight='bold')
axes[3].set_title(f'Speed-HR Relationship (correlation: {corr:.3f})', fontsize=12)
axes[3].legend(loc='best')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Model/results/workout_7_investigation.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved detailed visualization to Model/results/workout_7_investigation.png")
