#!/usr/bin/env python3
"""
Quick workout viewer - terminal-based with matplotlib.

Memory-efficient viewer that loads one workout at a time.
Usage: python quick_view.py [line_number] [--running-only]

Author: OpenCode
Date: 2025-01-11
"""

import sys
import ast
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


DATA_FILE = Path(__file__).parent.parent.parent / "DATA" / "raw" / "endomondoHR_proper-002.json"


def get_line(filepath: Path, line_number: int) -> str:
    """Get a specific line using sed (memory efficient)."""
    result = subprocess.run(
        ['sed', '-n', f'{line_number}p', str(filepath)],
        capture_output=True, text=True
    )
    return result.stdout


def count_lines(filepath: Path) -> int:
    """Count lines using wc -l."""
    result = subprocess.run(['wc', '-l', str(filepath)], capture_output=True, text=True)
    return int(result.stdout.split()[0])


def find_running_lines(filepath: Path, start: int = 1, count: int = 100, require_speed: bool = True) -> list:
    """Find line numbers of running workouts using grep."""
    # Find lines with running workouts that have speed data
    if require_speed:
        # Need both 'sport': 'run' AND 'speed':
        result = subprocess.run(
            ['grep', '-n', "'sport': 'run'", str(filepath)],
            capture_output=True, text=True
        )
    else:
        result = subprocess.run(
            ['grep', '-n', "'sport': 'run'", str(filepath)],
            capture_output=True, text=True
        )
    
    lines = []
    for line in result.stdout.split('\n'):
        if line:
            line_num = int(line.split(':')[0])
            if require_speed and "'speed':" not in line:
                continue
            if line_num >= start:
                lines.append(line_num)
                if len(lines) >= count:
                    break
    return lines


def parse_workout(line: str) -> dict:
    """Parse workout from string."""
    return ast.literal_eval(line.strip())


def find_valid_length(arr: np.ndarray, threshold: int = 10) -> int:
    """Find where data becomes padded."""
    for i in range(len(arr) - threshold, 0, -1):
        if len(set(arr[i:i+threshold])) > 1:
            return min(i + threshold, len(arr))
    return len(arr)


def plot_workout(workout: dict, line_num: int, save_path: Optional[Path] = None):
    """Plot workout HR and speed."""
    hr = np.array(workout['heart_rate'])
    speed = np.array(workout.get('speed', [0] * len(hr)))  # Default to zeros if missing
    altitude = np.array(workout.get('altitude', [0] * len(hr)))
    timestamps = np.array(workout['timestamp'])
    
    # Time in minutes
    time_min = (timestamps - timestamps[0]) / 60
    
    # Find valid data length
    valid_len = min(find_valid_length(hr), find_valid_length(speed))
    
    # Compute correlation
    if valid_len > 10:
        corr = np.corrcoef(speed[:valid_len], hr[:valid_len])[0, 1]
    else:
        corr = np.nan
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Workout #{line_num} | Sport: {workout['sport']} | "
        f"User: {workout['userId']} | ID: {workout['id']}\n"
        f"Gender: {workout['gender']} | Duration: {time_min[-1]:.1f} min | "
        f"Speed-HR Correlation: {corr:.3f}",
        fontsize=12
    )
    
    # Top plot: HR and Speed
    ax1 = axes[0]
    ax2 = ax1.twinx()
    
    # HR
    line1, = ax1.plot(time_min[:valid_len], hr[:valid_len], 'r-', 
                       linewidth=1.5, label='Heart Rate', alpha=0.8)
    ax1.set_ylabel('Heart Rate (BPM)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim([max(40, hr[:valid_len].min() - 10), min(220, hr[:valid_len].max() + 10)])
    
    # Speed
    line2, = ax2.plot(time_min[:valid_len], speed[:valid_len], 'b-', 
                       linewidth=1.5, label='Speed', alpha=0.8)
    ax2.set_ylabel('Speed (km/h)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim([0, speed[:valid_len].max() * 1.1])
    
    # Legend
    ax1.legend([line1, line2], ['Heart Rate', 'Speed'], loc='upper right')
    
    # Mark padded region
    if valid_len < len(hr):
        ax1.axvspan(time_min[valid_len-1], time_min[-1], 
                    alpha=0.2, color='gray', label='Padded')
        ax1.axvline(time_min[valid_len-1], color='gray', linestyle='--', alpha=0.5)
    
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Altitude
    ax3 = axes[1]
    ax3.fill_between(time_min[:valid_len], altitude[:valid_len], 
                     alpha=0.3, color='green')
    ax3.plot(time_min[:valid_len], altitude[:valid_len], 'g-', 
             linewidth=1.5, label='Altitude')
    ax3.set_ylabel('Altitude (m)', color='green')
    ax3.set_xlabel('Time (minutes)')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True, alpha=0.3)
    
    # Stats box
    stats_text = (
        f"HR: {hr[:valid_len].mean():.0f} ± {hr[:valid_len].std():.0f} BPM "
        f"({hr[:valid_len].min():.0f}-{hr[:valid_len].max():.0f})\n"
        f"Speed: {speed[:valid_len].mean():.1f} ± {speed[:valid_len].std():.1f} km/h\n"
        f"Elevation: {altitude[:valid_len].min():.0f}-{altitude[:valid_len].max():.0f} m\n"
        f"Valid points: {valid_len}/{len(hr)} ({valid_len/len(hr)*100:.1f}%)"
    )
    ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Quick workout viewer')
    parser.add_argument('line', type=int, nargs='?', default=1, 
                        help='Line number to view (default: 1)')
    parser.add_argument('--running-only', '-r', action='store_true',
                        help='Only show running workouts')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--batch', '-b', type=int, default=None,
                        help='View multiple workouts starting from line')
    parser.add_argument('--info', '-i', action='store_true',
                        help='Show workout info without plotting')
    
    args = parser.parse_args()
    
    print(f"Loading from {DATA_FILE}...")
    
    if args.running_only:
        print("Finding running workouts...")
        running_lines = find_running_lines(DATA_FILE, start=args.line, count=args.batch or 1)
        if not running_lines:
            print("No running workouts found from that line")
            return
        lines_to_view = running_lines
    else:
        if args.batch:
            lines_to_view = list(range(args.line, args.line + args.batch))
        else:
            lines_to_view = [args.line]
    
    for line_num in lines_to_view:
        print(f"\nLoading workout at line {line_num}...")
        line = get_line(DATA_FILE, line_num)
        
        if not line.strip():
            print(f"Empty line at {line_num}")
            continue
        
        try:
            workout = parse_workout(line)
        except Exception as e:
            print(f"Failed to parse line {line_num}: {e}")
            continue
        
        if args.info:
            print(f"  Sport: {workout['sport']}")
            print(f"  Gender: {workout['gender']}")
            print(f"  User ID: {workout['userId']}")
            print(f"  Workout ID: {workout['id']}")
            print(f"  Points: {len(workout['heart_rate'])}")
            print(f"  HR range: {min(workout['heart_rate'])}-{max(workout['heart_rate'])}")
            if 'speed' in workout:
                print(f"  Speed range: {min(workout['speed']):.1f}-{max(workout['speed']):.1f}")
            else:
                print(f"  Speed: NOT AVAILABLE")
        else:
            save_path = None
            if args.save:
                if args.batch:
                    save_path = Path(args.save).parent / f"{Path(args.save).stem}_{line_num}{Path(args.save).suffix}"
                else:
                    save_path = Path(args.save)
            
            plot_workout(workout, line_num, save_path)


if __name__ == "__main__":
    main()
