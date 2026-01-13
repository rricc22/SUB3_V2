#!/usr/bin/env python3
"""
Build index of workouts with complete data (HR, speed, altitude).

Memory-efficient: processes one line at a time, never loads full file.
Outputs a simple text file with line numbers for fast lookup.

Author: OpenCode
Date: 2025-01-11
"""

import sys
from pathlib import Path
from time import time

DATA_FILE = Path(__file__).parent.parent.parent / "DATA" / "raw" / "endomondoHR_proper-002.json"
INDEX_DIR = Path(__file__).parent.parent.parent / "DATA" / "indices"


def build_index():
    """Stream through file and build indices."""
    
    # Create index directory
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Counters
    total = 0
    complete_run = []  # Running with HR, speed, altitude
    complete_bike = []  # Cycling with HR, speed, altitude
    run_no_speed = []  # Running without speed
    bike_no_speed = []  # Cycling without speed
    other = []  # Other sports
    
    start_time = time()
    
    print(f"Scanning {DATA_FILE}...")
    print("This may take a few minutes...\n")
    
    with open(DATA_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            
            # Progress every 10k lines
            if line_num % 10000 == 0:
                elapsed = time() - start_time
                rate = line_num / elapsed
                print(f"  Processed {line_num:,} lines ({rate:.0f} lines/sec)...")
            
            # Quick string checks (faster than parsing)
            has_hr = "'heart_rate':" in line
            has_speed = "'speed':" in line
            has_altitude = "'altitude':" in line
            is_run = "'sport': 'run'" in line
            is_bike = "'sport': 'bike'" in line
            
            # Must have HR (all workouts should have this)
            if not has_hr:
                continue
            
            # Categorize
            if is_run:
                if has_speed and has_altitude:
                    complete_run.append(line_num)
                else:
                    run_no_speed.append(line_num)
            elif is_bike:
                if has_speed and has_altitude:
                    complete_bike.append(line_num)
                else:
                    bike_no_speed.append(line_num)
            else:
                other.append(line_num)
    
    elapsed = time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"\nTotal lines scanned: {total:,}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\n--- Breakdown ---")
    print(f"Running (complete):     {len(complete_run):,}")
    print(f"Running (no speed):     {len(run_no_speed):,}")
    print(f"Cycling (complete):     {len(complete_bike):,}")
    print(f"Cycling (no speed):     {len(bike_no_speed):,}")
    print(f"Other sports:           {len(other):,}")
    
    # Save indices
    print(f"\n--- Saving indices to {INDEX_DIR} ---")
    
    def save_index(filename: str, lines: list):
        filepath = INDEX_DIR / filename
        with open(filepath, 'w') as f:
            for ln in lines:
                f.write(f"{ln}\n")
        print(f"  {filename}: {len(lines):,} workouts")
    
    save_index("running_complete.txt", complete_run)
    save_index("running_no_speed.txt", run_no_speed)
    save_index("cycling_complete.txt", complete_bike)
    save_index("cycling_no_speed.txt", bike_no_speed)
    save_index("other_sports.txt", other)
    
    # Save summary
    summary_path = INDEX_DIR / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Index built: {time()}\n")
        f.write(f"Total workouts: {total}\n")
        f.write(f"Running (complete): {len(complete_run)}\n")
        f.write(f"Running (no speed): {len(run_no_speed)}\n")
        f.write(f"Cycling (complete): {len(complete_bike)}\n")
        f.write(f"Cycling (no speed): {len(bike_no_speed)}\n")
        f.write(f"Other sports: {len(other)}\n")
    
    print(f"\nDone! Index files saved to {INDEX_DIR}")
    print(f"\nFor the model, use: running_complete.txt ({len(complete_run):,} workouts)")
    
    return {
        'total': total,
        'running_complete': len(complete_run),
        'running_no_speed': len(run_no_speed),
        'cycling_complete': len(complete_bike),
        'cycling_no_speed': len(bike_no_speed),
        'other': len(other)
    }


if __name__ == "__main__":
    build_index()
