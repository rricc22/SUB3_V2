#!/usr/bin/env python3
"""
Apple Watch Gallery Generator for Workout Inspection.

Generates thumbnail plots and an HTML gallery with lazy loading
for rapid visual inspection of Apple Watch outdoor run workouts.

Usage:
    python generate_apple_watch_gallery.py

Author: Claude
Date: 2026-01-13
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
STATS_FILE = PROJECT_ROOT / "DATA" / "raw" / "apple_watch_stats.csv"
APPLE_WATCH_DIR = PROJECT_ROOT / "DATA" / "raw" / "apple_health_export_User1"

OUTPUT_DIR = PROJECT_ROOT / "EDA" / "apple_watch_gallery"
THUMBS_DIR = OUTPUT_DIR / "thumbs"
HTML_FILE = OUTPUT_DIR / "index.html"

# Thumbnail settings - bigger than original
THUMB_WIDTH = 400
THUMB_HEIGHT = 200
THUMB_DPI = 60

# Processing
NUM_WORKERS = max(1, mp.cpu_count() - 2)
BATCH_SIZE = 100  # Process in batches for progress updates


# =============================================================================
# Data Loading
# =============================================================================

def load_workout_stats():
    """Load workout statistics from CSV."""
    print("Loading Apple Watch workout stats...")
    stats_df = pd.read_csv(STATS_FILE)

    # Convert to list of dicts for easier processing
    workouts = []
    for _, row in stats_df.iterrows():
        workouts.append({
            'workout_id': row['workout_id'],
            'start_time': row['start_time'],
            'duration_min': row['duration_seconds'] / 60,
            'distance_km': row['distance_km'],
            'avg_speed_kmh': row['avg_speed_kmh'],
            'avg_hr': row['avg_hr'] if pd.notna(row['avg_hr']) else 0,
            'max_hr': row['max_hr'] if pd.notna(row['max_hr']) else 0,
            'elevation_gain': row['elevation_gain'],
            'hr_coverage': row['hr_coverage'],
            'num_points': row['num_points'],
        })

    print(f"Loaded {len(workouts)} workouts")
    return workouts


def load_workout_timeseries(workout_id: str):
    """Load time series data for a single workout."""
    route_file = APPLE_WATCH_DIR / f"Outdoor Run-Route-{workout_id}.csv"
    hr_file = APPLE_WATCH_DIR / f"Outdoor Run-Heart Rate-{workout_id}.csv"

    if not route_file.exists():
        return None

    try:
        # Load route data (GPS, speed, elevation)
        route_df = pd.read_csv(route_file)
        route_df['Timestamp'] = pd.to_datetime(route_df['Timestamp']).dt.tz_localize(None)

        # Extract time series
        timestamps = route_df['Timestamp'].values
        speed = route_df['Speed (m/s)'].values
        altitude = route_df['Altitude (m)'].values

        # Load heart rate data if available
        heart_rate = np.full(len(timestamps), np.nan)
        if hr_file.exists():
            hr_df = pd.read_csv(hr_file)
            hr_df['Timestamp'] = pd.to_datetime(hr_df['Date/Time']).dt.tz_localize(None)

            # Merge HR with route timestamps using nearest match
            route_df['heart_rate'] = np.nan
            hr_df = hr_df.sort_values('Timestamp')
            route_df = route_df.sort_values('Timestamp')

            merged_df = pd.merge_asof(
                route_df[['Timestamp']].reset_index(),
                hr_df[['Timestamp', 'Avg (count/min)']].rename(columns={'Avg (count/min)': 'heart_rate'}),
                on='Timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('10s')
            )

            heart_rate = merged_df['heart_rate'].values

        # Convert timestamps to minutes from start
        timestamps = pd.to_datetime(timestamps)
        time_min = (timestamps - timestamps[0]).total_seconds() / 60

        return {
            'time_min': time_min,
            'heart_rate': heart_rate,
            'speed': speed * 3.6,  # Convert m/s to km/h
            'altitude': altitude,
        }

    except Exception as e:
        return None


# =============================================================================
# Plot Generation
# =============================================================================

def generate_thumbnail(args):
    """Generate a single thumbnail. Args: (workout_info, output_path)"""
    workout_info, output_path = args
    fig = None

    try:
        # Load time series data
        data = load_workout_timeseries(workout_info['workout_id'])

        if data is None:
            return False, workout_info['workout_id'], "Failed to load data"

        time_min = data['time_min']
        hr = data['heart_rate']
        speed = data['speed']
        altitude = data['altitude']

        if len(time_min) < 10:
            return False, workout_info['workout_id'], "Too short"

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(THUMB_WIDTH/THUMB_DPI, THUMB_HEIGHT/THUMB_DPI),
                                             dpi=THUMB_DPI)
        fig.subplots_adjust(hspace=0.15, left=0.05, right=0.98, top=0.92, bottom=0.05)

        # HR plot (red) - only if we have HR data
        has_hr = not np.all(np.isnan(hr))
        if has_hr:
            valid_hr = hr[~np.isnan(hr)]
            valid_time_hr = time_min[~np.isnan(hr)]
            if len(valid_hr) > 0:
                ax1.plot(valid_time_hr, valid_hr, 'r-', linewidth=1.0, alpha=0.8)
                ax1.set_ylim([max(40, valid_hr.min() - 5), min(220, valid_hr.max() + 5)])
            else:
                ax1.text(0.5, 0.5, 'No HR data', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=8, color='gray')
        else:
            ax1.text(0.5, 0.5, 'No HR data', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=8, color='gray')
        ax1.set_xlim([0, time_min[-1]])
        ax1.set_ylabel('HR', fontsize=6, color='red')
        ax1.tick_params(labelbottom=False, labelleft=True, labelsize=6)
        ax1.grid(True, alpha=0.3)

        # Speed plot (blue)
        ax2.plot(time_min, speed, 'b-', linewidth=1.0, alpha=0.8)
        ax2.set_ylim([0, max(1, speed.max() * 1.1)])
        ax2.set_xlim([0, time_min[-1]])
        ax2.set_ylabel('Speed\n(km/h)', fontsize=6, color='blue')
        ax2.tick_params(labelbottom=False, labelleft=True, labelsize=6)
        ax2.grid(True, alpha=0.3)

        # Altitude plot (green)
        ax3.plot(time_min, altitude, 'g-', linewidth=1.0, alpha=0.8)
        ax3.fill_between(time_min, altitude.min(), altitude, alpha=0.3, color='green')
        ax3.set_ylim([altitude.min() - 5, altitude.max() + 5])
        ax3.set_xlim([0, time_min[-1]])
        ax3.set_ylabel('Alt (m)', fontsize=6, color='green')
        ax3.set_xlabel('Time (min)', fontsize=6)
        ax3.tick_params(labelsize=6)
        ax3.grid(True, alpha=0.3)

        # Title with key info
        hr_text = f"HR:{workout_info['avg_hr']:.0f}" if workout_info['avg_hr'] > 0 else "HR:N/A"
        title = f"{workout_info['workout_id']} | {hr_text} | {workout_info['duration_min']:.0f}min | {workout_info['distance_km']:.1f}km"

        color = 'white' if workout_info['hr_coverage'] > 50 else 'orange'
        fig.suptitle(title, fontsize=7, color=color, y=0.98)

        # Save
        plt.savefig(output_path, dpi=THUMB_DPI, facecolor='#2a2a2a', edgecolor='none')
        plt.close(fig)

        return True, workout_info['workout_id'], None

    except Exception as e:
        if fig is not None:
            plt.close(fig)
        return False, workout_info['workout_id'], str(e)


# =============================================================================
# HTML Generation
# =============================================================================

def generate_html(workouts: list, failed_ids: set):
    """Generate HTML gallery with lazy loading."""

    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>Apple Watch Workout Gallery - {total} workouts</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #fff;
            padding: 10px;
        }}
        .header {{
            position: sticky;
            top: 0;
            background: #1a1a1a;
            padding: 15px;
            z-index: 100;
            border-bottom: 2px solid #333;
            margin-bottom: 15px;
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 8px;
            color: #4fc3f7;
        }}
        .stats {{
            font-size: 13px;
            color: #999;
            margin-bottom: 10px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 8px;
            font-size: 12px;
        }}
        .legend span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .dot.good {{ background: #4CAF50; }}
        .dot.partial {{ background: #FF9800; }}
        .dot.none {{ background: #f44336; }}

        .controls {{
            margin-top: 12px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .controls button {{
            background: #555;
            color: #fff;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        }}
        .controls button:hover {{ background: #666; }}
        .controls .checked-count {{
            background: #2196F3;
            padding: 8px 14px;
            border-radius: 4px;
            font-size: 13px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
        }}
        .thumb-container {{
            position: relative;
        }}
        .thumb {{
            position: relative;
            cursor: pointer;
            border-radius: 6px;
            overflow: hidden;
            background: #2a2a2a;
            aspect-ratio: 2/1;
            border: 2px solid transparent;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .thumb img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .thumb img.loaded {{ opacity: 1; }}
        .thumb:hover {{
            transform: scale(1.03);
            border-color: #4fc3f7;
        }}

        .thumb.good {{ border-color: #4CAF50; }}
        .thumb.partial {{ border-color: #FF9800; }}
        .thumb.none {{ border-color: #f44336; }}

        .checkbox {{
            position: absolute;
            top: 6px;
            left: 6px;
            z-index: 10;
            width: 22px;
            height: 22px;
            cursor: pointer;
            accent-color: #2196F3;
        }}
        .thumb-container.checked .thumb {{
            opacity: 0.5;
            outline: 3px solid #2196F3;
            outline-offset: -3px;
        }}

        /* Modal for full-size view */
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.95);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .modal.active {{ display: flex; }}
        .modal img {{
            max-width: 95vw;
            max-height: 95vh;
            object-fit: contain;
        }}
        .modal-info {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
        }}
        .close {{
            position: absolute;
            top: 20px;
            right: 30px;
            font-size: 40px;
            cursor: pointer;
            color: #fff;
        }}

        /* Navigation */
        .nav {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #333;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
        }}
        .nav kbd {{
            background: #555;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 0 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Apple Watch Workout Gallery</h1>
        <div class="stats">{total} workouts | Avg: {avg_duration:.0f}min, {avg_distance:.1f}km | HR Coverage: {hr_good} good, {hr_partial} partial, {hr_none} none | {failed} failed</div>
        <div class="legend">
            <span><div class="dot good"></div> Good HR (&gt;50%)</span>
            <span><div class="dot partial"></div> Partial HR (1-50%)</span>
            <span><div class="dot none"></div> No HR (0%)</span>
        </div>
        <div class="controls">
            <button onclick="exportChecked()">Export Checked</button>
            <button onclick="clearChecked()">Clear All</button>
            <button onclick="checkVisible()">Check Visible</button>
            <span class="checked-count">Checked: <span id="checked-count">0</span></span>
        </div>
    </div>

    <div class="grid">
        {thumbnails}
    </div>

    <div class="modal" id="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img id="modal-img" src="">
        <div class="modal-info" id="modal-info"></div>
    </div>

    <div class="nav">
        <kbd>↑</kbd><kbd>↓</kbd> Scroll | <kbd>ESC</kbd> Close | Click to zoom
    </div>

    <script>
        // Lazy loading
        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.onload = () => img.classList.add('loaded');
                    observer.unobserve(img);
                }}
            }});
        }}, {{ rootMargin: '200px' }});

        document.querySelectorAll('.thumb img').forEach(img => observer.observe(img));

        // Modal
        function openModal(src, info) {{
            document.getElementById('modal-img').src = src;
            document.getElementById('modal-info').textContent = info;
            document.getElementById('modal').classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('modal').classList.remove('active');
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});

        document.getElementById('modal').addEventListener('click', (e) => {{
            if (e.target.id === 'modal') closeModal();
        }});

        // Checkbox functionality
        function updateCount() {{
            const checked = document.querySelectorAll('.checkbox:checked');
            document.getElementById('checked-count').textContent = checked.length;

            document.querySelectorAll('.thumb-container').forEach(container => {{
                const checkbox = container.querySelector('.checkbox');
                if (checkbox.checked) {{
                    container.classList.add('checked');
                }} else {{
                    container.classList.remove('checked');
                }}
            }});
        }}

        function clearChecked() {{
            document.querySelectorAll('.checkbox').forEach(cb => cb.checked = false);
            updateCount();
        }}

        function checkVisible() {{
            const containers = document.querySelectorAll('.thumb-container');
            containers.forEach(container => {{
                const rect = container.getBoundingClientRect();
                if (rect.top >= 0 && rect.bottom <= window.innerHeight) {{
                    container.querySelector('.checkbox').checked = true;
                }}
            }});
            updateCount();
        }}

        function exportChecked() {{
            const checked = Array.from(document.querySelectorAll('.checkbox:checked'))
                .map(cb => {{
                    const container = cb.closest('.thumb-container');
                    return container.dataset.id;
                }});

            if (checked.length === 0) {{
                alert('No workouts checked!');
                return;
            }}

            const data = {{
                'checked_workout_ids': checked,
                'total': checked.length,
                'exported_at': new Date().toISOString()
            }};

            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'checked_apple_watch_workouts_' + new Date().toISOString().split('T')[0] + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            alert(`Exported ${{checked.length}} workout IDs!`);
        }}
    </script>
</body>
</html>'''

    # Generate thumbnail HTML
    thumbnails = []
    hr_categories = {'good': 0, 'partial': 0, 'none': 0}
    total_duration = 0
    total_distance = 0

    for w in workouts:
        if w['workout_id'] in failed_ids:
            continue

        # Categorize by HR coverage
        if w['hr_coverage'] > 50:
            hr_class = 'good'
            hr_categories['good'] += 1
        elif w['hr_coverage'] > 0:
            hr_class = 'partial'
            hr_categories['partial'] += 1
        else:
            hr_class = 'none'
            hr_categories['none'] += 1

        total_duration += w['duration_min']
        total_distance += w['distance_km']

        thumb_path = f"thumbs/{w['workout_id']}.png"
        hr_text = f"HR: {w['avg_hr']:.0f} ({w['hr_coverage']:.0f}%)" if w['avg_hr'] > 0 else "No HR data"
        info = f"{w['workout_id']} | {hr_text} | {w['duration_min']:.0f}min | {w['distance_km']:.1f}km | {w['avg_speed_kmh']:.1f}km/h | +{w['elevation_gain']:.0f}m"

        thumbnails.append(
            f'<div class="thumb-container" data-id="{w["workout_id"]}">'
            f'<input type="checkbox" class="checkbox" onclick="event.stopPropagation(); updateCount();">'
            f'<div class="thumb {hr_class}" onclick="openModal(\'{thumb_path}\', \'{info}\')">'
            f'<img data-src="{thumb_path}" alt="Workout {w["workout_id"]}">'
            f'</div>'
            f'</div>'
        )

    successful_count = len(workouts) - len(failed_ids)
    avg_duration = total_duration / successful_count if successful_count > 0 else 0
    avg_distance = total_distance / successful_count if successful_count > 0 else 0

    html = html_template.format(
        total=successful_count,
        avg_duration=avg_duration,
        avg_distance=avg_distance,
        hr_good=hr_categories['good'],
        hr_partial=hr_categories['partial'],
        hr_none=hr_categories['none'],
        failed=len(failed_ids),
        thumbnails='\n        '.join(thumbnails)
    )

    with open(HTML_FILE, 'w') as f:
        f.write(html)

    print(f"HTML gallery saved to: {HTML_FILE}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("APPLE WATCH WORKOUT GALLERY GENERATOR")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    # Load workout list
    workouts = load_workout_stats()

    # Check which thumbnails already exist
    existing = set()
    for f in THUMBS_DIR.glob("*.png"):
        existing.add(f.stem)

    to_generate = [w for w in workouts if w['workout_id'] not in existing]
    print(f"Thumbnails to generate: {len(to_generate)} (skipping {len(existing)} existing)")

    failed_ids = set()

    if len(to_generate) == 0:
        print("All thumbnails already exist!")
    else:
        # Prepare tasks
        tasks = [(w, THUMBS_DIR / f"{w['workout_id']}.png") for w in to_generate]

        # Process with multiprocessing
        print(f"\nGenerating thumbnails with {NUM_WORKERS} workers...")
        start_time = time.time()

        completed = 0
        total_tasks = len(tasks)

        # Process in batches
        for chunk_start in range(0, total_tasks, BATCH_SIZE):
            chunk_end = min(chunk_start + BATCH_SIZE, total_tasks)
            chunk_tasks = tasks[chunk_start:chunk_end]

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(generate_thumbnail, task): task for task in chunk_tasks}

                for future in as_completed(futures):
                    success, workout_id, error = future.result()
                    completed += 1

                    if not success:
                        failed_ids.add(workout_id)
                        if error and completed % 10 == 0:
                            print(f"    Failed: {workout_id} - {error}")

                    # Progress update
                    if completed % 20 == 0 or completed == total_tasks:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (total_tasks - completed) / rate if rate > 0 else 0
                        print(f"  [{completed}/{total_tasks}] {rate:.1f}/s | ETA: {eta:.0f}s | Failed: {len(failed_ids)}")

        elapsed = time.time() - start_time
        print(f"\nGeneration complete in {elapsed:.1f} seconds")
        print(f"  Success: {total_tasks - len(failed_ids)}")
        print(f"  Failed: {len(failed_ids)}")

    # Generate HTML
    print("\nGenerating HTML gallery...")
    generate_html(workouts, failed_ids)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Open in browser: file://{HTML_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
