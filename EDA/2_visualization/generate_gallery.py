#!/usr/bin/env python3
"""
Fast HTML Gallery Generator for Workout Inspection.

Generates small PNG thumbnails and an HTML gallery with lazy loading
for rapid visual inspection of ~46K workouts.

Usage:
    python generate_gallery.py

Author: OpenCode
Date: 2026-01-12
"""

import json
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import sys
import gc

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA = PROJECT_ROOT / "DATA" / "raw" / "endomondoHR_proper-002.json"
STAGE1_OUTPUT = PROJECT_ROOT / "Preprocessing" / "stage1_full_output.json"
STAGE2_OUTPUT = PROJECT_ROOT / "Preprocessing" / "stage2_output.json"
COMPUTED_SPEED_FILE = PROJECT_ROOT / "DATA" / "processed" / "running_computed_speed.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "EDA" / "gallery"
THUMBS_DIR = OUTPUT_DIR / "thumbs"
HTML_FILE = OUTPUT_DIR / "index.html"

# Thumbnail settings - small but readable
THUMB_WIDTH = 300
THUMB_HEIGHT = 150
THUMB_DPI = 50  # Low DPI for small files

# Processing
NUM_WORKERS = max(1, mp.cpu_count() - 2)  # Leave 2 cores free
BATCH_SIZE = 500  # Process in batches for progress updates


# =============================================================================
# Data Loading
# =============================================================================

def load_computed_speeds():
    """Load computed speeds into a lookup dict."""
    speed_lookup = {}
    if not COMPUTED_SPEED_FILE.exists():
        print("WARNING: No computed speed file found!")
        return speed_lookup
    
    print("Loading computed speeds from file...")
    with open(COMPUTED_SPEED_FILE, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            speed_lookup[data['line_num']] = np.array(data['speed'])
    
    print(f"  Loaded {len(speed_lookup):,} computed speeds")
    return speed_lookup


def load_good_workouts():
    """Load all good workout line numbers and their status."""
    print("Loading stage 1 results...")
    with open(STAGE1_OUTPUT, 'r') as f:
        stage1 = json.load(f)
    
    print("Loading stage 2 results...")
    with open(STAGE2_OUTPUT, 'r') as f:
        stage2 = json.load(f)
    
    workouts = []
    
    # Stage 1 PASS workouts
    for w in stage1['workouts']:
        if w['decision'] == 'PASS':
            workouts.append({
                'line_number': w['line_number'],
                'workout_id': w['workout_id'],
                'status': 'PASS',
                'hr_mean': w['stats'].get('hr_mean', 0),
                'duration': w['stats'].get('duration_min', 0),
            })
    
    # Stage 2 auto-passed
    for w in stage2['auto_passed']:
        workouts.append({
            'line_number': w['line_number'],
            'workout_id': w['workout_id'],
            'status': 'AUTO_PASS',
            'hr_mean': w['stats'].get('hr_mean', 0),
            'duration': w['stats'].get('duration_min', 0),
        })
    
    # Stage 2 LLM KEEP
    for w in stage2['llm_validated']:
        if w.get('decision') == 'KEEP':
            workouts.append({
                'line_number': w['line_number'],
                'workout_id': w['workout_id'],
                'status': 'LLM_KEEP',
                'hr_mean': w.get('stats', {}).get('hr_mean', 0),
                'duration': w.get('stats', {}).get('duration_min', 0),
            })
        elif w.get('decision') == 'FIX':
            workouts.append({
                'line_number': w['line_number'],
                'workout_id': w['workout_id'],
                'status': 'LLM_FIX',
                'hr_mean': w.get('stats', {}).get('hr_mean', 0),
                'duration': w.get('stats', {}).get('duration_min', 0),
                'offset': w.get('estimated_offset', 0),
            })
    
    print(f"Total good workouts: {len(workouts)}")
    return workouts


# =============================================================================
# Plot Generation
# =============================================================================

def get_workout_data(line_number: int) -> dict | None:
    """Load workout data from raw file."""
    try:
        # Read directly instead of using linecache to avoid memory buildup
        with open(RAW_DATA, 'r') as f:
            for i, line in enumerate(f, start=1):
                if i == line_number:
                    if not line.strip():
                        return None
                    return ast.literal_eval(line.strip())
        return None
    except Exception:
        return None


def find_valid_length(arr: np.ndarray, threshold: int = 10) -> int:
    """Find where data becomes padded."""
    if len(arr) < threshold:
        return len(arr)
    for i in range(len(arr) - threshold, 0, -1):
        if len(set(arr[i:i+threshold])) > 1:
            return min(i + threshold, len(arr))
    return len(arr)


def generate_thumbnail(args):
    """Generate a single thumbnail. Args: (workout_info, output_path, raw_data_path, speed_lookup)"""
    workout_info, output_path, raw_data_path, speed_lookup = args
    fig = None
    
    try:
        # Load data directly with file path
        workout = None
        with open(raw_data_path, 'r') as f:
            for i, line in enumerate(f, start=1):
                if i == workout_info['line_number']:
                    if line.strip():
                        workout = ast.literal_eval(line.strip())
                    break
        
        if workout is None:
            return False, workout_info['line_number'], "No data"
        
        hr = np.array(workout.get('heart_rate', []))
        speed = np.array(workout.get('speed', []))
        timestamps = np.array(workout.get('timestamp', []))
        
        if len(hr) == 0 or len(timestamps) == 0:
            return False, workout_info['line_number'], "Empty arrays"
        
        # Handle missing speed - use computed if available
        if len(speed) == 0:
            line_num = workout_info['line_number']
            if line_num in speed_lookup:
                speed = speed_lookup[line_num]
                # Truncate/pad to match HR length
                if len(speed) > len(hr):
                    speed = speed[:len(hr)]
                elif len(speed) < len(hr):
                    speed = np.pad(speed, (0, len(hr) - len(speed)), constant_values=0)
            else:
                speed = np.zeros_like(hr)
        
        # Time in minutes
        time_min = (timestamps - timestamps[0]) / 60
        
        # Find valid length
        valid_len = min(find_valid_length(hr), find_valid_length(speed), len(time_min))
        if valid_len < 10:
            return False, workout_info['line_number'], "Too short"
        
        # Create compact figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(THUMB_WIDTH/THUMB_DPI, THUMB_HEIGHT/THUMB_DPI), 
                                        dpi=THUMB_DPI)
        fig.subplots_adjust(hspace=0.1, left=0.02, right=0.98, top=0.92, bottom=0.08)
        
        # HR plot (red)
        ax1.plot(time_min[:valid_len], hr[:valid_len], 'r-', linewidth=0.8)
        ax1.set_ylim([max(40, hr[:valid_len].min() - 5), min(220, hr[:valid_len].max() + 5)])
        ax1.set_xlim([0, time_min[valid_len-1]])
        ax1.axis('off')
        
        # Speed plot (blue)
        ax2.plot(time_min[:valid_len], speed[:valid_len], 'b-', linewidth=0.8)
        ax2.set_ylim([0, max(1, speed[:valid_len].max() * 1.1)])
        ax2.set_xlim([0, time_min[valid_len-1]])
        ax2.axis('off')
        
        # Title with key info
        status = workout_info['status']
        hr_mean = hr[:valid_len].mean()
        duration = time_min[valid_len-1]
        
        # Color-code title by status
        color = {'PASS': 'green', 'AUTO_PASS': 'orange', 'LLM_KEEP': 'blue', 'LLM_FIX': 'purple'}.get(status, 'black')
        fig.suptitle(f"#{workout_info['line_number']} HR:{hr_mean:.0f} {duration:.0f}m", 
                     fontsize=6, color=color, y=0.98)
        
        # Save
        plt.savefig(output_path, dpi=THUMB_DPI, facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return True, workout_info['line_number'], None
        
    except Exception as e:
        if fig is not None:
            plt.close(fig)
        return False, workout_info['line_number'], str(e)


# =============================================================================
# HTML Generation
# =============================================================================

def generate_html(workouts: list, failed_lines: list):
    """Generate HTML gallery with lazy loading."""
    
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>Workout Gallery - {total} workouts</title>
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
            padding: 10px;
            z-index: 100;
            border-bottom: 1px solid #333;
            margin-bottom: 10px;
        }}
        .header h1 {{ font-size: 18px; margin-bottom: 5px; }}
        .stats {{ font-size: 12px; color: #888; }}
        .legend {{ display: flex; gap: 15px; margin-top: 5px; font-size: 11px; }}
        .legend span {{ display: flex; align-items: center; gap: 4px; }}
        .legend .dot {{ width: 10px; height: 10px; border-radius: 50%; }}
        .dot.pass {{ background: #4CAF50; }}
        .dot.auto {{ background: #FF9800; }}
        .dot.keep {{ background: #2196F3; }}
        .dot.fix {{ background: #9C27B0; }}
        
        .controls {{
            margin-top: 10px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .controls button {{
            background: #555;
            color: #fff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .controls button:hover {{ background: #666; }}
        .controls .checked-count {{
            background: #2196F3;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 4px;
        }}
        .thumb-container {{
            position: relative;
        }}
        .thumb {{
            position: relative;
            cursor: pointer;
            border-radius: 4px;
            overflow: hidden;
            background: #333;
            aspect-ratio: 2/1;
        }}
        .thumb img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        .thumb img.loaded {{ opacity: 1; }}
        .thumb:hover {{ transform: scale(1.02); }}
        
        .thumb.pass {{ border: 2px solid #4CAF50; }}
        .thumb.auto {{ border: 2px solid #FF9800; }}
        .thumb.keep {{ border: 2px solid #2196F3; }}
        .thumb.fix {{ border: 2px solid #9C27B0; }}
        
        .checkbox {{
            position: absolute;
            top: 4px;
            left: 4px;
            z-index: 10;
            width: 20px;
            height: 20px;
            cursor: pointer;
            accent-color: #2196F3;
        }}
        .thumb-container.checked .thumb {{
            opacity: 0.6;
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
            background: rgba(0,0,0,0.8);
            padding: 10px 20px;
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
            padding: 10px;
            border-radius: 8px;
            font-size: 12px;
        }}
        .nav kbd {{
            background: #555;
            padding: 2px 6px;
            border-radius: 3px;
            margin: 0 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Workout Gallery</h1>
        <div class="stats">{total} workouts | {pass_count} PASS | {auto_count} AUTO_PASS | {keep_count} LLM_KEEP | {fix_count} LLM_FIX | {failed} failed</div>
        <div class="legend">
            <span><div class="dot pass"></div> PASS</span>
            <span><div class="dot auto"></div> AUTO_PASS</span>
            <span><div class="dot keep"></div> LLM_KEEP</span>
            <span><div class="dot fix"></div> LLM_FIX (needs HR offset)</span>
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
            
            // Visual feedback
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
                    return container.dataset.line;
                }});
            
            if (checked.length === 0) {{
                alert('No workouts checked!');
                return;
            }}
            
            // Create downloadable JSON file
            const data = {{
                'checked_line_numbers': checked.map(Number),
                'total': checked.length,
                'exported_at': new Date().toISOString()
            }};
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'checked_workouts_' + new Date().toISOString().split('T')[0] + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert(`Exported ${{checked.length}} workout line numbers!`);
        }}
    </script>
</body>
</html>'''
    
    # Generate thumbnail HTML
    thumbnails = []
    status_counts = {'PASS': 0, 'AUTO_PASS': 0, 'LLM_KEEP': 0, 'LLM_FIX': 0}
    
    for w in workouts:
        if w['line_number'] in failed_lines:
            continue
            
        status = w['status']
        status_class = {'PASS': 'pass', 'AUTO_PASS': 'auto', 'LLM_KEEP': 'keep', 'LLM_FIX': 'fix'}.get(status, 'pass')
        status_counts[status] = status_counts.get(status, 0) + 1
        
        thumb_path = f"thumbs/{w['line_number']}.png"
        info = f"Line {w['line_number']} | ID: {w['workout_id']} | HR: {w['hr_mean']:.0f} | {w['duration']:.0f}min | {status}"
        
        thumbnails.append(
            f'<div class="thumb-container" data-line="{w["line_number"]}">'
            f'<input type="checkbox" class="checkbox" onclick="event.stopPropagation(); updateCount();">'
            f'<div class="thumb {status_class}" onclick="openModal(\'{thumb_path}\', \'{info}\')">'
            f'<img data-src="{thumb_path}" alt="Workout {w["line_number"]}">'
            f'</div>'
            f'</div>'
        )
    
    html = html_template.format(
        total=len(workouts) - len(failed_lines),
        pass_count=status_counts['PASS'],
        auto_count=status_counts['AUTO_PASS'],
        keep_count=status_counts['LLM_KEEP'],
        fix_count=status_counts['LLM_FIX'],
        failed=len(failed_lines),
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
    print("WORKOUT GALLERY GENERATOR")
    print("=" * 60)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load computed speeds
    speed_lookup = load_computed_speeds()
    
    # Load workout list
    workouts = load_good_workouts()
    
    # Check which thumbnails already exist (for resume capability)
    existing = set()
    for f in THUMBS_DIR.glob("*.png"):
        try:
            existing.add(int(f.stem))
        except:
            pass
    
    to_generate = [w for w in workouts if w['line_number'] not in existing]
    print(f"Thumbnails to generate: {len(to_generate)} (skipping {len(existing)} existing)")
    
    failed_lines = []
    
    if len(to_generate) == 0:
        print("All thumbnails already exist!")
    else:
        # Prepare tasks (include raw data path and speed lookup)
        tasks = [(w, THUMBS_DIR / f"{w['line_number']}.png", RAW_DATA, speed_lookup) for w in to_generate]
        
        # Process with multiprocessing
        print(f"\nGenerating thumbnails with {NUM_WORKERS} workers...")
        start_time = time.time()
        
        completed = 0
        
        # Process in smaller batches to reduce memory pressure
        chunk_size = BATCH_SIZE
        total_tasks = len(tasks)
        
        for chunk_start in range(0, total_tasks, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_tasks)
            chunk_tasks = tasks[chunk_start:chunk_end]
            
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(generate_thumbnail, task): task for task in chunk_tasks}
                
                for future in as_completed(futures):
                    success, line_num, error = future.result()
                    completed += 1
                    
                    if not success:
                        failed_lines.append(line_num)
                    
                    # Progress update
                    if completed % 100 == 0 or completed == total_tasks:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (total_tasks - completed) / rate if rate > 0 else 0
                        print(f"  [{completed}/{total_tasks}] {rate:.1f}/s | ETA: {eta/60:.1f}min | Failed: {len(failed_lines)}")
            
            # Force garbage collection between chunks
            import gc
            gc.collect()
        
        elapsed = time.time() - start_time
        print(f"\nGeneration complete in {elapsed/60:.1f} minutes")
        print(f"  Success: {total_tasks - len(failed_lines)}")
        print(f"  Failed: {len(failed_lines)}")
    
    # Generate HTML
    print("\nGenerating HTML gallery...")
    generate_html(workouts, failed_lines)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Open in browser: file://{HTML_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
