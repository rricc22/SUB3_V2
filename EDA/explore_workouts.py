#!/usr/bin/env python3
"""
Memory-efficient workout explorer for endomondoHR dataset.

Streams workouts from 5GB JSON file without loading entire file into memory.
Displays heart rate and speed time-series for each workout.

Author: OpenCode
Date: 2025-01-11
"""

import streamlit as st
import ast
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import linecache
from datetime import datetime


# Constants - use absolute paths
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent  # EDA -> SUB3_V2
DATA_FILE = PROJECT_ROOT / "DATA" / "raw" / "endomondoHR_proper-002.json"
INDEX_DIR = PROJECT_ROOT / "DATA" / "indices"
COMPUTED_SPEED_FILE = PROJECT_ROOT / "DATA" / "processed" / "running_computed_speed.jsonl"
COMPUTED_SPEED_INDEX = PROJECT_ROOT / "DATA" / "indices" / "computed_speed_offsets.idx"
CACHE_FILE = PROJECT_ROOT / "EDA" / ".workout_index.txt"


def load_index(sport: str = 'running_complete') -> List[int]:
    """Load pre-built index of line numbers."""
    index_file = INDEX_DIR / f"{sport}.txt"
    if not index_file.exists():
        return []
    with open(index_file, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_speed_offset_index() -> Dict[int, int]:
    """Load byte-offset index for computed speeds (lightweight ~585KB)."""
    import struct
    
    offsets = {}
    if not COMPUTED_SPEED_INDEX.exists():
        return offsets
    
    with open(COMPUTED_SPEED_INDEX, 'rb') as f:
        data = f.read()
        # Each entry is 12 bytes: 4 (line_num) + 8 (offset)
        entry_size = 12
        num_entries = len(data) // entry_size
        
        for i in range(num_entries):
            line_num, offset = struct.unpack('<IQ', data[i*entry_size:(i+1)*entry_size])
            offsets[line_num] = offset
    
    return offsets


def get_computed_speed(line_num: int, offset_index: Dict[int, int]) -> Optional[List[float]]:
    """Fetch computed speed for a single workout using byte offset (lazy loading)."""
    if line_num not in offset_index:
        return None
    
    byte_offset = offset_index[line_num]
    
    with open(COMPUTED_SPEED_FILE, 'rb') as f:
        f.seek(byte_offset)
        line = f.readline()
        try:
            data = json.loads(line.decode('utf-8'))
            return data['speed']
        except (json.JSONDecodeError, KeyError):
            return None


def count_lines(filepath: Path) -> int:
    """Count total lines in file efficiently using wc -l."""
    import subprocess
    filepath = Path(filepath).resolve()  # Ensure absolute path
    try:
        result = subprocess.run(['wc', '-l', str(filepath)], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.split()[0])
    except Exception:
        pass
    
    # Fallback: count lines manually (slower but reliable)
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count


def get_line(filepath: Path, line_number: int) -> str:
    """Get a specific line from file (1-indexed)."""
    # Clear linecache to avoid memory buildup
    linecache.clearcache()
    return linecache.getline(str(filepath), line_number)


def parse_workout(line: str) -> Optional[Dict]:
    """Parse a single workout line into a dictionary."""
    try:
        return ast.literal_eval(line.strip())
    except (SyntaxError, ValueError) as e:
        return None


def smooth_data(data: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing."""
    if window <= 1:
        return data
    kernel = np.ones(window) / window
    # Pad to avoid boundary effects
    padded = np.pad(data, (window//2, window//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(data)]


def create_workout_figure(workout: Dict, smooth_window: int = 1, show_raw: bool = True) -> go.Figure:
    """Create a dual-axis plot showing HR and speed.
    
    Args:
        workout: Workout data dict
        smooth_window: Window size for smoothing (1 = no smoothing)
        show_raw: If smoothing, also show raw data as faded line
    """
    hr = np.array(workout['heart_rate'])
    speed = np.array(workout.get('speed', [0] * len(hr)))
    altitude = np.array(workout.get('altitude', [0] * len(hr)))
    timestamps = np.array(workout['timestamp'])
    
    has_speed = 'speed' in workout
    is_computed_speed = workout.get('_speed_source') == 'computed'
    
    # Convert timestamps to relative minutes
    time_minutes = (timestamps - timestamps[0]) / 60
    
    # Find valid data (non-padded) - detect where values become constant
    def find_valid_length(arr: np.ndarray, threshold: int = 10) -> int:
        """Find where data becomes padded (constant values)."""
        for i in range(len(arr) - threshold, 0, -1):
            if len(set(arr[i:i+threshold])) > 1:
                return min(i + threshold, len(arr))
        return len(arr)
    
    valid_len = find_valid_length(hr)
    if has_speed:
        valid_len = min(valid_len, find_valid_length(speed))
    
    # Create subplots with secondary y-axis on first row
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Heart Rate & Speed', 'Altitude Profile'),
        vertical_spacing=0.15,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Apply smoothing if requested
    hr_display = smooth_data(hr[:valid_len], smooth_window) if smooth_window > 1 else hr[:valid_len]
    speed_display = smooth_data(speed[:valid_len], smooth_window) if smooth_window > 1 else speed[:valid_len]
    
    # HR raw trace (faded) if smoothing
    if smooth_window > 1 and show_raw:
        fig.add_trace(
            go.Scatter(
                x=time_minutes[:valid_len],
                y=hr[:valid_len],
                name='HR (raw)',
                line=dict(color='#e74c3c', width=1),
                opacity=0.3,
                hoverinfo='skip'
            ),
            row=1, col=1, secondary_y=False
        )
    
    # HR trace (primary y-axis)
    hr_name = f'Heart Rate (smoothed {smooth_window})' if smooth_window > 1 else 'Heart Rate'
    fig.add_trace(
        go.Scatter(
            x=time_minutes[:valid_len],
            y=hr_display,
            name=hr_name,
            line=dict(color='#e74c3c', width=2),
            hovertemplate='HR: %{y:.0f} BPM<br>Time: %{x:.1f} min<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Speed raw trace (faded) if smoothing
    if has_speed and smooth_window > 1 and show_raw:
        fig.add_trace(
            go.Scatter(
                x=time_minutes[:valid_len],
                y=speed[:valid_len],
                name='Speed (raw)',
                line=dict(color='#3498db', width=1),
                opacity=0.3,
                hoverinfo='skip'
            ),
            row=1, col=1, secondary_y=True
        )
    
    # Speed trace (secondary y-axis)
    if has_speed:
        speed_name = f'Speed (smoothed {smooth_window})' if smooth_window > 1 else 'Speed'
        if is_computed_speed:
            speed_name += ' [GPS]'
        fig.add_trace(
            go.Scatter(
                x=time_minutes[:valid_len],
                y=speed_display,
                name=speed_name,
                line=dict(color='#3498db', width=2),
                hovertemplate='Speed: %{y:.1f} km/h<br>Time: %{x:.1f} min<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
    
    # Altitude trace
    fig.add_trace(
        go.Scatter(
            x=time_minutes[:valid_len],
            y=altitude[:valid_len],
            name='Altitude',
            line=dict(color='#27ae60', width=2),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.2)',
            hovertemplate='Alt: %{y:.0f} m<br>Time: %{x:.1f} min<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Mark padded region if exists
    if valid_len < len(hr):
        fig.add_vrect(
            x0=time_minutes[valid_len-1],
            x1=time_minutes[-1],
            fillcolor="rgba(128, 128, 128, 0.2)",
            layer="below",
            line_width=0,
            annotation_text="Padded",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        height=650,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    # Update y-axes - use selector for secondary_y instead of parameter
    fig.update_yaxes(
        title_text="Heart Rate (BPM)", 
        title_font=dict(color='#e74c3c'),
        tickfont=dict(color='#e74c3c'),
        row=1, col=1
    )
    # Secondary y-axis for speed (yaxis2)
    fig.update_layout(
        yaxis2=dict(
            title_text="Speed (km/h)",
            title_font=dict(color='#3498db'),
            tickfont=dict(color='#3498db'),
            overlaying='y',
            side='right'
        )
    )
    fig.update_yaxes(title_text="Altitude (m)", row=2, col=1)
    
    # Update x-axes
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    
    return fig


def compute_stats(workout: Dict) -> Dict:
    """Compute summary statistics for a workout."""
    hr = np.array(workout['heart_rate'])
    speed = np.array(workout.get('speed', [0] * len(hr)))
    altitude = np.array(workout.get('altitude', [0] * len(hr)))
    timestamps = np.array(workout['timestamp'])
    has_speed = 'speed' in workout
    
    # Duration in minutes
    duration = (timestamps[-1] - timestamps[0]) / 60
    
    # Find valid (non-padded) data
    # Simple heuristic: last non-constant value
    valid_mask = np.ones(len(hr), dtype=bool)
    for i in range(len(hr) - 1, 0, -1):
        if hr[i] == hr[i-1] and speed[i] == speed[i-1]:
            valid_mask[i] = False
        else:
            break
    
    hr_valid = hr[valid_mask]
    speed_valid = speed[valid_mask]
    
    # Speed-HR correlation
    if has_speed and len(hr_valid) > 10:
        correlation = np.corrcoef(speed_valid, hr_valid)[0, 1]
    else:
        correlation = np.nan
    
    # Elevation gain
    alt_diff = np.diff(altitude[valid_mask])
    elevation_gain = np.sum(alt_diff[alt_diff > 0])
    
    return {
        'duration_min': duration,
        'valid_points': np.sum(valid_mask),
        'total_points': len(hr),
        'padding_pct': (1 - np.sum(valid_mask) / len(hr)) * 100,
        'hr_mean': np.mean(hr_valid),
        'hr_std': np.std(hr_valid),
        'hr_min': np.min(hr_valid),
        'hr_max': np.max(hr_valid),
        'speed_mean': np.mean(speed_valid),
        'speed_std': np.std(speed_valid),
        'speed_max': np.max(speed_valid),
        'elevation_gain': elevation_gain,
        'correlation': correlation
    }


def detect_quality_issues(workout: Dict) -> List[str]:
    """Detect potential quality issues in workout data."""
    issues = []
    hr = np.array(workout['heart_rate'])
    speed = np.array(workout.get('speed', [0] * len(hr)))
    has_speed = 'speed' in workout
    
    # HR spikes (>30 BPM change per timestep)
    hr_diff = np.abs(np.diff(hr))
    if np.any(hr_diff > 30):
        issues.append(f"HR spikes detected ({np.sum(hr_diff > 30)} instances)")
    
    # HR flatlines (>10 identical consecutive values)
    flatline_count = 0
    current_run = 1
    for i in range(1, len(hr)):
        if hr[i] == hr[i-1]:
            current_run += 1
            if current_run > 10:
                flatline_count += 1
        else:
            current_run = 1
    if flatline_count > 0:
        issues.append(f"HR flatlines detected ({flatline_count} regions)")
    
    # Impossible speeds (>25 km/h for running)
    if has_speed and workout.get('sport') == 'run' and np.any(speed > 25):
        issues.append(f"Impossible running speed (>{np.max(speed):.1f} km/h)")
    
    # HR out of physiological range
    if np.any(hr < 40) or np.any(hr > 220):
        issues.append(f"HR out of range ({np.min(hr)}-{np.max(hr)} BPM)")
    
    # Zero HR values
    if np.any(hr == 0):
        issues.append(f"Zero HR values ({np.sum(hr == 0)} points)")
    
    return issues


def build_sport_index(filepath: Path) -> Dict[str, List[int]]:
    """Build index of line numbers by sport type."""
    index = {'run': [], 'bike': [], 'other': []}
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            if "'sport': 'run'" in line:
                index['run'].append(i)
            elif "'sport': 'bike'" in line:
                index['bike'].append(i)
            else:
                index['other'].append(i)
            
            # Progress update every 10k lines
            if i % 10000 == 0:
                print(f"Indexed {i} workouts...")
    
    return index


def find_next_matching_workout(filepath: Path, start_line: int, sport: str, direction: int = 1, max_search: int = 1000) -> Optional[int]:
    """Find next workout matching the sport filter.
    
    Args:
        filepath: Path to data file
        start_line: Line to start searching from
        sport: 'run', 'bike', or 'all'
        direction: 1 for forward, -1 for backward
        max_search: Maximum lines to search
    
    Returns:
        Line number of matching workout, or None if not found
    """
    import subprocess
    
    if sport == 'all':
        return start_line
    
    sport_key = "'sport': 'run'" if sport == 'run' else "'sport': 'bike'"
    
    current = start_line
    for _ in range(max_search):
        if current < 1:
            return None
            
        # Get line using sed (memory efficient)
        result = subprocess.run(
            ['sed', '-n', f'{current}p', str(filepath)],
            capture_output=True, text=True
        )
        line = result.stdout
        
        if sport_key in line:
            # Also check if it has speed data for running
            if sport == 'run' and "'speed':" not in line:
                current += direction
                continue
            return current
        
        current += direction
    
    return None


def main():
    st.set_page_config(
        page_title="Workout Explorer",
        page_icon="🏃",
        layout="wide"
    )
    
    st.title("🏃 Endomondo Workout Explorer")
    st.markdown("**Memory-efficient viewer for 5GB workout dataset**")
    
    # Sidebar controls
    st.sidebar.header("Navigation")
    
    # Get total lines (cached in session state)
    if 'total_workouts' not in st.session_state:
        with st.spinner("Counting workouts..."):
            st.session_state.total_workouts = count_lines(DATA_FILE)
    
    total = st.session_state.total_workouts
    st.sidebar.metric("Total Workouts", f"{total:,}")
    
    # Sport filter - use pre-built indices
    sport_filter = st.sidebar.selectbox(
        "Filter by Sport",
        ["Running - Original (11.5K)", "Running - GPS Computed (49.9K)", "Running - All (61.4K)", "Cycling (complete)"],
        index=2  # Default to all running
    )
    
    # Load appropriate index
    if sport_filter == "Running - Original (11.5K)":
        index_name = 'running_complete'
    elif sport_filter == "Running - GPS Computed (49.9K)":
        index_name = 'running_computed_speed'
    elif sport_filter == "Running - All (61.4K)":
        index_name = 'running_all'  # We'll create this combined index
    elif sport_filter == "Cycling (complete)":
        index_name = 'cycling_complete'
    else:
        index_name = None
    
    # Load computed speeds offset index (lightweight ~585KB instead of ~1-2GB)
    if 'speed_offset_index' not in st.session_state:
        with st.spinner("Loading speed index..."):
            st.session_state.speed_offset_index = load_speed_offset_index()
    speed_offset_index = st.session_state.speed_offset_index
    
    # Load index into session state (cached)
    if index_name:
        cache_key = f'index_{index_name}'
        if cache_key not in st.session_state:
            with st.spinner(f"Loading {index_name} index..."):
                st.session_state[cache_key] = load_index(index_name)
        workout_lines = st.session_state[cache_key]
        st.sidebar.success(f"Filtered: {len(workout_lines):,} workouts")
    else:
        workout_lines = None
    
    # Initialize position in filtered list
    if 'filter_position' not in st.session_state:
        st.session_state.filter_position = 0
    
    # Navigation method
    nav_method = st.sidebar.radio(
        "Navigation",
        ["Sequential", "Random", "By Line Number"]
    )
    
    if nav_method == "By Line Number":
        line_num = st.sidebar.number_input(
            "Line Number",
            min_value=1,
            max_value=total,
            value=workout_lines[st.session_state.filter_position] if workout_lines else 1,
            step=1
        )
        # Find position in index
        if workout_lines and line_num in workout_lines:
            st.session_state.filter_position = workout_lines.index(line_num)
        workout_idx = line_num
    elif nav_method == "Random":
        if st.sidebar.button("🎲 Random Workout"):
            if workout_lines:
                st.session_state.filter_position = np.random.randint(0, len(workout_lines))
            else:
                st.session_state.filter_position = np.random.randint(1, total + 1)
        if workout_lines:
            workout_idx = workout_lines[st.session_state.filter_position]
        else:
            workout_idx = st.session_state.filter_position
    else:  # Sequential
        if workout_lines:
            workout_idx = workout_lines[st.session_state.filter_position]
        else:
            workout_idx = st.session_state.filter_position + 1
    
    # Prev/Next navigation (now instant with index!)
    max_pos = len(workout_lines) - 1 if workout_lines else total - 1
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("⬅️ Prev"):
            st.session_state.filter_position = max(0, st.session_state.filter_position - 1)
            st.rerun()
    with col2:
        if workout_lines:
            st.write(f"{st.session_state.filter_position + 1}/{len(workout_lines)}")
        else:
            st.write(f"{workout_idx:,}")
    with col3:
        if st.button("Next ➡️"):
            st.session_state.filter_position = min(max_pos, st.session_state.filter_position + 1)
            st.rerun()
    
    # Jump controls
    st.sidebar.markdown("---")
    jump_col1, jump_col2 = st.sidebar.columns(2)
    with jump_col1:
        if st.button("⏮️ First"):
            st.session_state.filter_position = 0
            st.rerun()
    with jump_col2:
        if st.button("Last ⏭️"):
            st.session_state.filter_position = max_pos
            st.rerun()
    
    # Position slider for quick navigation
    if workout_lines:
        new_pos = st.sidebar.slider(
            "Position",
            min_value=0,
            max_value=max_pos,
            value=st.session_state.filter_position,
            format="%d"
        )
        if new_pos != st.session_state.filter_position:
            st.session_state.filter_position = new_pos
            st.rerun()
        
        workout_idx = workout_lines[st.session_state.filter_position]
    
    # Visualization options
    st.sidebar.markdown("---")
    st.sidebar.header("Visualization")
    
    smooth_window = st.sidebar.slider(
        "Smoothing Window",
        min_value=1,
        max_value=30,
        value=1,
        help="Moving average window size (1 = no smoothing)"
    )
    
    show_raw = st.sidebar.checkbox("Show raw data", value=True, 
                                   help="Show faded raw data when smoothing")
    
    # Load and display workout
    st.markdown("---")
    
    with st.spinner(f"Loading workout {workout_idx:,}..."):
        line = get_line(DATA_FILE, workout_idx)
        workout = parse_workout(line)
    
    if workout is None:
        st.error(f"Failed to parse workout at line {workout_idx}")
        return
    
    # Inject computed speed if available and workout doesn't have speed (lazy loading)
    if 'speed' not in workout and workout_idx in speed_offset_index:
        computed_speed = get_computed_speed(workout_idx, speed_offset_index)
        if computed_speed:
            workout['speed'] = computed_speed
            workout['_speed_source'] = 'computed'
            speed_source = "GPS (computed)"
        else:
            speed_source = "Not available"
    elif 'speed' in workout:
        workout['_speed_source'] = 'original'
        speed_source = "Original"
    else:
        speed_source = "Not available"
    
    # Show workout data availability
    current_sport = workout.get('sport', 'unknown')
    has_speed = 'speed' in workout
    has_altitude = 'altitude' in workout
    
    # Display workout info
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Sport", workout.get('sport', 'unknown').title())
    with col2:
        st.metric("Gender", workout.get('gender', 'unknown').title())
    with col3:
        st.metric("User ID", workout.get('userId'))
    with col4:
        st.metric("Workout ID", workout.get('id'))
    with col5:
        st.metric("Speed Source", speed_source)
    
    # Compute and display stats
    stats = compute_stats(workout)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{stats['duration_min']:.1f} min")
    with col2:
        st.metric("Avg HR", f"{stats['hr_mean']:.0f} BPM")
    with col3:
        st.metric("Avg Speed", f"{stats['speed_mean']:.1f} km/h")
    with col4:
        st.metric("Elevation Gain", f"{stats['elevation_gain']:.0f} m")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("HR Range", f"{stats['hr_min']:.0f}-{stats['hr_max']:.0f}")
    with col2:
        st.metric("Speed/HR Corr", f"{stats['correlation']:.3f}")
    with col3:
        st.metric("Valid Points", f"{stats['valid_points']}/{stats['total_points']}")
    with col4:
        st.metric("Padding", f"{stats['padding_pct']:.1f}%")
    
    # Quality issues
    issues = detect_quality_issues(workout)
    if issues:
        st.warning("⚠️ **Quality Issues Detected:**\n" + "\n".join(f"- {issue}" for issue in issues))
    else:
        st.success("✅ No obvious quality issues detected")
    
    # Main visualization
    fig = create_workout_figure(workout, smooth_window=smooth_window, show_raw=show_raw)
    st.plotly_chart(fig, width='stretch')
    
    # Raw data expander
    with st.expander("📊 Raw Data"):
        col1, col2 = st.columns(2)
        raw_first = {'heart_rate': workout['heart_rate'][:10]}
        raw_last = {'heart_rate': workout['heart_rate'][-10:]}
        if 'speed' in workout:
            raw_first['speed'] = workout['speed'][:10]
            raw_last['speed'] = workout['speed'][-10:]
        if 'altitude' in workout:
            raw_first['altitude'] = workout['altitude'][:10]
            raw_last['altitude'] = workout['altitude'][-10:]
        with col1:
            st.write("**First 10 data points:**")
            st.json(raw_first)
        with col2:
            st.write("**Last 10 data points:**")
            st.json(raw_last)
    
    # URL link
    if workout.get('url'):
        st.markdown(f"[🔗 View on Endomondo]({workout['url']})")


if __name__ == "__main__":
    main()
