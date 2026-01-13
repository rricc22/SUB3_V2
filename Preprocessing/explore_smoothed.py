#!/usr/bin/env python3
"""
Interactive Streamlit app to explore smoothed workout data.

Usage:
    streamlit run Preprocessing/explore_smoothed.py

Author: Claude Code
Date: 2026-01-13
"""

import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


@st.cache_data
def load_data(smoothed_path, original_path):
    """Load smoothed and original data (cached)."""
    with open(smoothed_path, 'r') as f:
        smoothed = json.load(f)

    original = None
    if original_path.exists():
        with open(original_path, 'r') as f:
            original = json.load(f)

    return smoothed, original


def create_comparison_plot(original_workout, smoothed_workout):
    """Create interactive comparison plot."""

    # Get timestamps
    timestamps = original_workout.get('timestamp', [])
    if len(timestamps) > 0:
        time_minutes = [(t - timestamps[0]) / 60 for t in timestamps]
    else:
        time_minutes = list(range(len(original_workout.get('heart_rate', []))))

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Heart Rate (BPM)', 'Speed (km/h)', 'Altitude (m)'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    # Heart Rate
    hr_orig = original_workout.get('heart_rate', [])
    hr_smooth = smoothed_workout.get('heart_rate', [])

    if len(hr_orig) > 0:
        fig.add_trace(
            go.Scatter(x=time_minutes[:len(hr_orig)], y=hr_orig,
                      mode='lines', name='Original HR',
                      line=dict(color='lightblue', width=1),
                      opacity=0.6),
            row=1, col=1
        )
    if len(hr_smooth) > 0:
        fig.add_trace(
            go.Scatter(x=time_minutes[:len(hr_smooth)], y=hr_smooth,
                      mode='lines', name='Smoothed HR',
                      line=dict(color='darkblue', width=2)),
            row=1, col=1
        )

    # Speed
    speed_orig = original_workout.get('speed', [])
    speed_smooth = smoothed_workout.get('speed', [])

    if len(speed_orig) > 0:
        fig.add_trace(
            go.Scatter(x=time_minutes[:len(speed_orig)], y=speed_orig,
                      mode='lines', name='Original Speed',
                      line=dict(color='lightsalmon', width=1),
                      opacity=0.6,
                      showlegend=False),
            row=2, col=1
        )
    if len(speed_smooth) > 0:
        fig.add_trace(
            go.Scatter(x=time_minutes[:len(speed_smooth)], y=speed_smooth,
                      mode='lines', name='Smoothed Speed',
                      line=dict(color='darkorange', width=2),
                      showlegend=False),
            row=2, col=1
        )

    # Altitude
    alt_orig = original_workout.get('altitude', [])
    alt_smooth = smoothed_workout.get('altitude', [])

    if len(alt_orig) > 0:
        fig.add_trace(
            go.Scatter(x=time_minutes[:len(alt_orig)], y=alt_orig,
                      mode='lines', name='Original Altitude',
                      line=dict(color='lightgreen', width=1),
                      opacity=0.6,
                      showlegend=False),
            row=3, col=1
        )
    if len(alt_smooth) > 0:
        fig.add_trace(
            go.Scatter(x=time_minutes[:len(alt_smooth)], y=alt_smooth,
                      mode='lines', name='Smoothed Altitude',
                      line=dict(color='darkgreen', width=2),
                      showlegend=False),
            row=3, col=1
        )

    # Update layout
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="BPM", row=1, col=1)
    fig.update_yaxes(title_text="km/h", row=2, col=1)
    fig.update_yaxes(title_text="meters", row=3, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )

    return fig


def main():
    st.set_page_config(page_title="Smoothed Workout Viewer", layout="wide")

    st.title("🏃 Smoothed Workout Data Explorer")
    st.markdown("Compare original vs smoothed workout data (moving average window=6)")

    # File paths
    project_root = Path(__file__).parent.parent
    smoothed_path = project_root / "Preprocessing/clean_dataset_smoothed.json"
    original_path = project_root / "Preprocessing/clean_dataset.json"

    # Check if files exist
    if not smoothed_path.exists():
        st.error(f"Smoothed data file not found: {smoothed_path}")
        return

    # Load data
    with st.spinner("Loading data..."):
        smoothed_data, original_data = load_data(smoothed_path, original_path)

    num_workouts = len(smoothed_data['workouts'])
    st.success(f"✅ Loaded {num_workouts:,} smoothed workouts")

    # Metadata
    if 'metadata' in smoothed_data:
        meta = smoothed_data['metadata']
        if 'smoothing_applied' in meta:
            smooth_meta = meta['smoothing_applied']
            st.info(f"**Smoothing Info**: Window size = {smooth_meta.get('window_size', 'N/A')}, "
                   f"Applied on {smooth_meta.get('timestamp', 'N/A')[:10]}")

    st.markdown("---")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # Workout selection
        workout_idx = st.number_input(
            "Workout Index",
            min_value=0,
            max_value=num_workouts - 1,
            value=0,
            step=1
        )

        st.markdown("---")

        # Quick jump options
        st.subheader("Quick Jump")
        if st.button("🎲 Random Workout"):
            workout_idx = np.random.randint(0, num_workouts)
            st.rerun()

        # Filter by workout type
        if 'workout_type' in smoothed_data['workouts'][0]:
            workout_types = list(set(w.get('workout_type', 'UNKNOWN')
                                    for w in smoothed_data['workouts']))
            selected_type = st.selectbox("Filter by Type", ['All'] + sorted(workout_types))

            if selected_type != 'All':
                filtered_indices = [i for i, w in enumerate(smoothed_data['workouts'])
                                  if w.get('workout_type') == selected_type]
                if filtered_indices:
                    st.info(f"Found {len(filtered_indices)} {selected_type} workouts")
                    if st.button("Go to first"):
                        workout_idx = filtered_indices[0]
                        st.rerun()

    # Get selected workout
    smoothed_workout = smoothed_data['workouts'][workout_idx]
    original_workout = None
    if original_data and workout_idx < len(original_data['workouts']):
        original_workout = original_data['workouts'][workout_idx]

    # Display workout info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Workout ID", smoothed_workout.get('workout_id', 'N/A'))
    with col2:
        st.metric("Type", smoothed_workout.get('workout_type', 'N/A'))
    with col3:
        duration = smoothed_workout.get('duration_min', 0)
        st.metric("Duration", f"{duration:.1f} min")
    with col4:
        points = len(smoothed_workout.get('heart_rate', []))
        st.metric("Data Points", points)

    # Stats
    st.markdown("---")
    st.subheader("Statistics")

    col1, col2, col3 = st.columns(3)

    hr = smoothed_workout.get('heart_rate', [])
    if len(hr) > 0:
        with col1:
            st.markdown("**Heart Rate (Smoothed)**")
            st.write(f"Mean: {np.mean(hr):.1f} BPM")
            st.write(f"Std: {np.std(hr):.1f} BPM")
            st.write(f"Range: {np.min(hr):.0f} - {np.max(hr):.0f} BPM")

    speed = smoothed_workout.get('speed', [])
    if len(speed) > 0:
        with col2:
            st.markdown("**Speed (Smoothed)**")
            st.write(f"Mean: {np.mean(speed):.1f} km/h")
            st.write(f"Std: {np.std(speed):.1f} km/h")
            st.write(f"Max: {np.max(speed):.1f} km/h")

    altitude = smoothed_workout.get('altitude', [])
    if len(altitude) > 0:
        with col3:
            st.markdown("**Altitude (Smoothed)**")
            st.write(f"Min: {np.min(altitude):.0f} m")
            st.write(f"Max: {np.max(altitude):.0f} m")
            elev_gain = np.sum(np.diff(altitude)[np.diff(altitude) > 0])
            st.write(f"Gain: {elev_gain:.0f} m")

    # Plot
    st.markdown("---")
    st.subheader("Time Series Comparison")

    if original_workout:
        fig = create_comparison_plot(original_workout, smoothed_workout)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("💡 Light colors = Original | Dark colors = Smoothed (window=6)")
    else:
        st.warning("Original data not available for comparison. Showing smoothed data only.")

        # Create simple plot for smoothed only
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Heart Rate', 'Speed', 'Altitude'),
                           vertical_spacing=0.08, shared_xaxes=True)

        timestamps = smoothed_workout.get('timestamp', [])
        if len(timestamps) > 0:
            time_minutes = [(t - timestamps[0]) / 60 for t in timestamps]
        else:
            time_minutes = list(range(len(hr)))

        if len(hr) > 0:
            fig.add_trace(go.Scatter(x=time_minutes[:len(hr)], y=hr,
                                    mode='lines', name='HR'), row=1, col=1)
        if len(speed) > 0:
            fig.add_trace(go.Scatter(x=time_minutes[:len(speed)], y=speed,
                                    mode='lines', name='Speed'), row=2, col=1)
        if len(altitude) > 0:
            fig.add_trace(go.Scatter(x=time_minutes[:len(altitude)], y=altitude,
                                    mode='lines', name='Altitude'), row=3, col=1)

        fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous") and workout_idx > 0:
            st.rerun()
    with col3:
        if st.button("Next ➡️") and workout_idx < num_workouts - 1:
            st.rerun()


if __name__ == "__main__":
    main()
