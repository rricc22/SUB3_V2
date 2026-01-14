"""
Streamlit App for Heart Rate Prediction
Professional version with workout category examples

Author: Riccardo
Date: 2026-01-14
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import io
import base64
import json
from pathlib import Path

# Must be at the top
st.set_page_config(
    page_title="Heart Rate Prediction",
    page_icon="♥",
    layout="wide"
)


# ============================================================================
# SAMPLE WORKOUTS (3 Categories) - Real data from test set
# ============================================================================

@st.cache_data
def load_sample_workouts():
    """Load real workout samples from JSON files."""
    samples = {}
    
    sample_files = {
        "Steady Pace Run": ("sample_steady.json", "steady_workouts.gif", "STEADY"),
        "Interval Training": ("sample_intervals.json", "intervals_workouts.gif", "INTERVALS"),
        "Progressive Run": ("sample_progressive.json", "progressive_workouts.gif", "PROGRESSIVE")
    }
    
    for name, (json_file, gif_file, category) in sample_files.items():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                samples[name] = {
                    "description": get_description(category, data),
                    "speed": data['speed'],
                    "altitude": data['altitude'],
                    "gender": data['gender'],
                    "category": category,
                    "gif": gif_file
                }
        except FileNotFoundError:
            # Fallback to synthetic data if JSON not found
            samples[name] = get_fallback_sample(name, category, gif_file)
    
    return samples

def get_description(category, data):
    """Generate description from real data."""
    speed = np.array(data['speed'])
    altitude = np.array(data['altitude'])
    
    if category == "STEADY":
        return f"Steady pace run: avg {speed.mean():.1f} km/h, {len(speed)} timesteps (~{len(speed)/10:.0f} min)"
    elif category == "INTERVALS":
        return f"Interval workout: variable pace {speed.min():.1f}-{speed.max():.1f} km/h, {len(speed)} timesteps"
    else:
        return f"Progressive run: increasing pace {speed[:50].mean():.1f}→{speed[-50:].mean():.1f} km/h"

def get_fallback_sample(name, category, gif_file):
    """Fallback synthetic data if JSON files not available."""
    if category == "STEADY":
        return {
            "description": "Steady pace run (synthetic example)",
            "speed": [10.0] * 150,
            "altitude": [100.0] * 150,
            "gender": "Male",
            "category": category,
            "gif": gif_file
        }
    elif category == "INTERVALS":
        return {
            "description": "Interval training (synthetic example)",
            "speed": ([8.0] * 30 + [14.0] * 30) * 3,
            "altitude": [100.0] * 180,
            "gender": "Male",
            "category": category,
            "gif": gif_file
        }
    else:
        return {
            "description": "Progressive run (synthetic example)",
            "speed": [8.0 + (i / 200.0) * 5.0 for i in range(200)],
            "altitude": [100.0 + i * 0.5 for i in range(200)],
            "gender": "Male",
            "category": category,
            "gif": gif_file
        }

SAMPLE_WORKOUTS = load_sample_workouts()


# ============================================================================
# FEATURE ENGINEERING (copied from Model/feature_engineering.py)
# ============================================================================

def create_lag_feature(data: np.ndarray, lag: int) -> np.ndarray:
    """Create a lag feature by shifting data forward."""
    lagged = np.empty_like(data)
    lagged[:lag] = data[0]
    lagged[lag:] = data[:-lag]
    return lagged


def compute_derivative(data: np.ndarray) -> np.ndarray:
    """Compute derivative (rate of change) of a signal."""
    derivative = np.zeros_like(data)
    derivative[1:] = np.diff(data)
    return derivative


def compute_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean using pandas."""
    if len(data) < window:
        return np.full_like(data, data.mean(), dtype=float)
    
    series = pd.Series(data)
    rolling = series.rolling(window=window, min_periods=1).mean()
    return rolling.values


def compute_cumulative_elevation_gain(altitude: np.ndarray) -> np.ndarray:
    """Compute cumulative elevation gain (only positive altitude changes)."""
    altitude_diff = np.zeros_like(altitude)
    altitude_diff[1:] = np.diff(altitude)
    elevation_gain = np.maximum(altitude_diff, 0)
    cumulative_gain = np.cumsum(elevation_gain)
    return cumulative_gain


def engineer_features(workout: Dict) -> np.ndarray:
    """
    Engineer all 11 features from a workout dictionary.
    
    Returns: Array of shape [seq_len, 11]
    """
    speed = np.array(workout['speed'], dtype=np.float32)
    altitude = np.array(workout['altitude'], dtype=np.float32)
    gender = 1.0 if workout.get('gender') == "Male" else 0.0
    
    seq_len = len(speed)
    features = np.zeros((seq_len, 11), dtype=np.float32)
    
    # Base features
    features[:, 0] = speed
    features[:, 1] = altitude
    features[:, 2] = gender
    
    # Lag features
    features[:, 3] = create_lag_feature(speed, lag=2)
    features[:, 4] = create_lag_feature(speed, lag=5)
    features[:, 5] = create_lag_feature(altitude, lag=30)
    
    # Derivative features
    features[:, 6] = compute_derivative(speed)
    features[:, 7] = compute_derivative(altitude)
    
    # Rolling mean features
    features[:, 8] = compute_rolling_mean(speed, window=10)
    features[:, 9] = compute_rolling_mean(speed, window=30)
    
    # Cumulative elevation gain
    features[:, 10] = compute_cumulative_elevation_gain(altitude)
    
    return features


# ============================================================================
# MODEL ARCHITECTURE (copied from Model/lstm.py)
# ============================================================================

import torch.nn as nn

class HeartRateLSTM_V2(nn.Module):
    """LSTM model for heart rate prediction from running data."""
    
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super(HeartRateLSTM_V2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)
    
    def forward(self, x, lengths=None):
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    try:
        checkpoint = torch.load('best_model.pt', map_location='cpu')
        
        model = HeartRateLSTM_V2(
            input_size=11,
            hidden_size=checkpoint['model_config']['hidden_size'],
            num_layers=checkpoint['model_config']['num_layers'],
            dropout=checkpoint['model_config']['dropout'],
            bidirectional=checkpoint['model_config']['bidirectional']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_heart_rate(speed_data: List[float], altitude_data: List[float], gender: str) -> Dict:
    """
    Predict heart rate from running data.
    
    Args:
        speed_data: List of speed values (km/h)
        altitude_data: List of altitude values (m)
        gender: "Male" or "Female"
    
    Returns:
        Dictionary with predictions and metadata
    """
    model, checkpoint = load_model()
    
    if model is None:
        return None
    
    # Prepare workout data
    workout = {
        'speed': speed_data,
        'altitude': altitude_data,
        'gender': gender
    }
    
    # Engineer features
    features = engineer_features(workout)  # [seq_len, 11]
    
    # Convert to tensor and add batch dimension
    features_tensor = torch.from_numpy(features).unsqueeze(0)  # [1, seq_len, 11]
    
    # Predict
    with torch.no_grad():
        predictions = model(features_tensor)  # [1, seq_len, 1]
    
    # Extract predictions
    hr_predictions = predictions[0, :, 0].numpy()
    
    return {
        'predictions': hr_predictions,
        'features': features,
        'speed': np.array(speed_data),
        'altitude': np.array(altitude_data),
        'checkpoint_info': checkpoint
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions(result: Dict):
    """Create visualization of predictions."""
    predictions = result['predictions']
    speed = result['speed']
    altitude = result['altitude']
    
    timesteps = np.arange(len(predictions))
    time_minutes = timesteps / 10.0  # Assuming 6-second intervals (10 per minute)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Heart Rate Prediction
    axes[0].plot(time_minutes, predictions, color='red', linewidth=2, label='Predicted HR')
    axes[0].fill_between(time_minutes, predictions - 5, predictions + 5, alpha=0.2, color='red')
    axes[0].set_ylabel('Heart Rate (BPM)', fontsize=12)
    axes[0].set_title('Heart Rate Prediction', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, time_minutes[-1])
    
    # Plot 2: Speed
    axes[1].plot(time_minutes, speed, color='blue', linewidth=2, label='Speed')
    axes[1].set_ylabel('Speed (km/h)', fontsize=12)
    axes[1].set_title('Running Speed', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, time_minutes[-1])
    
    # Plot 3: Altitude
    axes[2].plot(time_minutes, altitude, color='green', linewidth=2, label='Altitude')
    axes[2].fill_between(time_minutes, altitude.min(), altitude, alpha=0.3, color='green')
    axes[2].set_ylabel('Altitude (m)', fontsize=12)
    axes[2].set_xlabel('Time (minutes)', fontsize=12)
    axes[2].set_title('Elevation Profile', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(0, time_minutes[-1])
    
    plt.tight_layout()
    return fig


def display_gif(gif_filename: str):
    """Display GIF animation."""
    try:
        with open(gif_filename, "rb") as f:
            contents = f.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="workout animation" style="width:100%; border-radius:8px;">',
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        st.warning(f"Animation file '{gif_filename}' not found. Showing placeholder.")
        # Create a simple placeholder
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'Animation Preview\nUnavailable', 
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error loading animation: {e}")


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("Heart Rate Prediction for Runners")
    st.markdown("""
    Predict heart rate (BPM) from your running workout data using deep learning.
    This model uses speed and altitude to predict physiological response.
    """)
    
    # Sidebar
    st.sidebar.header("Model Information")
    model, checkpoint = load_model()
    
    if checkpoint:
        st.sidebar.metric("Validation MAE", f"{checkpoint['val_mae']:.2f} BPM")
        st.sidebar.metric("Model Epoch", checkpoint['epoch'])
        st.sidebar.metric("Parameters", f"{checkpoint['model_config']['num_parameters']:,}")
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **Model**: 2-layer LSTM  
        **Input Features**: 11 (speed, altitude, temporal)  
        **Target**: Sub-3-hour marathon training
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Sample Workouts", "Manual Input", "CSV Upload"])
    
    # ========================================================================
    # TAB 1: SAMPLE WORKOUTS (NEW!)
    # ========================================================================
    with tab1:
        st.header("Try Sample Workouts")
        st.markdown("Select a workout category to see heart rate predictions:")
        
        # Create columns for samples
        cols = st.columns(3)
        
        for idx, (sample_name, sample_data) in enumerate(SAMPLE_WORKOUTS.items()):
            with cols[idx]:
                st.subheader(sample_name)
                st.caption(sample_data['description'])
                
                # Display GIF animation
                display_gif(sample_data['gif'])
                
                # Button to run prediction
                if st.button(f"Predict {sample_name}", key=f"btn_{idx}"):
                    with st.spinner("Predicting..."):
                        result = predict_heart_rate(
                            sample_data['speed'],
                            sample_data['altitude'],
                            sample_data['gender']
                        )
                    
                    if result:
                        st.success("Prediction complete!")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Avg HR", f"{result['predictions'].mean():.1f} BPM")
                        col2.metric("Max HR", f"{result['predictions'].max():.1f} BPM")
                        col3.metric("Min HR", f"{result['predictions'].min():.1f} BPM")
                        col4.metric("Duration", f"{len(sample_data['speed'])/10:.1f} min")
                        
                        # Plot
                        fig = plot_predictions(result)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Download button
                        csv_data = pd.DataFrame({
                            'time_seconds': np.arange(len(result['predictions'])) * 6,
                            'speed_kmh': sample_data['speed'],
                            'altitude_m': sample_data['altitude'],
                            'predicted_hr_bpm': result['predictions']
                        })
                        
                        csv_buffer = io.StringIO()
                        csv_data.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"hr_predictions_{sample_name.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
    
    # ========================================================================
    # TAB 2: MANUAL INPUT
    # ========================================================================
    with tab2:
        st.header("Manual Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            st.subheader("Speed Data (km/h)")
            speed_input = st.text_area(
                "Enter speed values (comma-separated)",
                value="10.5,11.0,11.5,12.0,11.8,11.5,11.0,10.5,10.0,9.5",
                height=100
            )
            
        with col2:
            st.subheader("Altitude Data (meters)")
            altitude_input = st.text_area(
                "Enter altitude values (comma-separated)",
                value="100,105,110,115,120,125,130,128,125,120",
                height=100
            )
        
        if st.button("Predict Heart Rate", type="primary"):
            try:
                # Parse input
                speed_data = [float(x.strip()) for x in speed_input.split(',')]
                altitude_data = [float(x.strip()) for x in altitude_input.split(',')]
                
                if len(speed_data) != len(altitude_data):
                    st.error("Speed and altitude must have the same number of values!")
                elif len(speed_data) < 10:
                    st.error("Please provide at least 10 data points.")
                else:
                    with st.spinner("Predicting..."):
                        result = predict_heart_rate(speed_data, altitude_data, gender)
                    
                    if result:
                        st.success("Prediction complete!")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Avg HR", f"{result['predictions'].mean():.1f} BPM")
                        col2.metric("Max HR", f"{result['predictions'].max():.1f} BPM")
                        col3.metric("Min HR", f"{result['predictions'].min():.1f} BPM")
                        col4.metric("Duration", f"{len(speed_data)/10:.1f} min")
                        
                        # Plot
                        fig = plot_predictions(result)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Download button
                        csv_data = pd.DataFrame({
                            'time_seconds': np.arange(len(result['predictions'])) * 6,
                            'speed_kmh': speed_data,
                            'altitude_m': altitude_data,
                            'predicted_hr_bpm': result['predictions']
                        })
                        
                        csv_buffer = io.StringIO()
                        csv_data.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="hr_predictions.csv",
                            mime="text/csv"
                        )
            
            except ValueError as e:
                st.error(f"Invalid input format. Please enter numbers separated by commas. Error: {e}")
    
    # ========================================================================
    # TAB 3: CSV UPLOAD
    # ========================================================================
    with tab3:
        st.header("Upload CSV File")
        st.markdown("""
        Upload a CSV file with columns: `speed`, `altitude` (and optionally `gender`).
        Each row represents one timestep (~6 seconds).
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            if 'speed' not in df.columns or 'altitude' not in df.columns:
                st.error("CSV must contain 'speed' and 'altitude' columns!")
            else:
                gender_csv = st.selectbox("Gender (if not in CSV)", ["Male", "Female"], key="csv_gender")
                
                if st.button("Predict from CSV", type="primary", key="csv_predict"):
                    try:
                        speed_data = df['speed'].tolist()
                        altitude_data = df['altitude'].tolist()
                        gender_value = df['gender'].iloc[0] if 'gender' in df.columns else gender_csv
                        
                        with st.spinner("Predicting..."):
                            result = predict_heart_rate(speed_data, altitude_data, gender_value)
                        
                        if result:
                            st.success("Prediction complete!")
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Avg HR", f"{result['predictions'].mean():.1f} BPM")
                            col2.metric("Max HR", f"{result['predictions'].max():.1f} BPM")
                            col3.metric("Min HR", f"{result['predictions'].min():.1f} BPM")
                            col4.metric("Duration", f"{len(speed_data)/10:.1f} min")
                            
                            # Plot
                            fig = plot_predictions(result)
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Download
                            df['predicted_hr_bpm'] = result['predictions']
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            
                            st.download_button(
                                label="Download Results (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name="hr_predictions_from_csv.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")


if __name__ == "__main__":
    main()
