# Data Quality Criteria & Validation
## SUB3_V2 Heart Rate Prediction

**Version**: 2.0  
**Last Updated**: 2025-01-10

---

## Overview

**Problem in V1**: Preprocessing focused on format validation (arrays exist, no NaN) but not sensor quality.

**Impact**: Models trained on low-quality data (HR spikes, GPS noise, sensor dropouts) → poor generalization.

**V2 Solution**: Manual validation of 50-100 workouts before preprocessing to establish quality baselines.

---

## Quality Dimensions

### 1. Heart Rate Sensor Quality

#### Common Issues

| Issue | Description | Detection Method | Action |
|-------|-------------|------------------|--------|
| **Spikes** | Sudden jumps (>30 BPM in 1 sec) | `np.diff(hr) > 30` | Flag/remove workout |
| **Flatlines** | Prolonged identical values (>10 sec) | `consecutive_duplicates > 10` | Flag/remove workout |
| **Dropouts** | HR = 0 mid-workout | `hr[10:-10] == 0` | Interpolate or remove |
| **Out-of-range** | HR < 40 or > 220 BPM | Direct check | Already filtered in V1 |
| **Strap disconnect** | HR drops to 60-70 suddenly | Pattern detection | Flag for review |

#### Validation Function

```python
def validate_hr_sensor(hr_array):
    """
    Validate heart rate sensor quality.
    
    Returns:
        (is_valid, quality_score, issues)
    """
    issues = []
    quality_score = 100.0
    
    # Check 1: Spikes
    hr_diff = np.abs(np.diff(hr_array))
    spike_count = np.sum(hr_diff > 30)
    if spike_count > 0:
        issues.append(f"HR spikes: {spike_count}")
        quality_score -= spike_count * 10
    
    # Check 2: Flatlines
    flatline_lengths = find_consecutive_duplicates(hr_array)
    max_flatline = max(flatline_lengths) if flatline_lengths else 0
    if max_flatline > 10:
        issues.append(f"HR flatline: {max_flatline} timesteps")
        quality_score -= 20
    
    # Check 3: Dropouts
    dropout_count = np.sum(hr_array[10:-10] == 0)
    if dropout_count > 0:
        issues.append(f"HR dropouts: {dropout_count}")
        quality_score -= dropout_count * 5
    
    # Check 4: Strap disconnect pattern
    if detect_strap_disconnect(hr_array):
        issues.append("Possible strap disconnect")
        quality_score -= 30
    
    is_valid = quality_score >= 70  # Threshold
    return is_valid, max(0, quality_score), issues
```

---

### 2. GPS/Speed Quality

#### Common Issues

| Issue | Description | Detection Method | Action |
|-------|-------------|------------------|--------|
| **GPS noise** | Speed oscillates rapidly | `std(speed_diff) > threshold` | Smooth or flag |
| **Impossible speeds** | Speed > 25 km/h sustained | `speed > 25 for >5 sec` | Remove workout |
| **Pauses** | Speed = 0 mid-run | `speed == 0 for >10 sec` | Segment or remove |
| **GPS drift** | Speed unrealistic for terrain | Context check | Manual review |

#### Validation Function

```python
def validate_gps_speed(speed_array):
    """
    Validate GPS and speed measurements.
    
    Returns:
        (is_valid, quality_score, issues)
    """
    issues = []
    quality_score = 100.0
    
    # Check 1: Impossible speeds (sustained >25 km/h)
    impossible_count = np.sum(speed_array > 25)
    if impossible_count > 5:  # >5 timesteps
        issues.append(f"Impossible speed: {impossible_count} timesteps")
        return False, 0, issues  # Hard fail
    
    # Check 2: GPS noise (rapid oscillations)
    speed_diff = np.abs(np.diff(speed_array))
    noise_level = np.std(speed_diff)
    if noise_level > 2.0:  # Threshold tuned from manual inspection
        issues.append(f"GPS noise: std={noise_level:.2f}")
        quality_score -= 15
    
    # Check 3: Pauses (not necessarily bad, but flag)
    pause_count = np.sum(speed_array == 0)
    if pause_count > 10:
        issues.append(f"Pauses: {pause_count} timesteps")
        quality_score -= 10  # Mild penalty
    
    # Check 4: Speed variability (should be relatively smooth)
    smoothness = compute_smoothness(speed_array)
    if smoothness < 0.7:  # Low smoothness = jerky
        issues.append(f"Jerky speed: smoothness={smoothness:.2f}")
        quality_score -= 10
    
    is_valid = quality_score >= 60
    return is_valid, max(0, quality_score), issues
```

---

### 3. Altitude Quality

#### Common Issues

| Issue | Description | Detection Method | Action |
|-------|-------------|------------------|--------|
| **Altitude drift** | Sudden jumps (>100m in 1 sec) | `np.diff(altitude) > 100` | Flag workout |
| **Barometric errors** | Inconsistent with GPS | Compare sources | Use GPS if available |
| **Noise** | High-frequency oscillations | Spectral analysis | Smooth |

#### Validation Function

```python
def validate_altitude(altitude_array):
    """
    Validate altitude measurements.
    
    Returns:
        (is_valid, quality_score, issues)
    """
    issues = []
    quality_score = 100.0
    
    # Check 1: Altitude jumps (>100m sudden change)
    altitude_diff = np.abs(np.diff(altitude_array))
    jump_count = np.sum(altitude_diff > 100)
    if jump_count > 0:
        issues.append(f"Altitude jumps: {jump_count}")
        quality_score -= jump_count * 20
    
    # Check 2: High-frequency noise
    altitude_smooth = savgol_filter(altitude_array, 11, 3)
    noise_level = np.mean(np.abs(altitude_array - altitude_smooth))
    if noise_level > 10:  # >10m average deviation
        issues.append(f"Altitude noise: {noise_level:.1f}m")
        quality_score -= 15
    
    is_valid = quality_score >= 60
    return is_valid, max(0, quality_score), issues
```

---

### 4. Temporal Consistency

#### Common Issues

| Issue | Description | Detection Method | Action |
|-------|-------------|------------------|--------|
| **Irregular sampling** | Timesteps >30 sec apart | `np.diff(timestamps) > 30` | Resample or flag |
| **Missing timestamps** | Not monotonically increasing | Check order | Remove workout |
| **Duplicate timestamps** | Same timestamp repeated | `np.diff(timestamps) == 0` | Remove duplicates |

#### Validation Function

```python
def validate_timestamps(timestamp_array):
    """
    Validate temporal consistency.
    
    Returns:
        (is_valid, quality_score, issues)
    """
    issues = []
    quality_score = 100.0
    
    # Check 1: Monotonically increasing
    if not np.all(np.diff(timestamp_array) > 0):
        issues.append("Non-monotonic timestamps")
        return False, 0, issues  # Hard fail
    
    # Check 2: Sampling rate regularity
    dt = np.diff(timestamp_array)
    median_dt = np.median(dt)
    irregular_count = np.sum(np.abs(dt - median_dt) > 10)  # >10s deviation
    if irregular_count > len(dt) * 0.1:  # >10% irregular
        issues.append(f"Irregular sampling: {irregular_count} gaps")
        quality_score -= 20
    
    # Check 3: Large gaps (>30 sec)
    large_gaps = np.sum(dt > 30)
    if large_gaps > 0:
        issues.append(f"Large gaps: {large_gaps} (>30s)")
        quality_score -= large_gaps * 10
    
    is_valid = quality_score >= 70
    return is_valid, max(0, quality_score), issues
```

---

### 5. Workout Validity

#### Label Accuracy

**Problem**: Some workouts labeled "run" may actually be:
- Walking (speed < 5 km/h sustained)
- Cycling (mislabeled, speed > 20 km/h)
- Mixed activity (run + walk intervals)

#### Validation

```python
def validate_activity_type(speed_array, expected_sport='run'):
    """
    Validate workout is actually the labeled sport.
    
    Returns:
        (is_valid, confidence, issues)
    """
    issues = []
    
    # Running: typical 8-15 km/h
    median_speed = np.median(speed_array)
    
    if expected_sport == 'run':
        if median_speed < 5:
            issues.append(f"Too slow for running: {median_speed:.1f} km/h")
            return False, 0.2, issues
        elif median_speed > 20:
            issues.append(f"Too fast for running: {median_speed:.1f} km/h (cycling?)")
            return False, 0.3, issues
        else:
            confidence = 1.0 if 8 <= median_speed <= 15 else 0.7
            return True, confidence, issues
    
    return True, 1.0, []
```

---

## Manual Annotation Workflow

### Step 1: Sample Selection

```python
def sample_workouts_for_annotation(all_workouts, n=100):
    """
    Select diverse sample for manual inspection.
    
    Strategy:
    - 50% random
    - 25% edge cases (very short, very long)
    - 25% suspicious (failed automated checks)
    """
    # Random baseline
    random_sample = random.sample(all_workouts, n // 2)
    
    # Edge cases
    sorted_by_length = sorted(all_workouts, key=lambda w: len(w['speed']))
    edge_cases = sorted_by_length[:n//8] + sorted_by_length[-n//8:]
    
    # Suspicious (run automated checks first)
    suspicious = [w for w in all_workouts if has_quality_issues(w)]
    suspicious_sample = random.sample(suspicious, min(n // 4, len(suspicious)))
    
    return random_sample + edge_cases + suspicious_sample
```

### Step 2: Annotation Interface

**Options**:
1. **Jupyter Notebook** (quickest for small scale)
2. **Streamlit App** (better UX, scrollable)
3. **Label Studio** (professional, overkill for 100 samples)

**Recommended**: Streamlit App

```python
# EDA/quality_annotation_app.py
import streamlit as st
import plotly.graph_objects as go

st.title("Workout Quality Annotation")

# Load workout
workout_id = st.slider("Workout ID", 0, len(workouts)-1, 0)
w = workouts[workout_id]

# Plot time-series
fig = go.Figure()
fig.add_trace(go.Scatter(y=w['speed'], name='Speed'))
fig.add_trace(go.Scatter(y=w['altitude'], name='Altitude', yaxis='y2'))
fig.add_trace(go.Scatter(y=w['heart_rate'], name='HR', yaxis='y3'))
st.plotly_chart(fig)

# Annotation form
quality_score = st.slider("Overall Quality (0-100)", 0, 100, 80)
hr_quality = st.selectbox("HR Sensor", ["Good", "Spikes", "Flatlines", "Dropouts"])
gps_quality = st.selectbox("GPS", ["Good", "Noisy", "Impossible speeds", "Pauses"])
altitude_quality = st.selectbox("Altitude", ["Good", "Jumps", "Noisy"])
is_valid = st.checkbox("Use in training", value=True)
notes = st.text_area("Notes")

# Save annotation
if st.button("Save & Next"):
    save_annotation(workout_id, quality_score, hr_quality, gps_quality, ...)
    st.rerun()  # Load next workout
```

### Step 3: Export Annotations

**Format**: CSV

```csv
workout_id,quality_score,hr_quality,gps_quality,altitude_quality,is_valid,notes
123,95,Good,Good,Good,True,""
456,65,Spikes,Noisy,Good,True,"3 HR spikes around minute 15"
789,30,Flatlines,Good,Jumps,False,"HR stuck at 150 for 5 minutes"
```

---

## Quality Thresholds

### Overall Quality Score

| Score | Label | Action |
|-------|-------|--------|
| 90-100 | Excellent | Use in training |
| 70-89 | Good | Use in training |
| 50-69 | Acceptable | Use with caution |
| 30-49 | Poor | Exclude from training |
| 0-29 | Very Poor | Exclude |

### Component Thresholds

| Component | Threshold | Hard Fail |
|-----------|-----------|-----------|
| HR Quality | ≥ 70 | < 30 (flatlines, major spikes) |
| GPS Quality | ≥ 60 | < 40 (impossible speeds) |
| Altitude Quality | ≥ 60 | < 30 (major jumps) |
| Temporal Quality | ≥ 70 | Non-monotonic |

---

## Implementation Plan

### Phase 0: Setup (Week 1, Day 1-2)

- [ ] Create Streamlit annotation app
- [ ] Sample 100 workouts (diverse + edge cases)
- [ ] Set up annotation CSV template

### Phase 1: Annotation (Week 1, Day 3-5)

- [ ] Manually annotate 100 workouts (~20 per day)
- [ ] Record quality scores and issues
- [ ] Document patterns and edge cases

### Phase 2: Analysis (Week 1, Day 6-7)

- [ ] Analyze annotation distribution
- [ ] Establish quality thresholds
- [ ] Document common failure modes
- [ ] Tune automated validation functions

### Phase 3: Automated Validation (Week 2)

- [ ] Implement `quality_filters.py`
- [ ] Run on all 974 workouts
- [ ] Report quality statistics
- [ ] Filter low-quality workouts

---

## Expected Outcomes

### Quality Statistics (Projected)

| Metric | Expected Value |
|--------|----------------|
| Excellent quality | 40-50% |
| Good quality | 30-40% |
| Acceptable quality | 10-20% |
| Poor quality | 5-10% |
| Very poor quality | 1-5% |

**Target**: Retain 70-80% of workouts (680-780 out of 974)

### Quality Improvements

| Aspect | Before (V1) | After (V2) |
|--------|-------------|------------|
| HR sensor validation | None | Spike/flatline/dropout detection |
| GPS validation | None | Impossible speed detection |
| Altitude validation | None | Jump detection |
| Temporal validation | None | Sampling gap detection |
| Activity validation | Sport label only | Speed-based verification |

---

## Quality Report Template

```markdown
# Data Quality Report - SUB3_V2

**Date**: 2025-01-XX
**Annotator**: Riccardo
**Workouts Annotated**: 100

## Summary Statistics

- **Excellent (90-100)**: 45 workouts (45%)
- **Good (70-89)**: 35 workouts (35%)
- **Acceptable (50-69)**: 12 workouts (12%)
- **Poor (30-49)**: 6 workouts (6%)
- **Very Poor (0-29)**: 2 workouts (2%)

**Recommended for training**: 80/100 (80%)

## Common Issues

### HR Sensor (23 workouts affected)
- Spikes: 12 workouts (>30 BPM jumps)
- Flatlines: 8 workouts (prolonged identical values)
- Dropouts: 3 workouts (HR = 0 mid-workout)

### GPS (15 workouts affected)
- Noise: 10 workouts (rapid oscillations)
- Impossible speeds: 3 workouts (>25 km/h)
- Pauses: 2 workouts (extended stops)

### Altitude (8 workouts affected)
- Jumps: 5 workouts (>100m sudden changes)
- Noise: 3 workouts (high-frequency oscillations)

## Recommendations

1. **Implement spike detection**: Remove HR values with >30 BPM/sec change
2. **Smooth GPS data**: Apply Savitzky-Golay filter (window=11, poly=3)
3. **Flag pauses**: Segment workouts at speed=0 >10 sec
4. **Quality threshold**: Use quality_score ≥ 70 for training

## Next Steps

1. Implement automated quality filters based on findings
2. Run filters on all 974 workouts
3. Report expected retention rate
4. Proceed to preprocessing with high-quality subset
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-10
