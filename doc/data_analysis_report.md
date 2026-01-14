# Clean Dataset Analysis Report

**File**: `Preprocessing/clean_dataset_smoothed.json` (1.7GB)
**Date**: 2026-01-14
**Total Workouts**: 46,250

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total workouts | 46,250 |
| Points per workout | 500 (all truncated/padded) |
| Avg duration | 70 min |
| Smoothing | 7-point window on HR, speed, altitude |

### Data Sources

| Source | Count |
|--------|-------|
| stage1_pass | 38,102 |
| triage_auto_pass | 8,126 |
| llm_keep | 11 |
| llm_fix | 11 |

---

## Workout Type Distribution

| Type | Count | % | Mean HR | HR Std | Mean Speed | Duration |
|------|-------|---|---------|--------|------------|----------|
| **STEADY** | 25,509 | 55.2% | 155.1 BPM | 10.2 BPM | 11.27 km/h | 68.6 min |
| **RECOVERY** | 17,874 | 38.6% | 135.8 BPM | 10.7 BPM | 10.26 km/h | 72.6 min |
| **INTENSIVE** | 2,633 | 5.7% | 174.1 BPM | 10.7 BPM | 12.25 km/h | 58.2 min |
| UNKNOWN | 223 | 0.5% | 124.8 BPM | 16.9 BPM | 9.52 km/h | 87.6 min |
| INTERVALS | 11 | 0.0% | 155.2 BPM | 18.9 BPM | 9.97 km/h | 115.7 min |

**Note**: INTERVALS severely underrepresented (only 11 workouts). Consider merging with INTENSIVE or treating as STEADY.

---

## Key Correlations

### Per-Workout Correlations (averaged)

| Correlation | Mean | Median | Std | Range | % Positive | % Strong (>0.5) |
|-------------|------|--------|-----|-------|------------|-----------------|
| **HR ↔ Speed** | **0.253** | 0.283 | 0.353 | [-0.89, 0.98] | 75.8% | 27.7% |
| HR ↔ Altitude | 0.014 | 0.030 | 0.345 | [-0.96, 0.95] | 53.5% | 7.5% |
| Speed ↔ Altitude | -0.028 | -0.025 | 0.257 | [-0.99, 0.99] | 45.9% | 2.1% |

### Global Correlations (pooled data)

| Correlation | Value |
|-------------|-------|
| HR ↔ Speed | **0.333** |
| HR ↔ Altitude | -0.005 |

**Critical insight**: Only **28% of workouts** have strong HR-Speed correlation (>0.5). The weak direct correlation (0.25) explains why temporal lag features are essential for HR prediction.

### HR-Speed Correlation by Workout Type

| Type | Mean Corr | Std |
|------|-----------|-----|
| INTENSIVE | 0.142 | 0.342 |
| INTERVALS | 0.290 | 0.364 |
| STEADY | 0.252 | 0.347 |
| RECOVERY | 0.271 | 0.359 |
| UNKNOWN | 0.289 | 0.355 |

---

## Variable Distributions

### Heart Rate (BPM)

| Statistic | Value |
|-----------|-------|
| Min | 30.8 |
| Max | 218.7 |
| Mean | 148.1 |
| Median | 149.6 |
| Std | 18.3 |

**Percentiles**: [5%, 25%, 50%, 75%, 95%] = [117, 138, 150, 160, 175]

### Per-Workout HR Statistics

| Metric | Mean | Std |
|--------|------|-----|
| Workout HR min | 95.4 | 18.0 |
| Workout HR max | 166.1 | 14.1 |
| Workout HR mean | 148.6 | 13.4 |
| Dynamic range (max-min) | 70.7 | 18.8 |

### Speed (km/h)

| Statistic | Value |
|-----------|-------|
| Min | 0.00 |
| Max | 121.77 |
| Mean | 10.91 |
| Median | 10.98 |
| Std | 2.48 |

**Typical pace**: 5.49 min/km (5:30 per km)

**Percentiles**: [5%, 25%, 50%, 75%, 95%] = [6.8, 9.6, 11.0, 12.3, 14.7]

### Speed Distribution (workout means)

| Range | Count | % |
|-------|-------|---|
| 0-8 km/h | 2,905 | 6.3% |
| 8-10 km/h | 11,024 | 23.8% |
| 10-12 km/h | 19,329 | 41.8% |
| 12-14 km/h | 10,646 | 23.0% |
| 14-20 km/h | 2,322 | 5.0% |

### Altitude (m)

| Statistic | Value |
|-----------|-------|
| Min | -500.0 |
| Max | 5,942.5 |
| Mean | 176.4 |
| Median | 72.6 |
| Std | 302.2 |

High variance indicates mix of flat and mountain workouts.

### Workout Duration (min)

| Statistic | Value |
|-----------|-------|
| Min | 8.5 |
| Max | 299.8 |
| Mean | 69.7 |
| Median | 61.1 |

---

## Data Quality Flags

| Issue | Count | % | Implication |
|-------|-------|---|-------------|
| Mean HR < 120 BPM | 1,044 | 2.3% | Possible HR offset errors |
| Max HR > 200 BPM | 398 | 0.9% | Possible sensor spikes |
| HR std < 5 BPM | 1,294 | 2.8% | Flat/stuck HR signal |
| HR-Speed corr < -0.3 | 3,415 | 7.4% | Suspicious inverse relationship |

**~7.4% of workouts show negative HR-Speed correlation**, which is physiologically unexpected during running. These may have:
- Remaining HR offset issues
- GPS/speed data quality problems
- Unusual workout patterns (walking breaks, cooldowns)

---

## Modeling Implications

### Why Direct HR-Speed Correlation is Weak (0.25)

1. **HR Response Lag**: Heart rate responds ~30-60 seconds after speed changes
2. **Individual Variation**: Different fitness levels produce different HR at same speed
3. **Fatigue Effect**: HR drifts upward over time at constant speed (cardiac drift)
4. **Temperature/Hydration**: External factors affect HR independently of speed

### Recommendations

1. **Temporal Features are Essential**
   - Lag features: `speed_lag_30`, `speed_lag_60`
   - Rolling averages: `rolling_speed_30`, `rolling_speed_60`
   - Cumulative features: `cumulative_distance`, `elapsed_time`

2. **Altitude Needs Derivatives**
   - Raw altitude correlation near zero (0.014)
   - Use `altitude_derivative` (climbing rate) instead
   - `cumulative_elevation_gain` for fatigue modeling

3. **Consider Workout Type as Feature**
   - Clear HR separation: INTENSIVE (174) vs RECOVERY (136)
   - Could improve predictions if type is known

4. **Address Remaining Data Quality Issues**
   - 3,415 workouts with negative correlation warrant review
   - 1,044 low-HR workouts may have offset errors

5. **Sequence Padding**
   - All sequences padded to 500 points
   - Masking correctly implemented in loss function

---

## Summary Statistics for Quick Reference

```
Dataset: 46,250 workouts × 500 timesteps
HR: 148 ± 18 BPM (range 31-219)
Speed: 10.9 ± 2.5 km/h (5:30/km pace)
HR-Speed correlation: 0.25 (per-workout), 0.33 (pooled)
Dynamic HR range: 71 ± 19 BPM per workout
Data quality issues: ~7% suspicious workouts
```
