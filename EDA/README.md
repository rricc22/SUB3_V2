# EDA (Exploratory Data Analysis)

Scripts for analyzing and visualizing the Endomondo running workout dataset.

## Directory Structure

```
EDA/
├── 1_data_indexing/        # Scripts for building data indices
├── 2_visualization/         # Interactive and static visualization tools
├── 3_quality_analysis/      # Quality assessment and anomaly detection
├── 4_feature_engineering/   # Speed computation and feature extraction
├── outputs/                 # Generated plots and analysis results
├── gallery/                 # HTML gallery for workout inspection
└── README.md               # This file
```

## Quick Start

### 1. View Individual Workouts
```bash
# Interactive Streamlit app
streamlit run 2_visualization/explore_workouts.py

# Quick matplotlib view
python3 2_visualization/quick_view.py --line 31
```

### 2. Browse All Workouts
```bash
# Generate HTML gallery (if not already generated)
python3 2_visualization/generate_gallery.py

# Open in browser
firefox EDA/gallery/index.html
```

### 3. Analyze Data Quality
```bash
# Compute anomaly features for workouts
python3 3_quality_analysis/compute_anomaly_features.py --lines 31,33,36

# Analyze checked patterns from gallery
python3 3_quality_analysis/analyze_checked_patterns.py --input checked_workouts.json

# Auto-flag weird patterns in dataset
python3 3_quality_analysis/flag_weird_patterns.py --threshold 0.15
```

### 4. Feature Engineering
```bash
# Compute speed from GPS for workouts missing speed data
python3 4_feature_engineering/compute_speed_from_gps.py

# Build speed index for fast lookups
python3 4_feature_engineering/build_speed_index.py
```

## Workflow

### Phase 1: Data Exploration
1. Use `explore_workouts.py` to interactively browse workouts
2. Generate gallery with `generate_gallery.py` for batch inspection
3. Use checkboxes in gallery to mark interesting/problematic workouts

### Phase 2: Quality Assessment (Your 10% Annotation Strategy)
1. **Manual**: Check ~4,600 workouts (10%) in gallery for weird patterns:
   - Stable speed but rising HR
   - Speed drops but HR increases  
   - Negative/zero correlation between HR and speed
   
2. **Export**: Click "Export Checked" in gallery → saves `checked_workouts_YYYY-MM-DD.json`

3. **Analyze**: Run `analyze_checked_patterns.py` to extract patterns:
   ```bash
   python3 3_quality_analysis/analyze_checked_patterns.py \
       --input checked_workouts_2026-01-13.json \
       --sample-size 4600
   ```
   This computes features for checked vs unchecked workouts and suggests thresholds.

4. **Auto-flag**: Apply learned thresholds to remaining 90%:
   ```bash
   python3 3_quality_analysis/flag_weird_patterns.py \
       --threshold 0.15 \
       --output flagged_workouts.json
   ```

### Phase 3: Dataset Finalization
1. Review flagged workouts in gallery
2. Export final exclusion list
3. Use in preprocessing pipeline

## Scripts Reference

### 1_data_indexing/
- `build_index.py` - Build line offset index for fast random access
- `build_speed_index.py` - Index computed speed data

### 2_visualization/
- `explore_workouts.py` - Streamlit app for interactive exploration
- `generate_gallery.py` - Generate HTML thumbnail gallery (~46K workouts)
- `quick_view.py` - Fast matplotlib plot of single workout

### 3_quality_analysis/
- `compute_anomaly_features.py` - Extract physiological anomaly features
- `analyze_checked_patterns.py` - Analyze manually checked workouts (10% sample)
- `flag_weird_patterns.py` - Auto-flag anomalies in remaining 90%
- `analyze_sample.py` - Quick statistical analysis of random samples

### 4_feature_engineering/
- `compute_speed_from_gps.py` - Calculate speed from GPS coordinates
  - Uses Haversine formula
  - Handles outliers and smoothing
  - ~50K workouts processed

## Data Quality Patterns

### Good Workout Characteristics
- HR-speed correlation: 0.3 - 0.6
- HR increases with speed (on flat terrain)
- HR stabilizes at constant pace
- Smooth, physiologically plausible transitions

### Weird Patterns to Flag
1. **Inverse relationship**: Speed ↓, HR ↑
2. **No relationship**: Correlation < 0.15
3. **Unstable steady state**: Flat speed, rising HR
4. **Sensor artifacts**: Extreme noise, impossible values

## Tips

- **Gallery performance**: Uses lazy loading, handles 46K+ thumbnails smoothly
- **Parallel processing**: Most scripts use multiprocessing (14 workers on this machine)
- **Memory efficiency**: Streaming JSON parsing for large datasets
- **Resume capability**: Scripts detect existing outputs and skip completed work

## See Also

- `/Preprocessing/` - Data cleaning and preprocessing pipeline
- `/Model/` - Training and evaluation scripts
- `PROJECT_SUMMARY.md` - High-level project overview
- `AGENTS.md` - Agent guidelines and project context
