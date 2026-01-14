# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SUB3_V2**: Heart Rate Prediction from Running Data (Version 2.0)

Predicts heart rate time-series from speed and altitude during running workouts using deep learning. The goal is to optimize sub-3-hour marathon training through personalized HR prediction.

**Current Status**: Active development with 3-stage preprocessing pipeline implemented (rule-based filtering, LLM validation, offset correction). V2 target: MAE < 10 BPM (base model).

## Core Architecture

### Data Pipeline (3 Stages)

The preprocessing pipeline uses a hybrid approach combining rule-based filters with LLM validation:

1. **Stage 1** (`Preprocessing/stage1_filter.py`): Rule-based anomaly detection
   - Hard filters: Auto-exclude invalid data (HR out of range, negative timestamps, length mismatches)
   - Soft flags: Send suspicious patterns to LLM (offset errors, spikes, flatlines)
   - Outputs: JSON with PASS/FLAG/EXCLUDE decisions and anomaly details

2. **Stage 2** (`Preprocessing/stage2_llm_validation.py`): LLM-based validation
   - Reviews flagged workouts with visualizations
   - Detects HR offset errors (common issue: HR shifted by -20 to -60 BPM)
   - Classifies workout types: INTENSIVE, INTERVALS, STEADY, RECOVERY
   - Outputs: Structured decisions with correction parameters

3. **Stage 3** (`Preprocessing/stage3_apply_corrections.py`): Apply fixes
   - Applies HR offset corrections from Stage 2
   - Generates clean dataset for model training

### Index-Based Access

The codebase uses line-based indexing for memory-efficient processing of large JSON files:

- `EDA/1_data_indexing/build_index.py`: Creates indices by workout type
- Index files stored in `DATA/indices/` (e.g., `running_all.txt`, `running_complete.txt`)
- Use `linecache.getline(filepath, line_number)` for random access without loading entire file

### Feature Engineering (11 features)

**Base features (3)**:
- speed (km/h), altitude (m), gender (binary)

**Engineered features (8)**:
- Lag: speed_lag_2, speed_lag_5, altitude_lag_30 (captures HR response delay)
- Derivatives: speed_derivative, altitude_derivative (acceleration effects)
- Rolling: rolling_speed_10, rolling_speed_30 (smooths noise)
- Cumulative: cumulative_elevation_gain (fatigue indicator)

**Rationale**: V1 had weak speed→HR correlation (0.25) due to missing temporal features. V2 targets correlation > 0.40.

### Model Architecture

**Target**: LSTM with 11 input features, 128 hidden units, 2 layers, dropout 0.3
- Masked loss: Ignores padded regions (43% of V1 sequences were padding)
- Sequence length: 500 timesteps (padded/truncated)
- Output: HR predictions in BPM (unnormalized for interpretable loss)

## Essential Commands

### Data Exploration
```bash
# Interactive workout viewer (Streamlit)
streamlit run EDA/2_visualization/explore_workouts.py

# Generate HTML gallery for all workouts
python3 EDA/2_visualization/generate_gallery.py

# Quick plot of single workout by line number
python3 EDA/2_visualization/quick_view.py --line 31
```

### Preprocessing Pipeline
```bash
# Stage 1: Rule-based filtering (all 61K running workouts)
python3 Preprocessing/stage1_filter.py --input DATA/raw/endomondoHR_proper-002.json \
    --index DATA/indices/running_all.txt --output Preprocessing/stage1_output.json

# Stage 2: LLM validation (only flagged workouts)
python3 Preprocessing/stage2_llm_validation.py --input Preprocessing/stage1_output.json \
    --output Preprocessing/stage2_output.json

# Stage 3: Apply corrections
python3 Preprocessing/stage3_apply_corrections.py --input Preprocessing/stage2_output.json \
    --output Preprocessing/clean_dataset.json
```

### Quality Analysis
```bash
# Compute anomaly features for specific workouts
python3 EDA/3_quality_analysis/compute_anomaly_features.py --lines 31,33,36

# Analyze manually checked patterns (10% sampling strategy)
python3 EDA/3_quality_analysis/analyze_checked_patterns.py --input checked_workouts.json

# Auto-flag weird patterns based on learned thresholds
python3 EDA/3_quality_analysis/flag_weird_patterns.py --threshold 0.15
```

### Index Building
```bash
# Build line-based indices for fast access
python3 EDA/1_data_indexing/build_index.py

# Build speed index (for GPS-computed speed)
python3 EDA/1_data_indexing/build_speed_index.py
```

### Model Training & Evaluation
```bash
# Train model (run from project root)
python3 Model/train.py --batch-size 64 --dropout 0.4 --lr 0.0005 --patience 3

# Train with GPU (recommended)
python3 Model/train.py --batch-size 128 --dropout 0.4 --lr 0.0005 --patience 5

# Evaluate trained model on test set
python3 Model/evaluate.py --checkpoint Model/checkpoints/best_model.pt \
    --data-dir DATA/processed --output-dir Model/results

# Training tips:
# - Batch size 64-128 prevents high gradient variance (was 16 in initial attempt)
# - Dropout 0.4 helps prevent overfitting (model was overfitting by epoch 1 with 0.3)
# - LR 0.0005 provides stable convergence (0.001 was too high)
# - Early stopping patience 3-5 catches best model quickly
# - Always run from project root (DATA/processed path is relative)
```

### Animations & Visualizations
```bash
# Generate all 5 animation types
python3 Model/animate_predictions.py --checkpoint Model/checkpoints/best_model.pt \
    --data-dir DATA/processed --output-dir Model/animations --fps 20

# Animation types created:
# 1. Gradual reveal (best workout) - Shows speed, HR prediction, ground truth appearing
# 2. Gradual reveal (worst workout) - Demonstrates challenging cases
# 3. Multi-workout comparison - 4 workouts side-by-side
# 4. Feature influence - Speed/altitude effect on HR with cursor tracking
# 5. Error heatmap - Error evolution across multiple workouts

# Investigate specific workout anomalies
python3 Model/investigate_workout.py  # Analyzes workout #7 offset error
```

## Key Design Patterns

### Memory-Efficient Streaming

The dataset is large (253K workouts, ~50GB JSON). Always use streaming:

```python
import linecache

# Read specific workout by line number (from index)
line = linecache.getline('DATA/raw/endomondoHR_proper-002.json', line_number)
workout = ast.literal_eval(line.strip())

# Always clear cache when done
linecache.clearcache()
```

### HR Offset Detection

**Critical problem**: Many workouts have HR sensor offset errors (shifted by -20 to -60 BPM).

**Detection heuristics** (in `stage1_filter.py`):
- Mean HR < 145 BPM during running speed > 10 km/h
- Max HR never reaches 160 BPM in 20+ minute workout
- Sudden HR drop > 50 BPM mid-workout
- Good HR-speed correlation but HR range too low

**LLM review needed**: Stage 2 analyzes plots to determine exact offset value and transition point.

### Masking for Padded Sequences

V1 computed loss on padded regions (43% padding). V2 uses masks:

```python
# Correct approach
mask = torch.ones(seq_len)
mask[original_length:] = 0  # Mask padding
loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum()

# Wrong approach (V1)
loss = ((predictions - targets) ** 2).mean()  # Includes padding pollution
```

## Important Context

### Model Performance

**V1 Baseline:**
- Best model (finetuned): 8.94 BPM MAE
- Baseline LSTM: 13.88 BPM MAE

**V2 Current (2026-01-14):**
- Test MAE: 11.90 BPM (RMSE: 14.42 BPM)
- Training stopped at epoch 6 (early stopping, patience=3)
- Config: batch_size=64, dropout=0.4, lr=0.0005, hidden=128, layers=2
- Dataset: 32,806 train / 7,299 val / 6,145 test samples
- Model params: 204,417 trainable parameters

**V2 Target:** < 10 BPM MAE (base model, no finetuning)

**Known Issues:**
- Model exhibits "regression to mean" - clusters predictions around 145-160 BPM
- Underpredicts high HR (>170 BPM), overpredicts low HR (<130 BPM)
- Some HR offset errors remain in test set (e.g., workout #4600: 31 BPM systematic offset)
- Needs weight decay (L2 regularization) and larger batch sizes for improvement

### Common Issues
1. **Offset errors**: Most common data quality issue (expected running HR: 150-190 BPM)
   - Some offset errors passed preprocessing and remain in test set
   - Example: Test workout #4600 has 31 BPM systematic offset (true HR ~120 BPM at 12.3 km/h pace)
   - Model correctly predicts ~155 BPM, but gets 31 BPM "error" vs incorrect ground truth
   - Signature: Constant error throughout workout, negative speed-HR correlation
2. **Weak correlation**: V1 had speed→HR correlation of only 0.25 (missing temporal features)
3. **Padding pollution**: 43% of sequences are padded (V2 uses masking)
4. **Distribution mismatch**: V1 test set had different statistics (V2 uses stratified splitting)
5. **Regression to mean**: Model clusters predictions around 145-160 BPM (typical training HR range)

### File Organization
- `DATA/raw/`: Raw Endomondo JSON (symlink to V1 project)
- `DATA/processed/`: Preprocessed tensors (train.pt, val.pt, test.pt), metadata, scaler params
- `DATA/indices/`: Line-based indices for fast lookup
- `EDA/`: Analysis scripts organized by purpose (indexing, visualization, quality, features)
- `Preprocessing/`: 3-stage pipeline scripts
- `Model/`: Training, evaluation, and visualization scripts
  - `train.py`: Main training script with W&B integration
  - `evaluate.py`: Test set evaluation with metrics and plots
  - `animate_predictions.py`: Generate 5 animation types
  - `investigate_workout.py`: Deep-dive analysis for anomalous workouts
  - `lstm.py`: HeartRateLSTM_V2 model architecture
  - `loss.py`: Masked loss functions (MSE/MAE) and metrics
  - `checkpoints/`: Saved model weights
  - `results/`: Evaluation outputs (plots, metrics JSON)
  - `animations/`: Generated GIF animations
- `doc/`: BMAD documentation (PRD, Architecture, Data Quality, Roadmap)

### Coding Conventions
- Use `ast.literal_eval()` to parse workout JSON (faster than `json.loads()`)
- Line numbers in indices are 1-based (matches `linecache` convention)
- All scripts accept `--input`, `--output`, `--verbose` flags
- Progress printed every 1000 workouts for long-running operations
- Parallel processing available (14 workers on this machine)

## V2-Specific Considerations

### Expected HR Ranges by Workout Type
These inform offset detection thresholds:
- INTENSIVE: 170-190+ BPM (threshold/tempo runs)
- INTERVALS: 155-185 ↔ 130-155 BPM (work/rest cycles)
- STEADY: 145-170 BPM (easy-moderate continuous)
- RECOVERY: 120-150 BPM (easy jog)

### Anomaly Priority
**Hard exclusions** (Stage 1 auto-exclude):
- HR < 30 or > 220 BPM
- Negative speed or timestamps
- Array length mismatches
- Duration < 5 min or < 50 data points

**Soft flags** (Stage 2 LLM review):
- HR offset suspects (HIGH priority)
- Sudden HR drops/rises
- Speed/altitude spikes
- Large timestamp gaps

### Animation Insights

The 5 animation types provide different perspectives on model performance:

1. **Gradual Reveal (Best)**: Shows model capability on ideal data (MAE ~2.6 BPM)
   - Speed profile reveals workout structure
   - HR prediction tracks ground truth closely
   - Error stays consistently low throughout

2. **Gradual Reveal (Worst)**: Exposes model limitations (MAE ~42 BPM)
   - Often indicates data quality issues (offset errors)
   - Low-intensity workouts where model regresses to mean
   - Useful for identifying preprocessing gaps

3. **Multi-Workout Comparison**: Demonstrates prediction quality distribution
   - Side-by-side comparison of 4 diverse workouts
   - Shows how model performs across intensity ranges
   - Helps identify workout types where model struggles

4. **Feature Influence**: Visualizes speed/altitude → HR causality
   - Red cursor sweeps through time showing current state
   - Reveals HR response lag to speed changes (~30 seconds typical)
   - Highlights altitude effect on cardiovascular demand

5. **Error Heatmap**: Detects systematic vs random errors
   - Green rows = consistently accurate workouts
   - Red rows = systematic issues (offset errors, sensor malfunction)
   - Horizontal red bars = constant offset throughout workout
   - Vertical red spikes = specific time periods with high error

**Debugging with Animations:**
- Constant error across time → HR offset issue in ground truth
- High error at speed changes → Insufficient lag features
- High error on climbs → Altitude features need tuning
- Random spikes → GPS noise or sensor dropouts

### Testing Strategy
When implementing new features:
1. Test on small sample first (use `--limit 100`)
2. Verify on known good/bad workouts (line numbers in gallery)
3. Check edge cases (first/last timesteps for lag features)
4. Validate output format matches downstream consumers
5. Generate animations to visually validate predictions across workout types

## Reference Projects
- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Dataset**: Endomondo HR from FitRec project (253K workouts, ~974 valid running after filtering)
