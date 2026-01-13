# SUB3_V2 Project Summary

**Created**: 2025-01-10  
**Last Updated**: 2026-01-13  
**Status**: Data Processing Complete, Ready for Model Training

---

## What is SUB3_V2?

Version 2.0 of the heart rate prediction project, focused on **data quality** and **feature engineering** to achieve better model performance than V1.

### Key Improvements

| Aspect | V1 | V2 | Impact |
|--------|----|----|--------|
| **Data Validation** | None | Manual + Automated | Remove low-quality workouts |
| **Features** | 3 | 11 | Capture temporal dynamics |
| **Masking** | No | Yes | Ignore 43% padding pollution |
| **Splitting** | Random | Stratified | Balance fitness levels |
| **Target MAE** | 13.88 BPM | < 10 BPM | -28% improvement |

---

## Project Structure

```
SUB3_V2/
├── README.md                    # Project overview + baseline metrics
├── AGENTS.md                    # Coding guidelines
├── CLAUDE.md                    # Claude-specific context
├── .gitignore                   # Git ignore rules
│
├── doc/                         # BMAD Documentation ✅
│   ├── GET_STARTED.md
│   ├── PROJECT_SUMMARY.md       # This file
│   ├── ROADMAP.md
│   └── TODO.md
│
├── DATA/
│   ├── raw/                     # Raw Endomondo data (symlink to V1)
│   ├── indices/                 # Line-based indices for fast access ✅
│   └── README.md
│
├── EDA/                         # Complete EDA pipeline ✅
│   ├── 1_data_indexing/         # Build indices (build_index.py, build_speed_index.py)
│   ├── 2_visualization/         # Streamlit app, gallery generator, quick_view
│   ├── 3_quality_analysis/      # Anomaly detection, pattern flagging
│   ├── 4_feature_engineering/   # GPS speed computation
│   ├── outputs/                 # Analysis visualizations
│   ├── gallery/                 # HTML gallery for workout inspection
│   └── README.md                ✅
│
├── Preprocessing/               # 3-stage pipeline ✅
│   ├── stage1_filter.py         # Rule-based filtering ✅
│   ├── stage2_llm_validation.py # LLM-based offset detection ✅
│   ├── stage3_apply_corrections.py # Apply fixes ✅
│   ├── merge_computed_speeds.py # Merge GPS-computed speeds ✅
│   ├── apply_moving_average.py  # Smoothing ✅
│   ├── clean_dataset.json       # Stage 3 output ✅
│   ├── clean_dataset_merged.json # With computed speeds ✅
│   ├── clean_dataset_smoothed.json # Final smoothed data ✅
│   └── README.md                ✅
│
├── Model/                       # To be implemented
│   ├── prepare_sequences.py     # (to be created)
│   ├── lstm.py                  # (to be created)
│   ├── train.py                 # (to be created)
│   ├── loss.py                  # (to be created)
│   ├── evaluate.py              # (to be created)
│   └── README.md
│
├── checkpoints/                 # Model weights (for training)
├── results/                     # Evaluation outputs
└── logs/                        # Training logs
```

---

## Completed Work ✅

### Phase -1: Documentation (Jan 10, 2025)
- ✅ Complete BMAD documentation (GET_STARTED, PROJECT_SUMMARY, ROADMAP, TODO)
- ✅ AGENTS.md with coding guidelines and V2 specifications
- ✅ CLAUDE.md with project context for AI assistants
- ✅ Module READMEs (EDA, Preprocessing)

### Phase 0: Data Exploration & Quality Infrastructure (Jan 11-13, 2026)
- ✅ **Data Indexing** (EDA/1_data_indexing/)
  - Line-based indices for memory-efficient access (61K+ workouts)
  - Speed index for GPS-computed data
  - Apple Watch data processing
  
- ✅ **Visualization Tools** (EDA/2_visualization/)
  - Interactive Streamlit app for workout exploration
  - HTML gallery generator for batch inspection (46K+ thumbnails)
  - Quick matplotlib plotting utility
  
- ✅ **Quality Analysis** (EDA/3_quality_analysis/)
  - Anomaly feature computation
  - Pattern flagging algorithms
  - Sample analysis tools
  - Support for 10% manual annotation strategy
  
- ✅ **Feature Engineering** (EDA/4_feature_engineering/)
  - GPS speed computation from coordinates (~50K workouts)
  - Haversine formula implementation with outlier handling

### Phase 1: Preprocessing Pipeline (Jan 11-13, 2026)
- ✅ **3-Stage Hybrid Pipeline**
  - **Stage 1**: Rule-based filters (stage1_filter.py)
    - Hard filters: Auto-exclude invalid data
    - Soft flags: Send suspicious patterns to LLM
    - Offset detection heuristics
    - Output: stage1_output.json (355KB flagged, 46MB full)
    
  - **Stage 2**: LLM validation (stage2_llm_validation.py)
    - Reviews flagged workouts with visualizations
    - Detects HR offset errors (-20 to -60 BPM)
    - Classifies workouts: INTENSIVE/INTERVALS/STEADY/RECOVERY
    - Output: stage2_output.json (13MB)
    
  - **Stage 3**: Apply corrections (stage3_apply_corrections.py)
    - Applies HR offset fixes
    - Output: clean_dataset.json (560MB)
    
- ✅ **Post-Processing**
  - Merged GPS-computed speeds (merge_computed_speeds.py)
  - Applied moving average smoothing (apply_moving_average.py)
  - Final dataset: clean_dataset_smoothed.json (2.3GB, ~94M lines)
  
- ✅ **Exploration Tools**
  - explore_smoothed.py: Analyze smoothed data
  - view_smoothed.py: Visualize smoothed workouts

---

## Next Steps (Implementation)

### Phase 2: Tensor Preparation & Feature Engineering

**Status**: 🔴 CURRENT PRIORITY  
**Goal**: Transform cleaned JSON data into PyTorch tensors with 11 engineered features.

**Tasks**:
1. Implement feature engineering module
   - [ ] Create `Model/feature_engineering.py`
   - [ ] Implement 8 temporal features (lag, derivatives, rolling, cumulative)
   - [ ] Test on sample workouts
   
2. Implement sequence preparation pipeline
   - [ ] Create `Model/prepare_sequences.py`
   - [ ] Load clean_dataset_smoothed.json
   - [ ] Engineer features for all workouts
   - [ ] Pad/truncate to 500 timesteps with masking
   - [ ] Stratified user splitting (70/15/15)
   - [ ] Normalize features (fit on train only, HR unnormalized)
   - [ ] Save tensors: train.pt, val.pt, test.pt
   
3. Verification
   - [ ] Create verification notebook
   - [ ] Check tensor shapes: [N, 500, 11]
   - [ ] Validate mask correctness
   - [ ] Compute feature-HR correlations (target: >0.35)

**Deliverables**:
- `Model/feature_engineering.py`
- `Model/prepare_sequences.py`
- `DATA/processed/train.pt`, `val.pt`, `test.pt`
- `DATA/processed/metadata.json`
- Verification notebook

### Phase 3: Model Training

**Goal**: Train LSTM with 11 features and masked loss.

**Tasks**:
1. Implement model (`Model/lstm.py`)
2. Implement masked loss (`Model/loss.py`)
3. Implement training loop (`Model/train.py`)
4. Train and validate model

**Deliverables**:
- `Model/lstm.py`
- `Model/loss.py`
- `Model/train.py`
- `checkpoints/best_model.pt`

### Phase 4: Evaluation

**Goal**: Evaluate V2 model and compare with V1 baseline.

**Tasks**:
1. Implement evaluation script (`Model/evaluate.py`)
2. Run test set evaluation
3. Compare V1 vs V2 results
4. Generate performance report

**Deliverables**:
- `Model/evaluate.py`
- `results/test_metrics.json`
- `results/v1_vs_v2_comparison.md`

---

## Key Design Decisions

### 1. Hybrid Preprocessing Pipeline (IMPLEMENTED ✅)

**Approach**: 3-stage rule-based + LLM validation

**Rationale**: Balance automation (85% pass-through) with intelligent review (15% flagged for offset detection).

**Implementation**:
- **Stage 1**: Rule-based filters detect definite errors and suspicious patterns
- **Stage 2**: LLM reviews visualizations to diagnose offset errors and classify workouts
- **Stage 3**: Apply validated corrections to generate clean dataset

**Impact**: Detected and corrected HR offset errors (common issue: -20 to -60 BPM shift)

### 2. Data Quality Focus (IMPLEMENTED ✅)

**Problem**: Raw data contains sensor errors, offset issues, GPS noise.

**Solution**: 
- Comprehensive anomaly catalog (30+ anomaly types)
- Automated detection with heuristics
- LLM validation for complex physiological plausibility
- Visualization tools for manual inspection (gallery, Streamlit app)

**Impact**: Clean dataset ready for training with validated HR ranges

### 3. Memory-Efficient Processing (IMPLEMENTED ✅)

**Problem**: 61K+ workouts in large JSON files (~50GB dataset)

**Solution**: Line-based indexing with `linecache` for random access without loading entire file

**Impact**: Process individual workouts efficiently, enable parallel processing

### 4. Feature Engineering (11 features vs 3) - TO IMPLEMENT

**Rationale**: Weak correlation (0.25) in V1 is partly due to missing temporal features.

**Features**:
- **Lag** (3): Capture physiological delay (HR responds 2 timesteps after speed change)
- **Derivatives** (2): Acceleration/deceleration directly affects HR
- **Rolling** (2): Smooth noise, capture sustained effort
- **Cumulative** (1): Total elevation gain affects fatigue

**Expected impact**: Increase correlation from 0.25 → 0.40

### 5. Masking (NEW in V2) - TO IMPLEMENT

**Problem**: 43% of sequences are padded with repeated last value.

**Solution**: Generate validity masks during preprocessing, ignore padded regions in loss.

**Impact**: Prevents learning on artificial data, cleaner gradients.

### 6. Stratified User Splitting - TO IMPLEMENT

**Problem**: V1 test set had different statistics (higher variance).

**Solution**: Stratify users by fitness level (avg HR proxy) to balance splits.

**Impact**: Reduce distribution mismatch, better generalization.

---

## Success Criteria

### Primary Metrics

| Metric | V1 Baseline | V2 Target | Status |
|--------|-------------|-----------|--------|
| **MAE (base)** | 13.88 BPM | < 10 BPM | Not yet tested |
| **R²** | 0.188 | > 0.35 | Not yet tested |
| **Correlation** | 0.25 | > 0.40 | Not yet tested |

### Secondary Metrics

- **Finetuned MAE**: < 7 BPM (vs 8.94 BPM in V1)
- **Training time**: < 30 min (vs ~20 min in V1)
- **Visual inspection**: Predictions follow trends

---

## Technology Stack

- **Language**: Python 3.11+
- **Deep Learning**: PyTorch 2.0+
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Annotation**: Streamlit
- **Version Control**: Git

---

## References

- **V1 Project**: `/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL`
- **Dataset**: Endomondo HR from FitRec project
- **Baseline**: LSTM Baseline (13.88 BPM MAE)
- **Best V1**: Finetuned Stage 1 (8.94 BPM MAE on Apple Watch)

---

## Documentation Status

- [x] Documentation files (README, AGENTS, GET_STARTED, etc.)
- [x] Module READMEs (EDA, Preprocessing)
- [x] BMAD documentation reference (see AGENTS.md for links)
- [x] CLAUDE.md context file for AI assistants
- [x] Data exploration infrastructure (EDA pipeline)
- [x] Preprocessing pipeline (3-stage hybrid approach)
- [x] Clean dataset generation (smoothed, ~94M lines)
- [ ] Feature engineering implementation (Phase 2)
- [ ] Tensor preparation pipeline (Phase 2)
- [ ] Model implementation (Phase 3)
- [ ] Training pipeline (Phase 3)
- [ ] Evaluation scripts (Phase 4)
- [ ] Final Results Report (after Phase 4)

---

**Project Lead**: Riccardo  
**Technical Documentation**: OpenCode  
**Methodology**: BMAD (Business, Model, Architecture, Development) + Hybrid Pipeline

**Status**: Data processing complete (Jan 13, 2026). Ready for tensor preparation and model training (Phase 2).
