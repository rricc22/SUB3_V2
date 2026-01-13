# SUB3_V2 Development Roadmap

**Project**: Heart Rate Prediction V2.0  
**Timeline**: 4 weeks (flexible)  
**Last Updated**: 2026-01-13

---

## 🎯 Project Vision

Achieve **MAE < 10 BPM** on base model (without finetuning) through data quality validation and feature engineering.

**Current V1 Baseline**: 13.88 BPM MAE  
**V2 Target**: < 10 BPM MAE (-28% improvement)

---

## 📅 Timeline Overview

```
Week 1: Data Quality ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% ✅
Week 2: Preprocessing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% ✅
Week 3: Model Training ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0%
Week 4: Evaluation ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0%
```

**Status**: Data processing complete. Ready for tensor preparation (Phase 2).

---

## 🗺️ Phase 0: Data Quality & Exploration (Week 1) ✅ COMPLETE

**Goal**: Establish data quality baselines and build exploration infrastructure

**Duration**: Jan 10-13, 2026 (3 days)  
**Status**: ✅ COMPLETE

### Completed Milestones

#### M0.1: Data Indexing ✅
- ✅ Built line-based indices for 61K+ workouts
- ✅ Speed index for GPS-computed data
- ✅ Apple Watch data processing
- ✅ Memory-efficient random access via linecache

**Deliverables**:
- ✅ `EDA/1_data_indexing/build_index.py`
- ✅ `EDA/1_data_indexing/build_speed_index.py`
- ✅ `DATA/indices/running_all.txt`, `running_complete.txt`

---

#### M0.2: Visualization Tools ✅
- ✅ Interactive Streamlit app for workout exploration
- ✅ HTML gallery generator (46K+ workout thumbnails)
- ✅ Quick matplotlib plotting utility
- ✅ Gallery with checkbox-based pattern flagging

**Deliverables**:
- ✅ `EDA/2_visualization/explore_workouts.py` (Streamlit app)
- ✅ `EDA/2_visualization/generate_gallery.py`
- ✅ `EDA/2_visualization/quick_view.py`
- ✅ `EDA/gallery/` (HTML inspection interface)

---

#### M0.3: Quality Analysis Infrastructure ✅
- ✅ Anomaly feature computation (30+ anomaly types)
- ✅ Pattern flagging algorithms
- ✅ Sample analysis tools
- ✅ Support for 10% manual annotation strategy

**Deliverables**:
- ✅ `EDA/3_quality_analysis/compute_anomaly_features.py`
- ✅ `EDA/3_quality_analysis/flag_weird_patterns.py`
- ✅ `EDA/3_quality_analysis/analyze_checked_patterns.py`
- ✅ `EDA/3_quality_analysis/analyze_sample.py`

---

#### M0.4: Feature Engineering Tools ✅
- ✅ GPS speed computation from coordinates
- ✅ Haversine formula implementation
- ✅ Outlier handling and smoothing
- ✅ Processed ~50K workouts

**Deliverables**:
- ✅ `EDA/4_feature_engineering/compute_speed_from_gps.py`

**Phase 0 Summary**: ✅ Complete infrastructure for data exploration, quality assessment, and preprocessing

---

## 🗺️ Phase 1: Data Preprocessing (Week 2) ✅ COMPLETE

**Goal**: Clean raw data and prepare for feature engineering

**Duration**: Jan 11-13, 2026 (2 days)  
**Status**: ✅ COMPLETE

### Completed Milestones

#### M1.1: 3-Stage Hybrid Pipeline ✅
- ✅ **Stage 1**: Rule-based filtering (stage1_filter.py)
  - Hard filters for auto-exclusion (invalid HR, timestamps, etc.)
  - Soft flags for LLM review (offset suspects, spikes, anomalies)
  - Offset detection heuristics
  - Processed all 61K+ running workouts
  
- ✅ **Stage 2**: LLM validation (stage2_llm_validation.py)
  - Reviewed flagged workouts with visualizations
  - Detected HR offset errors (-20 to -60 BPM shifts)
  - Classified workouts: INTENSIVE/INTERVALS/STEADY/RECOVERY
  - Provided structured correction parameters
  
- ✅ **Stage 3**: Apply corrections (stage3_apply_corrections.py)
  - Applied HR offset fixes from Stage 2
  - Generated clean dataset

**Deliverables**:
- ✅ `Preprocessing/stage1_filter.py`
- ✅ `Preprocessing/stage2_llm_validation.py`
- ✅ `Preprocessing/stage3_apply_corrections.py`
- ✅ `Preprocessing/stage1_output.json` (355KB flagged, 46MB full)
- ✅ `Preprocessing/stage2_output.json` (13MB)
- ✅ `Preprocessing/clean_dataset.json` (560MB)

---

#### M1.2: Post-Processing ✅
- ✅ Merged GPS-computed speeds (merge_computed_speeds.py)
- ✅ Applied moving average smoothing (apply_moving_average.py)
- ✅ Created exploration tools (explore_smoothed.py, view_smoothed.py)

**Deliverables**:
- ✅ `Preprocessing/merge_computed_speeds.py`
- ✅ `Preprocessing/apply_moving_average.py`
- ✅ `Preprocessing/clean_dataset_merged.json` (1.7GB)
- ✅ `Preprocessing/clean_dataset_smoothed.json` (2.3GB, ~94M lines)
- ✅ `Preprocessing/explore_smoothed.py`
- ✅ `Preprocessing/view_smoothed.py`

**Phase 1 Summary**: ✅ Complete data cleaning pipeline with offset correction and smoothing

---

## 🗺️ Phase 2: Tensor Preparation & Feature Engineering (Current)

**Goal**: Transform cleaned JSON data into PyTorch tensors with 11 engineered features

**Duration**: 3-5 days  
**Dependencies**: Phase 1 complete ✅  
**Risk**: Low (well-specified)  
**Status**: 🔴 IN PROGRESS

### Milestones

#### M2.1: Feature Engineering Implementation (Days 1-2)
- [ ] Create `Model/feature_engineering.py`
- [ ] Implement lag features
  - [ ] speed_lag_2 (HR delay ~2 timesteps)
  - [ ] speed_lag_5 (sustained effort)
  - [ ] altitude_lag_30 (elevation HR lag)
- [ ] Implement derivatives
  - [ ] speed_derivative (acceleration/deceleration)
  - [ ] altitude_derivative (grade changes)
- [ ] Implement rolling statistics
  - [ ] rolling_speed_10 (moving avg ~1 min)
  - [ ] rolling_speed_30 (moving avg ~3 min)
- [ ] Implement cumulative features
  - [ ] cumulative_elevation_gain (fatigue indicator)
- [ ] Test on sample workouts
- [ ] Write unit tests

**Deliverables**:
- `Model/feature_engineering.py` (~200 lines)
- Unit tests for each feature

**Success Criteria**:
- All 8 features generated correctly
- Edge cases handled (first/last timesteps, NaN handling)
- Features validated on sample workouts

---

#### M2.2: Sequence Preparation Pipeline (Days 3-4)
- [ ] Create `Model/prepare_sequences.py`
- [ ] Load clean_dataset_smoothed.json (streaming)
- [ ] Apply feature engineering to all workouts
- [ ] Implement padding/truncation with mask generation
  - [ ] Pad to 500 timesteps
  - [ ] Generate validity masks (1=valid, 0=padded)
- [ ] Implement stratified user splitting
  - [ ] Compute user fitness levels (avg HR proxy)
  - [ ] Stratify by fitness quartiles
  - [ ] Split users 70/15/15
- [ ] Implement normalization
  - [ ] Fit StandardScaler on train data only
  - [ ] Apply to all splits
  - [ ] Keep HR unnormalized (interpretable loss)
- [ ] Save tensors
  - [ ] features: [N, 500, 11]
  - [ ] heart_rate: [N, 500, 1]
  - [ ] mask: [N, 500, 1]
  - [ ] metadata.json
  - [ ] scaler_params.json

**Deliverables**:
- `Model/prepare_sequences.py` (~500 lines)
- `DATA/processed/train.pt`
- `DATA/processed/val.pt`
- `DATA/processed/test.pt`
- `DATA/processed/metadata.json`
- `DATA/processed/scaler_params.json`

**Success Criteria**:
- Pipeline runs without errors
- Tensor shapes correct: [N, 500, 11]
- Mask correctly identifies padded regions
- No user overlap between splits

---

#### M2.3: Verification & Analysis (Day 5)
- [ ] Create verification notebook (`Model/verify_tensors.ipynb`)
- [ ] Load and inspect tensors
- [ ] Verify shapes and data types
- [ ] Check mask correctness
  - [ ] mask.sum() == original_lengths.sum()
  - [ ] Padded regions have mask=0
- [ ] Compute feature-HR correlations
  - [ ] Speed → HR
  - [ ] speed_lag_2 → HR (should be higher!)
  - [ ] Other features → HR
- [ ] Visualize feature distributions
- [ ] Check for data leakage
- [ ] Generate preprocessing report

**Deliverables**:
- `Model/verify_tensors.ipynb`
- Feature correlation report
- Data quality summary

**Success Criteria**:
- Feature-HR correlation > 0.35 (improvement over V1's 0.25)
- No data leakage between splits
- Padding ratio documented
- All checks pass

**Phase 2 Exit Criteria**:
- ✅ Feature engineering implemented
- ✅ PyTorch tensors generated (train/val/test)
- ✅ Feature correlations improved
- ✅ Verification complete

---

## 🗺️ Phase 3: Model Training (Week 3)

**Goal**: Train LSTM with 11 features and masked loss

**Duration**: 5-7 days  
**Dependencies**: Phase 2 complete  
**Risk**: Medium (hyperparameter tuning may take time)  
**Status**: ⏳ PENDING

### Milestones

#### M3.1: Model Implementation (Days 1-2)
- [ ] Create `Model/lstm.py`
- [ ] Implement `HeartRateLSTM_v2` class
  - [ ] input_size=11, hidden_size=128, num_layers=2, dropout=0.3
  - [ ] forward() accepts [B, 500, 11], returns [B, 500, 1]
- [ ] Create `Model/loss.py`
- [ ] Implement `MaskedMSELoss` class
  - [ ] Computes loss * mask, then sum / mask.sum()
- [ ] Test model forward pass
- [ ] Test loss computation with masking
- [ ] Write unit tests

**Deliverables**:
- `Model/lstm.py` (~150 lines)
- `Model/loss.py` (~50 lines)
- Unit tests

**Success Criteria**:
- Model accepts [N, 500, 11] input
- Loss correctly ignores padded regions
- Gradient flows correctly

---

#### M3.2: Training Pipeline (Days 3-4)
- [ ] Create `Model/train.py`
- [ ] Implement dataset and dataloader
- [ ] Implement training loop
  - [ ] Forward pass
  - [ ] Masked loss computation
  - [ ] Backward pass with gradient clipping (1.0)
  - [ ] Optimizer step
- [ ] Implement validation loop
- [ ] Implement early stopping (patience=10)
- [ ] Implement learning rate scheduling (ReduceLROnPlateau)
- [ ] Implement checkpoint saving
- [ ] Add logging (TensorBoard or file)
- [ ] Add command-line arguments

**Deliverables**:
- `Model/train.py` (~500 lines)

**Success Criteria**:
- Training runs without errors
- Loss decreases over epochs
- Checkpoints save correctly

---

#### M3.3: Initial Training Run (Day 5)
- [ ] Train with default hyperparameters
  ```bash
  python3 Model/train.py --epochs 100 --batch_size 16
  ```
- [ ] Monitor training curves
- [ ] Check for overfitting
- [ ] Evaluate on validation set

**Expected Results**:
- Training converges in <50 epochs
- Validation MAE < 13 BPM (at least as good as V1)

**Deliverables**:
- `checkpoints/lstm_v2_baseline.pt`
- `results/training_curves.png`
- Validation metrics

---

#### M3.4: Hyperparameter Tuning (Days 6-7)
- [ ] Experiment 1: hidden_size (64, 128, 256)
- [ ] Experiment 2: dropout (0.2, 0.3, 0.4)
- [ ] Experiment 3: learning_rate (0.0001, 0.0005, 0.001)
- [ ] Compare all results
- [ ] Select best configuration
- [ ] Re-train with best config

**Deliverables**:
- Hyperparameter search results table
- `checkpoints/best_model.pt`

**Phase 3 Exit Criteria**:
- ✅ Model trained successfully
- ✅ Validation MAE < 12 BPM (improvement over V1's 13.88)
- ✅ Best checkpoint saved
- ✅ Training curves healthy

---

## 🗺️ Phase 4: Evaluation & Analysis (Week 4)

**Goal**: Comprehensive evaluation and comparison with V1

**Duration**: 5-7 days  
**Dependencies**: Phase 3 complete  
**Risk**: Low (evaluation is straightforward)  
**Status**: ⏳ PENDING

### Milestones

#### M4.1: Test Set Evaluation (Day 1)
- [ ] Create `Model/evaluate.py`
- [ ] Load best checkpoint
- [ ] Run evaluation on test set
- [ ] Compute metrics (MAE, RMSE, R²)
- [ ] Generate visualizations
  - [ ] Prediction vs actual (sample workouts)
  - [ ] Error distribution
  - [ ] MAE histogram

**Deliverables**:
- `Model/evaluate.py` (~300 lines)
- `results/test_metrics.json`
- Visualization plots

**Success Criteria**:
- Test MAE < 10 BPM (target achieved!)
- R² > 0.35
- Visual predictions follow trends

---

#### M4.2: Ablation Study (Days 2-3)
- [ ] Train with 3 base features only (baseline)
- [ ] Train with base + lag features (6 total)
- [ ] Train with base + lag + derivatives (8 total)
- [ ] Train with base + lag + derivatives + rolling (10 total)
- [ ] Train with all 11 features
- [ ] Compare results in table
- [ ] Analyze feature importance

**Deliverables**:
- Ablation study results table
- Feature importance analysis

**Success Criteria**:
- Each feature group contributes positively
- Lag features show largest improvement

---

#### M4.3: V1 vs V2 Comparison (Day 4)
- [ ] Load V1 baseline results
- [ ] Create comparison table
  - [ ] MAE (V1: 13.88, V2: ?)
  - [ ] RMSE
  - [ ] R²
  - [ ] Training time
- [ ] Analyze improvement sources
  - [ ] Feature engineering contribution
  - [ ] Masking contribution
  - [ ] Quality filtering contribution
- [ ] Generate comparison report

**Deliverables**:
- `results/v1_vs_v2_comparison.md`
- Comparison visualization

**Success Criteria**:
- V2 outperforms V1 on all metrics
- Improvements quantified and explained

---

#### M4.4: Final Report & Documentation (Days 5-7)
- [ ] Write final results report
  - [ ] Executive summary
  - [ ] Detailed results
  - [ ] Ablation study findings
  - [ ] V1 vs V2 comparison
  - [ ] Lessons learned
- [ ] Update README.md with V2 results
- [ ] Update CHANGELOG.md
- [ ] Update PROJECT_SUMMARY.md
- [ ] Generate presentation slides (optional)
- [ ] Code cleanup and documentation
- [ ] Create reproducibility instructions

**Deliverables**:
- `results/FINAL_REPORT_V2.md`
- Updated documentation
- Clean, well-commented code

**Phase 4 Exit Criteria**:
- ✅ Test MAE < 10 BPM (or documented if not achieved)
- ✅ Complete ablation study
- ✅ V1 vs V2 comparison complete
- ✅ Final report published

---

## 🎉 Project Completion Criteria

### Primary Goals

- [x] **Documentation**: Comprehensive BMAD documentation ✅
- [x] **Data Exploration**: Complete EDA infrastructure ✅
- [x] **Data Cleaning**: 3-stage preprocessing pipeline ✅
- [ ] **Tensor Preparation**: Feature engineering + sequence prep
- [ ] **Model Training**: LSTM with 11 features and masking
- [ ] **Evaluation**: Test MAE < 10 BPM

### Success Metrics

| Metric | V1 Baseline | V2 Target | Status |
|--------|-------------|-----------|--------|
| **MAE (base)** | 13.88 BPM | < 10 BPM | ⏳ Pending |
| **R²** | 0.188 | > 0.35 | ⏳ Pending |
| **Correlation** | 0.25 | > 0.40 | ⏳ Pending |

### Deliverables Checklist

**Documentation** (complete):
- [x] README.md
- [x] AGENTS.md
- [x] CLAUDE.md
- [x] ROADMAP.md (this file)
- [x] TODO.md
- [x] PROJECT_SUMMARY.md
- [x] GET_STARTED.md
- [x] Module READMEs (EDA, Preprocessing)

**EDA Pipeline** (complete):
- [x] Data indexing (build_index.py, build_speed_index.py)
- [x] Visualization (Streamlit app, gallery, quick_view)
- [x] Quality analysis (anomaly detection, pattern flagging)
- [x] Feature engineering prep (GPS speed computation)

**Preprocessing** (complete):
- [x] Stage 1: Rule-based filtering
- [x] Stage 2: LLM validation
- [x] Stage 3: Apply corrections
- [x] Post-processing (merge speeds, smoothing)
- [x] Clean dataset (2.3GB, ~94M lines)

**Model** (to be implemented):
- [ ] Model/feature_engineering.py
- [ ] Model/prepare_sequences.py
- [ ] Model/lstm.py
- [ ] Model/loss.py
- [ ] Model/train.py
- [ ] Model/evaluate.py

**Data** (in progress):
- [x] Raw data (symlink to V1)
- [x] Indices (line-based)
- [x] Clean dataset (smoothed)
- [ ] DATA/processed/train.pt
- [ ] DATA/processed/val.pt
- [ ] DATA/processed/test.pt

**Results** (future):
- [ ] checkpoints/best_model.pt
- [ ] results/test_metrics.json
- [ ] results/FINAL_REPORT_V2.md

---

## 🚀 Quick Start

### Current Phase: Phase 2 (Tensor Preparation)

```bash
cd /home/riccardo/Documents/SUB3_V2

# 1. Review clean dataset
python3 Preprocessing/view_smoothed.py --workout-id <ID>

# 2. Start Phase 2: Feature engineering
# See TODO.md for detailed task breakdown

# 3. Implement feature engineering
# Create Model/feature_engineering.py

# 4. Implement sequence preparation
# Create Model/prepare_sequences.py
```

---

## 📊 Progress Tracking

### Overall Progress: 50% (Data Processing Complete)

```
[████████████████░░░░░░░░░░░░░░░░] 50%

✅ Phase -1: Documentation (100%)
✅ Phase  0: Data Exploration (100%)
✅ Phase  1: Data Preprocessing (100%)
🔴 Phase  2: Tensor Preparation (0%) ← YOU ARE HERE
⏳ Phase  3: Model Training (0%)
⏳ Phase  4: Evaluation (0%)
```

### Phase Status

| Phase | Status | Start | End | Duration | Progress |
|-------|--------|-------|-----|----------|----------|
| **Documentation** | ✅ Complete | Jan 10 | Jan 10 | 1 day | 100% |
| **Phase 0: Data Exploration** | ✅ Complete | Jan 11 | Jan 13 | 3 days | 100% |
| **Phase 1: Preprocessing** | ✅ Complete | Jan 11 | Jan 13 | 2 days | 100% |
| **Phase 2: Tensor Prep** | 🔴 In Progress | Jan 13 | - | 3-5 days | 0% |
| **Phase 3: Training** | ⏳ Not Started | - | - | 5-7 days | 0% |
| **Phase 4: Evaluation** | ⏳ Not Started | - | - | 5-7 days | 0% |

---

## 🎯 Risk Management

### High Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **MAE target not achieved** | Medium | High | Accept 10-12 BPM as realistic ceiling |
| **Manual annotation too slow** | Medium | Medium | Reduce to 50 workouts if needed |
| **Feature engineering bugs** | Low | High | Extensive unit testing |
| **Overfitting on small dataset** | Medium | Medium | Strong regularization (dropout 0.3) |

---

## 📝 Notes & Decisions

### Design Decisions

1. **Masking approach chosen** over variable-length sequences (simpler implementation)
2. **Streamlit annotation app** over Label Studio (faster for 100 samples)
3. **11 features** based on EDA correlation analysis
4. **Quality threshold: 70/100** (to be validated in Phase 0)

### Open Questions

- [ ] What annotation tool will be used? (Streamlit recommended)
- [ ] Should we reduce sequence length to 400? (reduces padding from 43% to ~30%)
- [ ] Include attention mechanism in V2.1? (future enhancement)

---

## 🔄 Version History

- **v1.0** (2025-01-10): Initial roadmap created
- **v2.0** (2026-01-13): Updated with completed Phases 0-1
  - Documentation phase completed
  - Data exploration infrastructure completed
  - 3-stage preprocessing pipeline implemented
  - Clean dataset generated (2.3GB smoothed)
  
---

**Next Update**: After Phase 2 completion  
**Owner**: Riccardo  
**Contributors**: Riccardo (implementation), OpenCode (documentation)
