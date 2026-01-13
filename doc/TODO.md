# SUB3_V2 TODO List

**Last Updated**: 2026-01-13  
**Current Phase**: Phase 2 - Tensor Preparation & Feature Engineering  
**Overall Progress**: 50% (Data processing complete)

---

## 📋 Quick Status

```
✅ Documentation Complete (Jan 10, 2025)
✅ Phase 0: Data Exploration & Quality Infrastructure (Jan 11-13, 2026)
✅ Phase 1: Data Preprocessing Pipeline (Jan 11-13, 2026)
🔴 Phase 2: Tensor Preparation & Feature Engineering (0/12 tasks) ← CURRENT
⏳ Phase 3: Model Training (0/16 tasks)
⏳ Phase 4: Evaluation (0/15 tasks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: Phases 0-1 complete, Phase 2 in progress
```

---

## 🔴 HIGH PRIORITY - Start Here!

### 🎯 Next Actions (Phase 2)

- [ ] **Day 1-2**: Implement feature engineering module
- [ ] **Day 3-4**: Implement sequence preparation pipeline  
- [ ] **Day 5**: Verify tensors and feature correlations

---

## ✅ Phase -1: Documentation (COMPLETE)

**Completed**: Jan 10, 2025

### Documentation Files (11/11) ✅

- [x] README.md (project overview)
- [x] AGENTS.md (coding guidelines)
- [x] CLAUDE.md (AI assistant context)
- [x] ROADMAP.md (development roadmap)
- [x] TODO.md (this file)
- [x] PROJECT_SUMMARY.md (status snapshot)
- [x] GET_STARTED.md (quick start guide)
- [x] EDA/README.md (EDA documentation)
- [x] Preprocessing/README.md (preprocessing documentation)

**Status**: ✅ 100% Complete

---

## ✅ Phase 0: Data Exploration & Quality Infrastructure (COMPLETE)

**Completed**: Jan 11-13, 2026  
**Duration**: 3 days

### Completed Work ✅

#### Data Indexing
- [x] Built line-based indices for 61K+ workouts (EDA/1_data_indexing/)
- [x] Speed index for GPS-computed data
- [x] Apple Watch data processing
- [x] Memory-efficient random access via linecache

**Files**:
- `EDA/1_data_indexing/build_index.py`
- `EDA/1_data_indexing/build_speed_index.py`
- `EDA/1_data_indexing/process_apple_watch.py`
- `DATA/indices/running_all.txt`, `running_complete.txt`

#### Visualization Tools
- [x] Interactive Streamlit app for workout exploration
- [x] HTML gallery generator (46K+ workout thumbnails)
- [x] Quick matplotlib plotting utility
- [x] Apple Watch gallery

**Files**:
- `EDA/2_visualization/explore_workouts.py`
- `EDA/2_visualization/generate_gallery.py`
- `EDA/2_visualization/generate_apple_watch_gallery.py`
- `EDA/2_visualization/quick_view.py`
- `EDA/gallery/` (HTML interface)

#### Quality Analysis
- [x] Anomaly feature computation (30+ anomaly types)
- [x] Pattern flagging algorithms
- [x] Sample analysis tools
- [x] Support for 10% manual annotation strategy

**Files**:
- `EDA/3_quality_analysis/compute_anomaly_features.py`
- `EDA/3_quality_analysis/flag_weird_patterns.py`
- `EDA/3_quality_analysis/analyze_checked_patterns.py`
- `EDA/3_quality_analysis/analyze_sample.py`

#### Feature Engineering Prep
- [x] GPS speed computation from coordinates (~50K workouts)
- [x] Haversine formula implementation with outlier handling

**Files**:
- `EDA/4_feature_engineering/compute_speed_from_gps.py`

**Phase 0 Summary**: ✅ Complete data exploration and quality infrastructure

---

## ✅ Phase 1: Data Preprocessing Pipeline (COMPLETE)

**Completed**: Jan 11-13, 2026  
**Duration**: 2 days

### Completed Work ✅

#### 3-Stage Hybrid Pipeline
- [x] **Stage 1**: Rule-based filtering (stage1_filter.py)
  - Hard filters for auto-exclusion
  - Soft flags for LLM review  
  - Offset detection heuristics
  - Processed all 61K+ running workouts
  
- [x] **Stage 2**: LLM validation (stage2_llm_validation.py)
  - Reviewed flagged workouts
  - Detected HR offset errors (-20 to -60 BPM)
  - Classified workouts: INTENSIVE/INTERVALS/STEADY/RECOVERY
  - Provided correction parameters
  
- [x] **Stage 3**: Apply corrections (stage3_apply_corrections.py)
  - Applied HR offset fixes
  - Generated clean dataset

**Files**:
- `Preprocessing/stage1_filter.py`
- `Preprocessing/stage2_llm_validation.py`
- `Preprocessing/stage3_apply_corrections.py`
- `Preprocessing/stage1_output.json` (355KB flagged, 46MB full)
- `Preprocessing/stage2_output.json` (13MB)
- `Preprocessing/clean_dataset.json` (560MB)

#### Post-Processing
- [x] Merged GPS-computed speeds
- [x] Applied moving average smoothing
- [x] Created exploration tools

**Files**:
- `Preprocessing/merge_computed_speeds.py`
- `Preprocessing/apply_moving_average.py`
- `Preprocessing/explore_smoothed.py`
- `Preprocessing/view_smoothed.py`
- `Preprocessing/clean_dataset_merged.json` (1.7GB)
- `Preprocessing/clean_dataset_smoothed.json` (2.3GB, ~94M lines)

**Phase 1 Summary**: ✅ Complete data cleaning with offset correction and smoothing

## 🔴 Phase 2: Tensor Preparation & Feature Engineering (IN PROGRESS)

**Goal**: Transform cleaned JSON into PyTorch tensors with 11 engineered features  
**Duration**: 3-5 days  
**Priority**: 🔴 HIGH - CURRENT PHASE

### M1.1: Feature Engineering (Days 8-9) - 0/10

- [ ] Create `Preprocessing/feature_engineering.py`
- [ ] Implement lag features
  - [ ] `create_lag_feature()` helper function
  - [ ] speed_lag_2
  - [ ] speed_lag_5
  - [ ] altitude_lag_30
- [ ] Implement derivative features
  - [ ] speed_derivative (np.diff)
  - [ ] altitude_derivative
- [ ] Implement rolling statistics
  - [ ] rolling_speed_10 (pandas rolling)
  - [ ] rolling_speed_30
- [ ] Implement cumulative features
  - [ ] cumulative_elevation (cumsum of positive altitude_derivative)
- [ ] Write unit tests for each feature
- [ ] Test on sample workout

**Deliverables**:
- `Preprocessing/feature_engineering.py` (150-200 lines)
- Unit tests

**Time Estimate**: 4-5 hours

---

### M1.2: Main Preprocessing Pipeline (Days 10-11) - 0/10

- [ ] Create `Preprocessing/prepare_sequences.py`
- [ ] Implement data loading
  - [ ] Load raw JSON
  - [ ] Apply quality filters
  - [ ] Track filtering statistics
- [ ] Implement feature engineering integration
  - [ ] Call `engineer_features()` for each workout
  - [ ] Concatenate base + engineered features
- [ ] Implement padding with masking
  - [ ] `pad_or_truncate_with_mask()` function
  - [ ] Generate validity masks (1=valid, 0=padded)
- [ ] Implement stratified user splitting
  - [ ] Compute user fitness levels (avg HR)
  - [ ] Stratify by fitness quartile
  - [ ] Split users 70/15/15
- [ ] Implement normalization
  - [ ] Fit StandardScaler on train data only
  - [ ] Apply to all splits
  - [ ] Keep HR unnormalized
- [ ] Implement tensor conversion
  - [ ] features: [N, 500, 11]
  - [ ] heart_rate: [N, 500, 1]
  - [ ] mask: [N, 500, 1]
- [ ] Save preprocessed data
  - [ ] train.pt, val.pt, test.pt
  - [ ] metadata.json
  - [ ] scaler_params.json
- [ ] Add logging and progress bars
- [ ] Write integration tests

**Deliverables**:
- `Preprocessing/prepare_sequences.py` (500-600 lines)
- Integration tests

**Time Estimate**: 6-8 hours

---

### M1.3: Run Preprocessing (Day 12) - 0/4

- [ ] Run preprocessing on full dataset
  ```bash
  python3 Preprocessing/prepare_sequences.py
  ```
- [ ] Verify output files exist
- [ ] Check tensor shapes
  ```python
  data = torch.load('DATA/processed/train.pt')
  print(data['features'].shape)  # Should be [~680, 500, 11]
  ```
- [ ] Validate quality statistics in metadata.json

**Deliverables**:
- `DATA/processed/train.pt`
- `DATA/processed/val.pt`
- `DATA/processed/test.pt`
- `DATA/processed/metadata.json`
- `DATA/processed/scaler_params.json`

**Time Estimate**: 1-2 hours (mostly waiting for script)

---

### M1.4: Verification & EDA (Days 13-14) - 0/10

- [ ] Create `Preprocessing/verify_preprocessing.ipynb`
- [ ] Load preprocessed tensors
- [ ] Verify shapes
  - [ ] features: [N, 500, 11]
  - [ ] heart_rate: [N, 500, 1]
  - [ ] mask: [N, 500, 1]
- [ ] Check mask correctness
  - [ ] mask sum == sum of original lengths
  - [ ] Padded regions have mask=0
- [ ] Compute feature correlations
  - [ ] Speed → HR
  - [ ] Speed_lag_2 → HR (should be higher!)
  - [ ] Other features → HR
- [ ] Visualize feature distributions
- [ ] Check for data leakage (no user overlap)
- [ ] Compute padding statistics
- [ ] Generate preprocessing report
- [ ] Create summary visualizations

**Deliverables**:
- `Preprocessing/verify_preprocessing.ipynb`
- Feature correlation report
- Preprocessing summary document

**Time Estimate**: 3-4 hours

---

**Phase 1 Summary**: 0/34 tasks (0%)  
**Estimated Time**: ~20-25 hours over 7 days

---

## ⏳ Phase 2: Model Training (0/22)

**Goal**: Train LSTM with 11 features and masked loss  
**Duration**: Week 3 (7 days)  
**Priority**: 🟡 MEDIUM (after Phase 1)

### M2.1: Model Implementation (Days 15-16) - 0/8

- [ ] Create `Model/lstm.py`
- [ ] Implement `HeartRateLSTM_v2` class
  - [ ] __init__ (input_size=11, hidden_size=128, num_layers=2, dropout=0.3)
  - [ ] forward (accept features [B, 500, 11])
  - [ ] Return predictions [B, 500, 1]
- [ ] Create `Model/loss.py`
- [ ] Implement `MaskedMSELoss` class
  - [ ] __init__
  - [ ] forward (pred, target, mask)
  - [ ] Compute loss * mask, then sum / mask.sum()
- [ ] Test model forward pass
- [ ] Test loss computation
- [ ] Write unit tests

**Deliverables**:
- `Model/lstm.py` (100-150 lines)
- `Model/loss.py` (30-50 lines)
- Unit tests

**Time Estimate**: 3-4 hours

---

### M2.2: Training Pipeline (Days 17-18) - 0/10

- [ ] Create `Model/train.py`
- [ ] Implement dataset and dataloader
  - [ ] TensorDataset from preprocessed data
  - [ ] DataLoader with num_workers=4
- [ ] Implement training loop
  - [ ] Forward pass
  - [ ] Masked loss computation
  - [ ] Backward pass
  - [ ] Gradient clipping (1.0)
  - [ ] Optimizer step
- [ ] Implement validation loop
- [ ] Implement early stopping
  - [ ] Track best validation loss
  - [ ] Patience counter (10 epochs)
- [ ] Implement learning rate scheduling
  - [ ] ReduceLROnPlateau (factor=0.5, patience=5)
- [ ] Implement checkpoint saving
  - [ ] Save best model
  - [ ] Save optimizer state
  - [ ] Save training config
- [ ] Implement logging
  - [ ] TensorBoard or file logging
  - [ ] Log train/val loss, MAE
- [ ] Add command-line arguments
- [ ] Add progress bars (tqdm)

**Deliverables**:
- `Model/train.py` (400-500 lines)

**Time Estimate**: 6-8 hours

---

### M2.3: Initial Training Run (Day 19) - 0/4

- [ ] Train with default hyperparameters
  ```bash
  python3 Model/train.py --model lstm --epochs 100 --batch_size 16
  ```
- [ ] Monitor training curves
- [ ] Check validation MAE
- [ ] Evaluate on validation set

**Expected Results**:
- Training converges in <50 epochs
- Validation MAE < 13 BPM (at least as good as V1)

**Deliverables**:
- `checkpoints/lstm_v2_baseline.pt`
- `results/training_curves.png`
- Validation metrics

**Time Estimate**: 2-3 hours (mostly waiting)

---

### M2.4: Hyperparameter Tuning (Days 20-21) - 0/6

- [ ] **Experiment 1**: hidden_size
  - [ ] Train with hidden_size=64
  - [ ] Train with hidden_size=128 (baseline)
  - [ ] Train with hidden_size=256
- [ ] **Experiment 2**: dropout
  - [ ] Train with dropout=0.2
  - [ ] Train with dropout=0.3 (baseline)
  - [ ] Train with dropout=0.4
- [ ] **Experiment 3**: learning_rate
  - [ ] Train with lr=0.0001
  - [ ] Train with lr=0.0005 (baseline)
  - [ ] Train with lr=0.001
- [ ] Compare all results
- [ ] Select best configuration
- [ ] Re-train with best config

**Deliverables**:
- Hyperparameter search results table
- `checkpoints/best_model.pt`

**Time Estimate**: 8-10 hours (mostly waiting)

---

**Phase 2 Summary**: 0/28 tasks (0%)  
**Estimated Time**: ~25-30 hours over 7 days

---

## ⏳ Phase 3: Evaluation & Analysis (0/18)

**Goal**: Comprehensive evaluation and comparison with V1  
**Duration**: Week 4 (7 days)  
**Priority**: 🟢 LOW (after Phase 2)

### M3.1: Test Set Evaluation (Day 22) - 0/5

- [ ] Create `Model/evaluate.py`
- [ ] Load best checkpoint
- [ ] Run evaluation on test set
- [ ] Compute metrics
  - [ ] MAE (primary metric)
  - [ ] RMSE
  - [ ] R²
  - [ ] Per-sample MAE
- [ ] Generate visualizations
  - [ ] Prediction vs actual (sample workouts)
  - [ ] Error distribution
  - [ ] MAE histogram

**Deliverables**:
- `Model/evaluate.py` (200-300 lines)
- `results/test_metrics.json`
- Visualization plots

**Time Estimate**: 3-4 hours

---

### M3.2: Ablation Study (Days 23-24) - 0/7

- [ ] **Baseline**: Train with 3 base features only
- [ ] **+Lag**: Train with base + lag features (6 total)
- [ ] **+Derivatives**: Train with base + lag + derivatives (8 total)
- [ ] **+Rolling**: Train with base + lag + derivatives + rolling (10 total)
- [ ] **All**: Train with all 11 features
- [ ] Compare results in table
- [ ] Analyze feature importance

**Deliverables**:
- Ablation study results table
- Feature importance analysis

**Time Estimate**: 6-8 hours (mostly waiting)

---

### M3.3: V1 vs V2 Comparison (Day 25) - 0/4

- [ ] Load V1 baseline results from `/SUB_3H_42KM_DL/Model/FINAL_RESULTS.md`
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

**Time Estimate**: 2-3 hours

---

### M3.4: Final Report & Documentation (Days 26-28) - 0/8

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
- [ ] Add final comments to code
- [ ] Create reproducibility instructions

**Deliverables**:
- `results/FINAL_REPORT_V2.md`
- Updated documentation
- Clean, well-commented code

**Time Estimate**: 6-8 hours

---

**Phase 3 Summary**: 0/24 tasks (0%)  
**Estimated Time**: ~20-25 hours over 7 days

---

## 📊 Progress Summary

### Overall Project Status

| Phase | Tasks Complete | Total Tasks | Progress | Status |
|-------|---------------|-------------|----------|--------|
| **Documentation** | 11 | 11 | 100% | ✅ Complete |
| **Phase 0: Data Quality** | 0 | 28 | 0% | ⏳ Not Started |
| **Phase 1: Preprocessing** | 0 | 34 | 0% | ⏳ Not Started |
| **Phase 2: Training** | 0 | 28 | 0% | ⏳ Not Started |
| **Phase 3: Evaluation** | 0 | 24 | 0% | ⏳ Not Started |
| **TOTAL** | **11** | **125** | **9%** | 🔄 In Progress |

### Time Estimates

| Phase | Estimated Hours | Status |
|-------|----------------|--------|
| Documentation | 8 hours | ✅ Complete |
| Phase 0 | 15-20 hours | ⏳ Pending |
| Phase 1 | 20-25 hours | ⏳ Pending |
| Phase 2 | 25-30 hours | ⏳ Pending |
| Phase 3 | 20-25 hours | ⏳ Pending |
| **TOTAL** | **88-108 hours** (~2-3 weeks) | - |

---

## 🎯 Current Priority Tasks

### This Week (Week 1: Data Quality)

1. 🔴 **HIGH**: Setup annotation environment (Day 1)
2. 🔴 **HIGH**: Annotate 100 workouts (Days 2-5)
3. 🟡 **MEDIUM**: Analyze annotations (Day 6)
4. 🟡 **MEDIUM**: Implement quality filters (Day 7)

### Next Week (Week 2: Preprocessing)

5. 🟡 **MEDIUM**: Implement feature engineering
6. 🟡 **MEDIUM**: Implement preprocessing pipeline
7. 🟢 **LOW**: Run preprocessing
8. 🟢 **LOW**: Verify results

---

## 🚀 Quick Commands

```bash
# Navigate to project
cd /home/riccardo/Documents/Collaborative-Projects/SUB3_V2

# Check current status
cat TODO.md

# View roadmap
cat ROADMAP.md

# Start Phase 0
cat docs/QUICK_START.md

# Update this TODO list
vim TODO.md
```

---

## 📝 Notes

### Completed Items Archive

**2025-01-10**:
- ✅ Created comprehensive documentation (11 files, 76 KB)
- ✅ BMAD methodology applied (PRD, Architecture, Data Quality)
- ✅ Project structure established
- ✅ Roadmap and TODO list created

### Blockers

- None currently

### Open Questions

- [ ] Which annotation tool? (Streamlit recommended)
- [ ] Quality threshold? (70/100 suggested, to be validated)
- [ ] Reduce sequence length to 400? (TBD based on Phase 1 results)

---

**Last Updated**: 2025-01-10  
**Next Review**: After Phase 0 completion  
**Owner**: Riccardo
