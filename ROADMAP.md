# SUB3_V2 Development Roadmap

**Project**: Heart Rate Prediction V2.0  
**Timeline**: 4 weeks (flexible)  
**Last Updated**: 2025-01-10

---

## 🎯 Project Vision

Achieve **MAE < 10 BPM** on base model (without finetuning) through data quality validation and feature engineering.

**Current V1 Baseline**: 13.88 BPM MAE  
**V2 Target**: < 10 BPM MAE (-28% improvement)

---

## 📅 Timeline Overview

```
Week 1: Data Quality ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 25%
Week 2: Preprocessing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50%
Week 3: Model Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75%
Week 4: Evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
```

---

## 🗺️ Phase 0: Data Quality Validation (Week 1)

**Goal**: Establish data quality baselines through manual annotation

**Duration**: 5-7 days  
**Dependencies**: None  
**Risk**: Medium (manual work can be time-consuming)

### Milestones

#### M0.1: Setup Annotation Environment (Day 1)
- [ ] Install dependencies (streamlit, plotly)
- [ ] Create annotation app (`EDA/quality_annotation_app.py`)
- [ ] Sample 100 workouts from raw data
- [ ] Test annotation workflow on 5 workouts

**Deliverables**:
- ✅ Working Streamlit app
- ✅ Sample dataset (100 workouts)

**Success Criteria**:
- App loads workouts and displays time-series plots
- Annotations save to CSV correctly

---

#### M0.2: Manual Annotation (Days 2-5)
- [ ] Annotate 20 workouts/day (4 days)
- [ ] Document common quality issues
- [ ] Take notes on edge cases

**Daily Progress**:
- Day 2: 20 workouts (20% complete)
- Day 3: 20 workouts (40% complete)
- Day 4: 20 workouts (60% complete)
- Day 5: 40 workouts (100% complete)

**Deliverables**:
- ✅ `DATA/quality_check/annotations.csv` (100 rows)
- ✅ Notes on quality issues

**Success Criteria**:
- 100 workouts annotated
- Quality score distribution documented

---

#### M0.3: Quality Analysis (Day 6)
- [ ] Analyze annotation statistics
- [ ] Establish quality thresholds
- [ ] Identify common failure modes
- [ ] Generate quality report

**Deliverables**:
- ✅ `DATA/quality_check/quality_report.md`
- ✅ Quality threshold recommendations

**Success Criteria**:
- Clear quality grade distribution (Excellent/Good/Poor)
- Documented thresholds (e.g., quality_score ≥ 70)

---

#### M0.4: Implement Automated Filters (Day 7)
- [ ] Implement `validate_hr_sensor()`
- [ ] Implement `validate_gps_speed()`
- [ ] Implement `validate_altitude()`
- [ ] Implement `validate_timestamps()`
- [ ] Test on annotated workouts

**Deliverables**:
- ✅ `Preprocessing/quality_filters.py`
- ✅ Unit tests for validation functions

**Success Criteria**:
- Automated filters match manual annotations >80% of time
- All functions have docstrings and tests

**Phase 0 Exit Criteria**:
- ✅ 100 workouts annotated
- ✅ Quality thresholds established
- ✅ Automated filters implemented
- ✅ Quality report generated

---

## 🗺️ Phase 1: Preprocessing Pipeline (Week 2)

**Goal**: Transform raw data into PyTorch tensors with engineered features

**Duration**: 5-7 days  
**Dependencies**: Phase 0 complete  
**Risk**: Low (well-specified)

### Milestones

#### M1.1: Feature Engineering Implementation (Days 8-9)
- [ ] Implement lag features (speed_lag_2, speed_lag_5, altitude_lag_30)
- [ ] Implement derivatives (speed_derivative, altitude_derivative)
- [ ] Implement rolling stats (rolling_speed_10, rolling_speed_30)
- [ ] Implement cumulative features (cumulative_elevation)
- [ ] Test on sample workout

**Deliverables**:
- ✅ `Preprocessing/feature_engineering.py`
- ✅ Unit tests for each feature

**Success Criteria**:
- All 8 features generated correctly
- Edge cases handled (first/last timesteps)
- Features validated on sample data

---

#### M1.2: Main Preprocessing Pipeline (Days 10-11)
- [ ] Implement workout loading with quality filtering
- [ ] Implement feature engineering integration
- [ ] Implement padding with mask generation
- [ ] Implement stratified user splitting
- [ ] Implement normalization (fit on train only)
- [ ] Implement tensor conversion

**Deliverables**:
- ✅ `Preprocessing/prepare_sequences.py`
- ✅ Integration tests

**Success Criteria**:
- Pipeline runs end-to-end without errors
- Output tensors have correct shapes
- Masking correctly identifies padded regions

---

#### M1.3: Run Preprocessing (Day 12)
- [ ] Run preprocessing on full dataset (974 workouts)
- [ ] Verify output tensor shapes
- [ ] Check data quality statistics
- [ ] Validate mask correctness

**Deliverables**:
- ✅ `DATA/processed/train.pt` (680 workouts)
- ✅ `DATA/processed/val.pt` (145 workouts)
- ✅ `DATA/processed/test.pt` (145 workouts)
- ✅ `DATA/processed/metadata.json`
- ✅ `DATA/processed/scaler_params.json`

**Success Criteria**:
- Tensor shapes: [N, 500, 11] for features
- Mask sum matches original lengths
- Train/val/test split is 70/15/15

---

#### M1.4: Verification & EDA (Days 13-14)
- [ ] Create verification notebook
- [ ] Visualize feature distributions
- [ ] Check feature correlations
- [ ] Validate mask coverage
- [ ] Generate preprocessing report

**Deliverables**:
- ✅ `Preprocessing/verify_preprocessing.ipynb`
- ✅ Feature correlation report
- ✅ Preprocessing summary

**Success Criteria**:
- Feature-HR correlation > 0.35 (vs 0.25 in V1)
- No data leakage between splits
- Padding ratio documented (~43%)

**Phase 1 Exit Criteria**:
- ✅ Preprocessed tensors generated
- ✅ Features validated (correlation improved)
- ✅ No data quality issues
- ✅ Verification notebook complete

---

## 🗺️ Phase 2: Model Training (Week 3)

**Goal**: Train LSTM with 11 features and masked loss

**Duration**: 5-7 days  
**Dependencies**: Phase 1 complete  
**Risk**: Medium (hyperparameter tuning may take time)

### Milestones

#### M2.1: Model Implementation (Days 15-16)
- [ ] Implement `HeartRateLSTM_v2` (11-feature input)
- [ ] Implement `MaskedMSELoss`
- [ ] Test model forward pass
- [ ] Test loss computation with masking

**Deliverables**:
- ✅ `Model/lstm.py`
- ✅ `Model/loss.py`
- ✅ Unit tests for model and loss

**Success Criteria**:
- Model accepts [N, 500, 11] input
- Loss correctly ignores padded regions
- Gradient flows correctly

---

#### M2.2: Training Pipeline (Days 17-18)
- [ ] Implement dataset and dataloader
- [ ] Implement training loop with masking
- [ ] Implement validation loop
- [ ] Implement early stopping
- [ ] Implement learning rate scheduling
- [ ] Implement checkpoint saving

**Deliverables**:
- ✅ `Model/train.py`
- ✅ Training configuration

**Success Criteria**:
- Training runs without errors
- Loss decreases over epochs
- Checkpoints save correctly

---

#### M2.3: Initial Training Run (Day 19)
- [ ] Train with default hyperparameters
- [ ] Monitor training curves
- [ ] Check for overfitting
- [ ] Evaluate on validation set

**Deliverables**:
- ✅ `checkpoints/lstm_v2_baseline.pt`
- ✅ Training curves plot
- ✅ Initial validation metrics

**Success Criteria**:
- Training converges (<50 epochs)
- Validation MAE < 13 BPM (at least as good as V1)
- No obvious overfitting

---

#### M2.4: Hyperparameter Tuning (Days 20-21)
- [ ] Tune hidden_size (64, 128, 256)
- [ ] Tune dropout (0.2, 0.3, 0.4)
- [ ] Tune learning_rate (0.0001, 0.0005, 0.001)
- [ ] Select best configuration

**Deliverables**:
- ✅ Hyperparameter search results
- ✅ Best model checkpoint

**Success Criteria**:
- Best configuration identified
- Validation MAE improvement over baseline

**Phase 2 Exit Criteria**:
- ✅ Model trained successfully
- ✅ Validation MAE < 12 BPM (improvement over V1)
- ✅ Best checkpoint saved
- ✅ Training curves look healthy

---

## 🗺️ Phase 3: Evaluation & Analysis (Week 4)

**Goal**: Comprehensive evaluation and comparison with V1

**Duration**: 5-7 days  
**Dependencies**: Phase 2 complete  
**Risk**: Low (evaluation is straightforward)

### Milestones

#### M3.1: Test Set Evaluation (Day 22)
- [ ] Load best checkpoint
- [ ] Run evaluation on test set
- [ ] Compute all metrics (MAE, RMSE, R²)
- [ ] Generate prediction visualizations

**Deliverables**:
- ✅ `results/test_metrics.json`
- ✅ Test set predictions
- ✅ Visualization plots

**Success Criteria**:
- Test MAE < 10 BPM (target achieved!)
- R² > 0.35
- Visual predictions follow trends

---

#### M3.2: Ablation Study (Days 23-24)
- [ ] Train with base features only (3)
- [ ] Train with +lag features (6)
- [ ] Train with +derivatives (8)
- [ ] Train with +rolling stats (10)
- [ ] Train with all features (11)
- [ ] Compare results

**Deliverables**:
- ✅ Ablation study results table
- ✅ Feature importance analysis

**Success Criteria**:
- Each feature group contributes positively
- Lag features show largest improvement

---

#### M3.3: V1 vs V2 Comparison (Day 25)
- [ ] Load V1 baseline results
- [ ] Compare metrics side-by-side
- [ ] Analyze improvement sources
- [ ] Generate comparison report

**Deliverables**:
- ✅ `results/v1_vs_v2_comparison.md`
- ✅ Comparison visualization

**Success Criteria**:
- V2 outperforms V1 on all metrics
- Improvements quantified and explained

---

#### M3.4: Final Report & Documentation (Days 26-28)
- [ ] Write final results report
- [ ] Update README with V2 results
- [ ] Update CHANGELOG
- [ ] Generate presentation slides
- [ ] Clean up code and comments

**Deliverables**:
- ✅ `results/FINAL_REPORT_V2.md`
- ✅ Updated documentation
- ✅ Presentation slides

**Success Criteria**:
- Complete documentation of results
- Clear comparison with V1
- Reproducible instructions

**Phase 3 Exit Criteria**:
- ✅ Test MAE < 10 BPM (or documented if not achieved)
- ✅ Complete ablation study
- ✅ V1 vs V2 comparison complete
- ✅ Final report published

---

## 🎉 Project Completion Criteria

### Primary Goals

- [x] **Documentation**: Comprehensive BMAD documentation ✅
- [ ] **Data Quality**: 100 workouts annotated
- [ ] **Preprocessing**: Feature engineering implemented
- [ ] **Training**: Model trained with masking
- [ ] **Evaluation**: Test MAE < 10 BPM

### Success Metrics

| Metric | V1 Baseline | V2 Target | Status |
|--------|-------------|-----------|--------|
| **MAE (base)** | 13.88 BPM | < 10 BPM | ⏳ Pending |
| **R²** | 0.188 | > 0.35 | ⏳ Pending |
| **Correlation** | 0.25 | > 0.40 | ⏳ Pending |

### Deliverables Checklist

**Documentation** (11 files):
- [x] README.md
- [x] AGENTS.md
- [x] ROADMAP.md (this file)
- [x] TODO.md
- [x] PROJECT_SUMMARY.md
- [x] docs/PRD.md
- [x] docs/ARCHITECTURE.md
- [x] docs/DATA_QUALITY.md
- [x] docs/CHANGELOG.md
- [x] docs/QUICK_START.md
- [x] Module READMEs (DATA, Preprocessing, Model)

**Code** (to be implemented):
- [ ] EDA/quality_annotation_app.py
- [ ] Preprocessing/quality_filters.py
- [ ] Preprocessing/feature_engineering.py
- [ ] Preprocessing/prepare_sequences.py
- [ ] Model/lstm.py
- [ ] Model/loss.py
- [ ] Model/train.py
- [ ] Model/evaluate.py

**Data**:
- [ ] DATA/quality_check/annotations.csv
- [ ] DATA/processed/train.pt
- [ ] DATA/processed/val.pt
- [ ] DATA/processed/test.pt

**Results**:
- [ ] checkpoints/best_model.pt
- [ ] results/test_metrics.json
- [ ] results/FINAL_REPORT_V2.md

---

## 🚀 Quick Start

### Current Phase: Phase 0 (Data Quality)

```bash
cd /home/riccardo/Documents/Collaborative-Projects/SUB3_V2

# 1. Review roadmap
cat ROADMAP.md

# 2. Check todo list
cat TODO.md

# 3. Start Phase 0
# See docs/QUICK_START.md for detailed instructions
```

---

## 📊 Progress Tracking

### Overall Progress: 25% (Documentation Complete)

```
[████████░░░░░░░░░░░░░░░░░░░░] 25%

✅ Phase -1: Documentation (100%)
⏳ Phase  0: Data Quality (0%)
⏳ Phase  1: Preprocessing (0%)
⏳ Phase  2: Model Training (0%)
⏳ Phase  3: Evaluation (0%)
```

### Phase Status

| Phase | Status | Start | End | Duration | Progress |
|-------|--------|-------|-----|----------|----------|
| **Documentation** | ✅ Complete | Jan 10 | Jan 10 | 1 day | 100% |
| **Phase 0: Data Quality** | ⏳ Not Started | - | - | 7 days | 0% |
| **Phase 1: Preprocessing** | ⏳ Not Started | - | - | 7 days | 0% |
| **Phase 2: Training** | ⏳ Not Started | - | - | 7 days | 0% |
| **Phase 3: Evaluation** | ⏳ Not Started | - | - | 7 days | 0% |

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
- Documentation phase completed

---

**Next Update**: After Phase 0 completion  
**Owner**: Riccardo  
**Contributors**: OpenCode (documentation), Riccardo (implementation)
