# Phase 2-4 Tasks (Continuation)

## 🔴 Phase 2: Tensor Preparation & Feature Engineering (IN PROGRESS)

### M2.1: Feature Engineering - 0/5 tasks
- [ ] Create `Model/feature_engineering.py`
- [ ] Implement lag features (speed_lag_2, speed_lag_5, altitude_lag_30)
- [ ] Implement derivatives (speed_derivative, altitude_derivative)
- [ ] Implement rolling stats (rolling_speed_10, rolling_speed_30)
- [ ] Implement cumulative_elevation_gain + write tests

### M2.2: Sequence Preparation - 0/6 tasks
- [ ] Create `Model/prepare_sequences.py`
- [ ] Load clean_dataset_smoothed.json (streaming)
- [ ] Apply feature engineering
- [ ] Pad/truncate to 500 timesteps with masking
- [ ] Stratified user splitting (70/15/15)
- [ ] Save train.pt, val.pt, test.pt

### M2.3: Verification - 0/1 task
- [ ] Create verify_tensors.ipynb (check shapes, correlations, mask)

**Phase 2 Total**: 0/12 tasks

---

## ⏳ Phase 3: Model Training

### M3.1: Model Implementation - 0/4 tasks
- [ ] Create Model/lstm.py (HeartRateLSTM_v2)
- [ ] Create Model/loss.py (MaskedMSELoss)
- [ ] Test forward pass
- [ ] Write unit tests

### M3.2: Training Pipeline - 0/6 tasks
- [ ] Create Model/train.py
- [ ] Implement training loop with masking
- [ ] Implement validation loop
- [ ] Early stopping + LR scheduling
- [ ] Checkpoint saving
- [ ] Logging (TensorBoard or file)

### M3.3: Initial Training - 0/3 tasks
- [ ] Train with default hyperparameters
- [ ] Monitor training curves
- [ ] Evaluate on validation set

### M3.4: Hyperparameter Tuning - 0/3 tasks
- [ ] Tune hidden_size (64, 128, 256)
- [ ] Tune dropout (0.2, 0.3, 0.4)
- [ ] Select best configuration

**Phase 3 Total**: 0/16 tasks

---

## ⏳ Phase 4: Evaluation

### M4.1: Test Evaluation - 0/4 tasks
- [ ] Create Model/evaluate.py
- [ ] Run on test set
- [ ] Compute MAE, RMSE, R²
- [ ] Generate visualizations

### M4.2: Ablation Study - 0/5 tasks
- [ ] Train with 3 base features
- [ ] Train with +lag (6 features)
- [ ] Train with +derivatives (8 features)
- [ ] Train with +rolling (10 features)
- [ ] Compare results

### M4.3: V1 vs V2 Comparison - 0/3 tasks
- [ ] Load V1 results
- [ ] Create comparison table
- [ ] Analyze improvement sources

### M4.4: Final Report - 0/3 tasks
- [ ] Write FINAL_REPORT_V2.md
- [ ] Update all documentation
- [ ] Code cleanup

**Phase 4 Total**: 0/15 tasks

---

**Grand Total**: Phases 0-1 complete, 43 tasks remaining (Phases 2-4)
