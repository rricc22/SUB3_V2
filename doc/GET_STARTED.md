# 🚀 Get Started with SUB3_V2

**Welcome to Version 2.0!**

This guide helps you continue with the heart rate prediction model implementation.

**Current Status**: Data processing complete ✅. Ready for tensor preparation (Phase 2).

---

## ⚡ Quick Status (2 minutes)

**What's Done** ✅:
- Documentation (BMAD docs, guides, READMEs)
- Data exploration infrastructure (indexing, visualization, gallery)
- 3-stage preprocessing pipeline (rule-based + LLM + corrections)
- Clean dataset with smoothing (2.3GB, ~94M lines)

**What's Next** 🔴:
- Feature engineering (11 features from speed/altitude)
- Tensor preparation (PyTorch format with masking)
- Model training (LSTM with masked loss)
- Evaluation (compare V1 vs V2)

---

## 📚 Documentation Quick Reference

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **README.md** | Project overview + baseline metrics | First |
| **ROADMAP.md** | Development plan (updated Jan 13) | For planning |
| **TODO.md** | Task checklist (updated Jan 13) | Daily tracking |
| **PROJECT_SUMMARY.md** | Complete status (updated Jan 13) | Weekly review |
| **GET_STARTED.md** | This file! | Right now |
| **AGENTS.md** | Coding guidelines + V2 specs | During implementation |
| **CLAUDE.md** | AI assistant context | Reference |
| **TODO_PHASE2-4.md** | Remaining tasks breakdown | Task planning |

---

## 🎯 Your Next Steps

### Right Now (5 min)

1. ✅ Read this file (you're doing it!)
2. [ ] Review current status in `PROJECT_SUMMARY.md`
3. [ ] Check Phase 2 tasks in `TODO.md` or `TODO_PHASE2-4.md`
4. [ ] Review baseline metrics in `README.md`

### Today (30 min)

5. [ ] Explore clean dataset:
   ```bash
   python3 Preprocessing/view_smoothed.py --workout-id <ID>
   ```
6. [ ] Review V2 feature specifications in `AGENTS.md`
7. [ ] Plan feature engineering implementation

### This Week (Phase 2: Tensor Preparation)

8. [ ] **Days 1-2**: Implement feature engineering module
9. [ ] **Days 3-4**: Implement sequence preparation pipeline
10. [ ] **Day 5**: Verify tensors and correlations

See `TODO.md` for detailed task breakdown.

---

## 🔑 Key Files to Implement

### Phase 2 (Current) - Tensor Preparation

```
Model/
├── feature_engineering.py       # Create 8 engineered features
└── prepare_sequences.py         # Main tensor preparation pipeline

DATA/processed/                  # Output directory
├── train.pt                     # Training tensors [N, 500, 11]
├── val.pt                       # Validation tensors
├── test.pt                      # Test tensors
├── metadata.json                # Dataset statistics
└── scaler_params.json           # Normalization parameters
```

### Phase 3 - Model Training

```
Model/
├── lstm.py                      # HeartRateLSTM_v2 (11 features)
├── loss.py                      # MaskedMSELoss
└── train.py                     # Training loop
```

### Phase 4 - Evaluation

```
Model/
└── evaluate.py                  # Test set evaluation

results/
├── test_metrics.json            # Final metrics
├── v1_vs_v2_comparison.md       # Comparison report
└── FINAL_REPORT_V2.md           # Complete results
```

---

## 📊 Success Criteria

Your V2 model should achieve:

| Metric | V1 Baseline | V2 Target | Current Status |
|--------|-------------|-----------|----------------|
| **MAE** | 13.88 BPM | < 10 BPM | Data ready, training pending |
| **R²** | 0.188 | > 0.35 | Data ready, training pending |
| **Correlation** | 0.25 | > 0.40 | Will measure after feature engineering |

---

## 💡 Pro Tips

### Current Phase (Phase 2)

- **Feature engineering**: Implement lag features first (biggest correlation impact)
- **Masking**: Essential for ignoring 43% padding pollution
- **Stratified splitting**: Balance fitness levels across train/val/test
- **Normalization**: Fit scaler on train only, keep HR unnormalized

### Code Quality

- **Follow AGENTS.md** for coding style (PEP 8)
- **Write docstrings** for all functions
- **Add unit tests** for critical functions
- **Use type hints** where helpful

### Progress Tracking

- **Update TODO.md daily** (check off completed tasks)
- **Update ROADMAP.md weekly** (update progress %)
- **Document blockers** in TODO.md Notes section

---

## 🆘 Need Help?

### Documentation

- **Technical questions**: See `AGENTS.md`
- **Project context**: See `CLAUDE.md`
- **Task details**: See `TODO.md` or `TODO_PHASE2-4.md`

### Reference Implementation

V1 project has working implementations you can reference:

```
/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/
├── Preprocessing/prepare_sequences.py    # V1 preprocessing (3 features)
├── Model/LSTM.py                         # V1 LSTM model
├── Model/train.py                        # V1 training loop
└── finetune/train_stage1.py              # Masking example
```

---

## 🎉 Ready to Continue!

You now have:
- ✅ Complete documentation
- ✅ Data exploration infrastructure  
- ✅ 3-stage preprocessing pipeline
- ✅ Clean, smoothed dataset (2.3GB)
- 📋 Clear roadmap for Phases 2-4

**Ready for Phase 2? Start with feature engineering!**

```bash
# Start Phase 2: Tensor Preparation
cat TODO.md | grep -A 20 "Phase 2"

# Or check detailed breakdown
cat doc/TODO_PHASE2-4.md
```

Good luck! 🚀

---

**Questions?** Review the documentation or refer to V1 implementation.  
**Project Lead**: Riccardo  
**Technical Docs**: OpenCode  
**Last Updated**: 2026-01-13

