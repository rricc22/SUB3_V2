# Preprocessing Pipeline - MVP Design Document

## Overview

Hybrid preprocessing pipeline with two main goals:

1. **Quality Control**: Detect & fix anomalies (especially HR offset errors)
2. **Workout Classification**: Categorize each workout by training type

**Scope**: All 61K running workouts (including GPS-computed speed)

**Key Insight**: Normal running HR should be **150-190 BPM**. If data shows 140-160 BPM range, suspect **offset error** (~20-30 BPM too low).

## Architecture

```
Raw Data (61K) 
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: RULE-BASED FILTERS (fast, free)                        │
│   • Hard filters (auto-exclude garbage data)                    │
│   • Soft flags (suspicious patterns → send to LLM)              │
│   • Offset detection heuristics                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓                              ↓
   PASS (~85%)                 FLAGGED (~15%)
      ↓                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: LLM VALIDATION (smart, ~$0.02/workout)                 │
│   • Analyze HR plot + stats                                     │
│   • Detect offset errors & suggest correction                   │
│   • Classify workout type                                       │
│   • Decision: KEEP / FIX (with params) / EXCLUDE                │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: WORKOUT CLASSIFICATION                                 │
│   • Intensive (threshold/tempo): HR 170-190+, steady high       │
│   • Intervals (alternating): HR oscillates 150-180 ↔ 130-150    │
│   • Steady (easy/moderate): HR 140-165, consistent              │
│   • Recovery (easy jog): HR 120-145, low effort                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
   Clean Data + Labels
```

---

## Workout Classification Categories

The LLM will classify each valid workout into one of **4 categories** based on HR patterns:

| Category | HR Pattern | Speed Pattern | Typical Duration | Description |
|----------|-----------|---------------|------------------|-------------|
| **INTENSIVE** | 170-190+ BPM sustained | Fast (12-16 km/h) | 20-60 min | Threshold, tempo, race pace |
| **INTERVALS** | Oscillates: 155-185 ↔ 130-155 | Varies with HR | 30-60 min | Work/rest cycles, fartlek |
| **STEADY** | 145-170 BPM consistent | Moderate (9-12 km/h) | 30-90 min | Easy-moderate continuous run |
| **RECOVERY** | 120-150 BPM low effort | Slow (7-10 km/h) | 20-45 min | Easy jog, active recovery |

### Classification Rules (for LLM guidance)

```
IF max_hr > 180 AND mean_hr > 170 AND hr_std < 10:
    → INTENSIVE (steady high effort)

IF hr_oscillation_detected AND oscillation_amplitude > 25 BPM:
    → INTERVALS (work/rest pattern)

IF 145 < mean_hr < 170 AND hr_std < 15:
    → STEADY (consistent moderate effort)

IF mean_hr < 150 AND max_hr < 165:
    → RECOVERY (low effort)
```

### Why Classification Matters

1. **Model training**: Different workout types may need different model weights
2. **Offset detection**: Expected HR range depends on workout type
3. **Data balancing**: Ensure train/val/test have similar workout type distributions
4. **Feature engineering**: Workout type could be an input feature

---

## Anomaly Catalog

### Category 1: Heart Rate Anomalies

| ID | Anomaly | Description | Detection Rule | Severity | Action |
|----|---------|-------------|----------------|----------|--------|
| HR01 | **HR_ZERO** | Zero values in HR data | `hr == 0` | HIGH | EXCLUDE or interpolate |
| HR02 | **HR_TOO_LOW** | Physiologically impossible low HR | `hr < 30 BPM` | HIGH | EXCLUDE |
| HR03 | **HR_TOO_HIGH** | Physiologically impossible high HR | `hr > 220 BPM` | HIGH | EXCLUDE or cap |
| HR04 | **HR_SPIKE** | Sudden unrealistic HR jump | `abs(diff(hr)) > 40 BPM` per timestep | MEDIUM | Flag for review |
| HR05 | **HR_FLATLINE** | Sensor dropout / stuck value | `>20 identical consecutive values` | MEDIUM | Exclude segment or interpolate |
| HR06 | **HR_LOW_VARIANCE** | Suspicious lack of variation | `std(hr) < 5 BPM` over workout | MEDIUM | Flag for review |
| HR07 | **HR_SUDDEN_DROP** | Possible offset issue (like workout 460652128) | `>50 BPM drop in <1 min` | HIGH | **LLM review for offset correction** |
| HR08 | **HR_SUDDEN_RISE** | Possible sensor reconnect or error | `>50 BPM rise in <1 min` (not at start) | MEDIUM | Flag for review |
| HR09 | **HR_OUT_OF_CONTEXT** | HR doesn't match expected effort | `HR < 100 during high speed running` | MEDIUM | **LLM review** |
| HR10 | **HR_NEGATIVE_CORRELATION** | HR decreases when speed increases | `corr(speed, hr) < -0.2` | MEDIUM | Flag for review |

### Category 2: Speed Anomalies

| ID | Anomaly | Description | Detection Rule | Severity | Action |
|----|---------|-------------|----------------|----------|--------|
| SP01 | **SPEED_TOO_HIGH** | Impossible running speed | `speed > 25 km/h sustained (>30s)` | HIGH | EXCLUDE or cap |
| SP02 | **SPEED_NEGATIVE** | Data corruption | `speed < 0` | HIGH | EXCLUDE |
| SP03 | **SPEED_ZERO_MOVING** | GPS dropout while HR shows effort | `speed == 0 AND hr > 140` for >1min | MEDIUM | Interpolate or flag |
| SP04 | **SPEED_SPIKE** | GPS glitch | `speed change > 15 km/h` per timestep | MEDIUM | Smooth or flag |
| SP05 | **SPEED_HR_MISMATCH** | Speed and HR don't correlate at all | `abs(correlation) < 0.05` over workout | LOW | Flag for review |

### Category 3: Timestamp Anomalies

| ID | Anomaly | Description | Detection Rule | Severity | Action |
|----|---------|-------------|----------------|----------|--------|
| TS01 | **TS_GAP_LARGE** | Large gap in recording | `gap > 300 seconds (5 min)` | HIGH | Split into segments or exclude |
| TS02 | **TS_GAP_MEDIUM** | Medium gap | `gap > 60 seconds (1 min)` | LOW | Interpolate or note |
| TS03 | **TS_NEGATIVE** | Time going backwards | `diff(ts) < 0` | HIGH | EXCLUDE |
| TS04 | **TS_DUPLICATE** | Duplicate timestamps | `diff(ts) == 0` | MEDIUM | Remove duplicates |
| TS05 | **TS_IRREGULAR** | Highly variable sampling rate | `std(diff(ts)) > 30s` | LOW | Flag for review |

### Category 4: Altitude Anomalies

| ID | Anomaly | Description | Detection Rule | Severity | Action |
|----|---------|-------------|----------------|----------|--------|
| AL01 | **ALT_SPIKE** | GPS altitude glitch | `abs(diff(alt)) > 50m` per timestep | MEDIUM | Smooth or interpolate |
| AL02 | **ALT_UNREALISTIC** | Impossible altitude | `alt < -500m OR alt > 6000m` | HIGH | Cap or exclude |
| AL03 | **ALT_FLAT** | No altitude variation (suspicious) | `std(alt) < 1m` over long workout | LOW | Flag (might be valid flat course) |

### Category 5: Data Integrity

| ID | Anomaly | Description | Detection Rule | Severity | Action |
|----|---------|-------------|----------------|----------|--------|
| DI01 | **LENGTH_MISMATCH** | Arrays have different lengths | `len(hr) != len(ts) != len(speed)` | HIGH | EXCLUDE |
| DI02 | **TOO_SHORT** | Not enough data for training | `duration < 5 min OR points < 50` | HIGH | EXCLUDE |
| DI03 | **TOO_MUCH_PADDING** | Excessive padding in sequence | `padding > 50%` of max_length | MEDIUM | Trim or exclude |
| DI04 | **MISSING_HR** | No heart rate data | `hr array empty or all NaN` | HIGH | EXCLUDE |
| DI05 | **MISSING_SPEED** | No speed data (if required) | `speed array empty` | MEDIUM | Exclude or use HR-only |
| DI06 | **PARSE_ERROR** | Cannot read workout data | Exception during parsing | HIGH | EXCLUDE |

### Category 6: Physiological Plausibility (LLM-only)

| ID | Anomaly | Description | Why LLM needed |
|----|---------|-------------|----------------|
| PH01 | **OFFSET_ERROR** | HR shifted by constant (like +60/-60 BPM) | Requires reasoning about expected HR zones for given effort |
| PH02 | **SCALE_ERROR** | HR multiplied/divided by factor | Requires pattern recognition across workout phases |
| PH03 | **SENSOR_SWAP** | Different sensor characteristics mid-workout | Requires understanding of sensor behavior patterns |
| PH04 | **WRONG_SPORT** | Data doesn't match "running" profile | HR/speed pattern suggests cycling, walking, or other activity |
| PH05 | **MIXED_WORKOUT** | Multiple activities merged | Requires segmentation and classification |

---

## PRIORITY: Offset Error Detection

**This is the most common problem in the dataset.**

### The Problem
- Expected running HR: **150-190 BPM**
- Observed in many workouts: **120-160 BPM** (suspiciously low)
- Likely cause: Sensor offset of **-20 to -40 BPM**

### Types of Offset Errors

| Type | Pattern | Example | Detection |
|------|---------|---------|-----------|
| **Full workout offset** | Entire workout shifted | All HR 20-30 BPM too low | Mean HR < 145 during hard effort |
| **Partial offset** | Offset starts mid-workout | Like workout 460652128 | Sudden HR drop (>50 BPM) |
| **Variable offset** | Offset changes over time | Drift in sensor calibration | HR doesn't match effort pattern |

### Offset Detection Heuristics (Rule-based)

```python
# Flag 1: Suspiciously low HR for running effort
if mean_hr < 145 and mean_speed > 10:  # km/h
    flag("POSSIBLE_OFFSET: Low HR for running speed")

# Flag 2: HR range too low for any real workout
if max_hr < 160 and duration > 20:  # minutes
    flag("POSSIBLE_OFFSET: Max HR never reaches effort zone")

# Flag 3: Sudden transition (partial offset)
if any(hr_drop > 50 in < 1 min) and not at_start:
    flag("POSSIBLE_OFFSET: Sudden drop mid-workout")

# Flag 4: HR-Speed correlation but wrong range
if correlation(hr, speed) > 0.3 and mean_hr < 140:
    flag("POSSIBLE_OFFSET: Good correlation but wrong HR range")
```

### Offset Correction Strategy

1. **Detect**: Rule-based flags identify suspicious workouts
2. **Analyze**: LLM reviews plot and determines:
   - Is this really an offset error?
   - What is the likely offset value?
   - Where does the offset start? (full workout or partial)
3. **Correct**: Apply offset: `hr_corrected = hr + offset_value`
4. **Validate**: Check corrected HR is in plausible range (150-190 for effort)

---

## Detection Priority

### Stage 1: Rule-Based HARD FILTERS (Auto-exclude)
These are definite errors - no LLM needed:
- HR_ZERO, HR_TOO_LOW, HR_TOO_HIGH (HR01-03)
- SPEED_NEGATIVE (SP02)
- TS_NEGATIVE (TS03)
- LENGTH_MISMATCH, MISSING_HR, PARSE_ERROR (DI01, DI04, DI06)
- TOO_SHORT (DI02)

### Stage 2: Rule-Based SOFT FLAGS (Need review)
These might be errors OR might be valid edge cases:
- HR_SPIKE, HR_FLATLINE (HR04-05)
- HR_SUDDEN_DROP, HR_SUDDEN_RISE (HR07-08) → **High priority for LLM**
- SPEED_TOO_HIGH, SPEED_SPIKE (SP01, SP04)
- TS_GAP_LARGE (TS01)
- ALT_SPIKE (AL01)

### Stage 3: LLM Validation
For all flagged samples:
1. Generate visualization (HR + Speed + Altitude plot)
2. Send to LLM with context
3. Get structured decision: `{action: KEEP|FIX|EXCLUDE, reason: str, fix_params: {}}`
4. Apply fix if applicable

---

## Confirmed Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| HR minimum valid | 30 BPM | Below this = sensor error |
| HR maximum valid | 220 BPM | Above this = sensor error |
| HR spike threshold | 40 BPM/timestep | Flag for review |
| HR flatline length | 20 consecutive | Sensor dropout |
| HR sudden drop | 50 BPM in <1 min | Possible offset issue |
| Speed max running | 25 km/h | Above = GPS error or cycling |
| Min workout duration | 5 min | Too short for training |
| Min data points | 50 | Too few samples |
| Large timestamp gap | 5 min | Split or exclude |
| **Expected HR range** | **150-190 BPM** | For normal running effort |
| **Offset suspect threshold** | **mean_hr < 145** | When speed > 10 km/h |

---

## Remaining Questions

### Q1: Workout Classification Thresholds
Are these HR ranges correct for your training style?

| Category | Proposed HR Range | Your typical range? |
|----------|------------------|---------------------|
| INTENSIVE | 170-190+ BPM | ? |
| INTERVALS | 155-185 ↔ 130-155 BPM | ? |
| STEADY | 145-170 BPM | ? |
| RECOVERY | 120-150 BPM | ? |

### Q2: LLM Provider
- Claude 3.5 Sonnet (recommended - good vision, reasonable cost)?
- GPT-4 Vision?
- Other?

### Q3: Output Format
- Keep as JSON lines?
- Convert to PyTorch tensors?
- Both?

---

## Case Study Reference

### Workout 460652128
- **Problem**: HR showed 90-120 BPM oscillation during interval training
- **Root cause**: -60 BPM offset after minute 12
- **Detection**: HR_SUDDEN_DROP rule flagged it
- **Diagnosis**: LLM reasoning identified offset issue
- **Fix**: Apply +60 BPM after transition point
- **Result**: Corrected HR shows realistic 150-175 BPM intervals

---

## Next Steps

1. ✅ Define anomaly catalog (done)
2. ✅ Define workout categories (done)
3. ✅ Document offset detection strategy (done)
4. **YOU**: Answer remaining questions (classification thresholds, LLM provider)
5. **NEXT**: Implement Stage 1 rule-based filters
6. **NEXT**: Design LLM prompt template
7. **TEST**: Run on 100 sample workouts
8. **SCALE**: Process full 61K dataset

