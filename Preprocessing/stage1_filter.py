#!/usr/bin/env python3
"""
Stage 1: Rule-Based Filtering for Workout Data Quality Control

This script applies rule-based filters to detect anomalies in workout data.
It outputs a JSON report with decisions: PASS, FLAG (for LLM review), or EXCLUDE.

Author: OpenCode
Date: 2025-01-11
"""

import ast
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import argparse
from datetime import datetime
import linecache


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Thresholds and parameters for anomaly detection."""
    
    # Heart Rate thresholds
    HR_MIN_VALID = 30           # Below = sensor error
    HR_MAX_VALID = 220          # Above = sensor error
    HR_SPIKE_THRESHOLD = 40     # BPM change per timestep
    HR_FLATLINE_LENGTH = 20     # Consecutive identical values
    HR_SUDDEN_DROP = 50         # BPM drop in short time
    HR_SUDDEN_RISE = 50         # BPM rise in short time (not at start)
    HR_LOW_VARIANCE = 5         # Std below this is suspicious
    HR_EXPECTED_MIN = 145       # Expected min for running effort
    HR_EXPECTED_MAX = 190       # Expected max for running effort
    
    # Speed thresholds (km/h)
    SPEED_MAX_RUNNING = 25      # Above = GPS error or not running
    SPEED_SPIKE_THRESHOLD = 15  # km/h change per timestep
    
    # Timestamp thresholds (seconds)
    TS_GAP_LARGE = 300          # 5 min gap
    TS_GAP_MEDIUM = 60          # 1 min gap
    
    # Altitude thresholds (meters)
    ALT_SPIKE_THRESHOLD = 50    # meters per timestep
    ALT_MIN_VALID = -500
    ALT_MAX_VALID = 6000
    
    # Data integrity
    MIN_DURATION_MINUTES = 5
    MIN_DATA_POINTS = 50
    
    # Offset detection
    OFFSET_SUSPECT_HR = 145     # Mean HR below this during effort = suspect
    OFFSET_SUSPECT_SPEED = 10   # km/h - if running this fast, HR should be higher


# =============================================================================
# Data Classes
# =============================================================================

class Decision(Enum):
    PASS = "PASS"           # Good quality, keep as-is
    FLAG = "FLAG"           # Needs LLM review
    EXCLUDE = "EXCLUDE"     # Definitely bad, exclude


class Severity(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    code: str
    description: str
    severity: str
    details: Dict = field(default_factory=dict)


@dataclass
class WorkoutAnalysis:
    """Complete analysis result for a workout."""
    line_number: int
    workout_id: Optional[int]
    user_id: Optional[int]
    decision: str
    anomalies: List[Dict]
    stats: Dict
    flags: List[str]
    suggested_action: Optional[str] = None
    
    def to_dict(self):
        return {
            "line_number": self.line_number,
            "workout_id": self.workout_id,
            "user_id": self.user_id,
            "decision": self.decision,
            "anomalies": self.anomalies,
            "stats": self.stats,
            "flags": self.flags,
            "suggested_action": self.suggested_action
        }


# =============================================================================
# Anomaly Detection Functions
# =============================================================================

def detect_hr_anomalies(hr: np.ndarray, timestamps: np.ndarray) -> List[Anomaly]:
    """Detect heart rate anomalies."""
    anomalies = []
    
    # HR01: Zero values
    zero_count = np.sum(hr == 0)
    if zero_count > 0:
        anomalies.append(Anomaly(
            code="HR01_ZERO",
            description="Zero HR values detected",
            severity=Severity.HIGH.value,
            details={"count": int(zero_count), "percentage": float(zero_count / len(hr) * 100)}
        ))
    
    # HR02: Too low
    too_low = np.sum(hr < Config.HR_MIN_VALID)
    if too_low > 0:
        anomalies.append(Anomaly(
            code="HR02_TOO_LOW",
            description=f"HR below {Config.HR_MIN_VALID} BPM",
            severity=Severity.HIGH.value,
            details={"count": int(too_low), "min_value": int(np.min(hr))}
        ))
    
    # HR03: Too high
    too_high = np.sum(hr > Config.HR_MAX_VALID)
    if too_high > 0:
        anomalies.append(Anomaly(
            code="HR03_TOO_HIGH",
            description=f"HR above {Config.HR_MAX_VALID} BPM",
            severity=Severity.HIGH.value,
            details={"count": int(too_high), "max_value": int(np.max(hr))}
        ))
    
    # HR04: Spikes
    hr_diff = np.abs(np.diff(hr))
    spike_count = np.sum(hr_diff > Config.HR_SPIKE_THRESHOLD)
    if spike_count > 0:
        max_spike = int(np.max(hr_diff))
        anomalies.append(Anomaly(
            code="HR04_SPIKE",
            description=f"HR spikes >{Config.HR_SPIKE_THRESHOLD} BPM/timestep",
            severity=Severity.MEDIUM.value,
            details={"count": int(spike_count), "max_spike": max_spike}
        ))
    
    # HR05: Flatline
    max_flatline = find_max_consecutive(hr)
    if max_flatline > Config.HR_FLATLINE_LENGTH:
        anomalies.append(Anomaly(
            code="HR05_FLATLINE",
            description=f"HR flatline >{Config.HR_FLATLINE_LENGTH} consecutive values",
            severity=Severity.MEDIUM.value,
            details={"max_consecutive": int(max_flatline)}
        ))
    
    # HR06: Low variance
    hr_std = np.std(hr)
    if hr_std < Config.HR_LOW_VARIANCE and len(hr) > 50:
        anomalies.append(Anomaly(
            code="HR06_LOW_VARIANCE",
            description=f"HR variance too low (std={hr_std:.1f})",
            severity=Severity.MEDIUM.value,
            details={"std": float(hr_std)}
        ))
    
    # HR07: Sudden drop (potential offset issue)
    sudden_drops = detect_sudden_changes(hr, timestamps, threshold=-Config.HR_SUDDEN_DROP)
    if sudden_drops:
        anomalies.append(Anomaly(
            code="HR07_SUDDEN_DROP",
            description="Sudden HR drop detected (possible offset issue)",
            severity=Severity.HIGH.value,
            details={"drops": sudden_drops}
        ))
    
    # HR08: Sudden rise (not at start)
    sudden_rises = detect_sudden_changes(hr, timestamps, threshold=Config.HR_SUDDEN_RISE, skip_start=True)
    if sudden_rises:
        anomalies.append(Anomaly(
            code="HR08_SUDDEN_RISE",
            description="Sudden HR rise detected mid-workout",
            severity=Severity.MEDIUM.value,
            details={"rises": sudden_rises}
        ))
    
    return anomalies


def detect_speed_anomalies(speed: np.ndarray, hr: np.ndarray) -> List[Anomaly]:
    """Detect speed anomalies."""
    anomalies = []
    
    if len(speed) == 0:
        return anomalies
    
    # SP01: Too high
    sustained_high = np.sum(speed > Config.SPEED_MAX_RUNNING)
    if sustained_high > 5:  # More than 5 points above threshold
        anomalies.append(Anomaly(
            code="SP01_TOO_HIGH",
            description=f"Speed >{Config.SPEED_MAX_RUNNING} km/h (sustained)",
            severity=Severity.HIGH.value,
            details={"count": int(sustained_high), "max_speed": float(np.max(speed))}
        ))
    
    # SP02: Negative
    negative_count = np.sum(speed < 0)
    if negative_count > 0:
        anomalies.append(Anomaly(
            code="SP02_NEGATIVE",
            description="Negative speed values",
            severity=Severity.HIGH.value,
            details={"count": int(negative_count)}
        ))
    
    # SP03: Zero while HR high
    if len(hr) == len(speed):
        zero_moving = np.sum((speed < 1) & (hr > 140))
        if zero_moving > 10:
            anomalies.append(Anomaly(
                code="SP03_ZERO_MOVING",
                description="Speed ~0 while HR >140 (GPS dropout?)",
                severity=Severity.MEDIUM.value,
                details={"count": int(zero_moving)}
            ))
    
    # SP04: Spike
    speed_diff = np.abs(np.diff(speed))
    spike_count = np.sum(speed_diff > Config.SPEED_SPIKE_THRESHOLD)
    if spike_count > 0:
        anomalies.append(Anomaly(
            code="SP04_SPIKE",
            description=f"Speed spikes >{Config.SPEED_SPIKE_THRESHOLD} km/h/timestep",
            severity=Severity.MEDIUM.value,
            details={"count": int(spike_count), "max_spike": float(np.max(speed_diff))}
        ))
    
    # SP05: HR-Speed mismatch (low correlation)
    if len(hr) == len(speed) and len(hr) > 50:
        valid_mask = (hr > 50) & (speed > 0)
        if np.sum(valid_mask) > 30:
            corr = np.corrcoef(speed[valid_mask], hr[valid_mask])[0, 1]
            if not np.isnan(corr) and abs(corr) < 0.05:
                anomalies.append(Anomaly(
                    code="SP05_HR_MISMATCH",
                    description="Speed-HR correlation near zero",
                    severity=Severity.LOW.value,
                    details={"correlation": float(corr)}
                ))
    
    return anomalies


def detect_timestamp_anomalies(timestamps: np.ndarray) -> List[Anomaly]:
    """Detect timestamp anomalies."""
    anomalies = []
    ts_diff = np.diff(timestamps)
    
    # TS01: Large gaps
    large_gaps = np.sum(ts_diff > Config.TS_GAP_LARGE)
    if large_gaps > 0:
        max_gap = int(np.max(ts_diff))
        anomalies.append(Anomaly(
            code="TS01_GAP_LARGE",
            description=f"Large timestamp gaps >{Config.TS_GAP_LARGE}s",
            severity=Severity.HIGH.value,
            details={"count": int(large_gaps), "max_gap_seconds": max_gap}
        ))
    
    # TS03: Negative (time going backwards)
    negative_ts = np.sum(ts_diff < 0)
    if negative_ts > 0:
        anomalies.append(Anomaly(
            code="TS03_NEGATIVE",
            description="Timestamps going backwards",
            severity=Severity.HIGH.value,
            details={"count": int(negative_ts)}
        ))
    
    # TS04: Duplicates
    duplicates = np.sum(ts_diff == 0)
    if duplicates > 0:
        anomalies.append(Anomaly(
            code="TS04_DUPLICATE",
            description="Duplicate timestamps",
            severity=Severity.MEDIUM.value,
            details={"count": int(duplicates)}
        ))
    
    # TS05: Irregular sampling
    ts_std = np.std(ts_diff)
    if ts_std > 30:
        anomalies.append(Anomaly(
            code="TS05_IRREGULAR",
            description="Highly irregular sampling rate",
            severity=Severity.LOW.value,
            details={"std_seconds": float(ts_std)}
        ))
    
    return anomalies


def detect_altitude_anomalies(altitude: np.ndarray) -> List[Anomaly]:
    """Detect altitude anomalies."""
    anomalies = []
    
    if len(altitude) == 0:
        return anomalies
    
    # AL01: Spikes
    alt_diff = np.abs(np.diff(altitude))
    spike_count = np.sum(alt_diff > Config.ALT_SPIKE_THRESHOLD)
    if spike_count > 0:
        anomalies.append(Anomaly(
            code="AL01_SPIKE",
            description=f"Altitude spikes >{Config.ALT_SPIKE_THRESHOLD}m/timestep",
            severity=Severity.MEDIUM.value,
            details={"count": int(spike_count), "max_spike": float(np.max(alt_diff))}
        ))
    
    # AL02: Unrealistic values
    if np.any(altitude < Config.ALT_MIN_VALID) or np.any(altitude > Config.ALT_MAX_VALID):
        anomalies.append(Anomaly(
            code="AL02_UNREALISTIC",
            description="Altitude outside valid range",
            severity=Severity.HIGH.value,
            details={"min": float(np.min(altitude)), "max": float(np.max(altitude))}
        ))
    
    return anomalies


def detect_data_integrity_issues(workout: Dict, hr: np.ndarray, timestamps: np.ndarray) -> List[Anomaly]:
    """Detect data integrity issues."""
    anomalies = []
    
    # DI02: Too short
    duration_min = (timestamps[-1] - timestamps[0]) / 60 if len(timestamps) > 1 else 0
    if duration_min < Config.MIN_DURATION_MINUTES:
        anomalies.append(Anomaly(
            code="DI02_TOO_SHORT",
            description=f"Workout too short ({duration_min:.1f} min)",
            severity=Severity.HIGH.value,
            details={"duration_minutes": float(duration_min)}
        ))
    
    if len(hr) < Config.MIN_DATA_POINTS:
        anomalies.append(Anomaly(
            code="DI02_TOO_FEW_POINTS",
            description=f"Too few data points ({len(hr)})",
            severity=Severity.HIGH.value,
            details={"point_count": len(hr)}
        ))
    
    # DI01: Length mismatch
    speed = workout.get('speed', [])
    altitude = workout.get('altitude', [])
    
    if len(speed) > 0 and len(speed) != len(hr):
        anomalies.append(Anomaly(
            code="DI01_LENGTH_MISMATCH",
            description="HR and speed arrays have different lengths",
            severity=Severity.HIGH.value,
            details={"hr_length": len(hr), "speed_length": len(speed)}
        ))
    
    if len(altitude) > 0 and len(altitude) != len(hr):
        anomalies.append(Anomaly(
            code="DI01_LENGTH_MISMATCH",
            description="HR and altitude arrays have different lengths",
            severity=Severity.HIGH.value,
            details={"hr_length": len(hr), "altitude_length": len(altitude)}
        ))
    
    return anomalies


def detect_offset_indicators(hr: np.ndarray, speed: np.ndarray, timestamps: np.ndarray) -> List[Anomaly]:
    """Detect indicators of potential HR offset errors."""
    anomalies = []
    
    if len(hr) < 50:
        return anomalies
    
    mean_hr = np.mean(hr)
    max_hr = np.max(hr)
    mean_speed = np.mean(speed) if len(speed) > 0 else 0
    
    # Flag 1: Low HR during decent running speed
    if mean_speed > Config.OFFSET_SUSPECT_SPEED and mean_hr < Config.OFFSET_SUSPECT_HR:
        estimated_offset = Config.HR_EXPECTED_MIN - mean_hr
        anomalies.append(Anomaly(
            code="PH01_OFFSET_SUSPECT",
            description=f"HR too low ({mean_hr:.0f}) for running speed ({mean_speed:.1f} km/h)",
            severity=Severity.HIGH.value,
            details={
                "mean_hr": float(mean_hr),
                "mean_speed": float(mean_speed),
                "estimated_offset": float(estimated_offset)
            }
        ))
    
    # Flag 2: Max HR never reaches effort zone
    duration_min = (timestamps[-1] - timestamps[0]) / 60
    if max_hr < 160 and duration_min > 20 and mean_speed > 8:
        anomalies.append(Anomaly(
            code="PH01_OFFSET_SUSPECT",
            description=f"Max HR ({max_hr}) never reaches effort zone in {duration_min:.0f}min workout",
            severity=Severity.HIGH.value,
            details={
                "max_hr": int(max_hr),
                "duration_min": float(duration_min)
            }
        ))
    
    # Flag 3: Good HR-speed correlation but wrong range
    if len(speed) == len(hr) and len(hr) > 50:
        valid_mask = (hr > 50) & (speed > 0)
        if np.sum(valid_mask) > 30:
            corr = np.corrcoef(speed[valid_mask], hr[valid_mask])[0, 1]
            if not np.isnan(corr) and corr > 0.3 and mean_hr < 140:
                anomalies.append(Anomaly(
                    code="PH01_OFFSET_SUSPECT",
                    description=f"Good HR-speed correlation ({corr:.2f}) but HR range too low",
                    severity=Severity.HIGH.value,
                    details={
                        "correlation": float(corr),
                        "mean_hr": float(mean_hr)
                    }
                ))
    
    return anomalies


# =============================================================================
# Helper Functions
# =============================================================================

def find_max_consecutive(arr: np.ndarray) -> int:
    """Find maximum consecutive identical values."""
    if len(arr) == 0:
        return 0
    
    max_run = 1
    current_run = 1
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    return max_run


def detect_sudden_changes(hr: np.ndarray, timestamps: np.ndarray, threshold: float, 
                          skip_start: bool = False, window_seconds: float = 60) -> List[Dict]:
    """Detect sudden HR changes within a time window."""
    changes = []
    
    start_idx = 10 if skip_start else 0  # Skip first 10 points if skip_start
    
    for i in range(start_idx, len(hr) - 1):
        # Look for changes within window_seconds
        j = i + 1
        while j < len(hr) and (timestamps[j] - timestamps[i]) < window_seconds:
            change = hr[j] - hr[i]
            
            if threshold < 0 and change < threshold:  # Looking for drops
                time_min = (timestamps[i] - timestamps[0]) / 60
                changes.append({
                    "time_min": float(time_min),
                    "from_hr": int(hr[i]),
                    "to_hr": int(hr[j]),
                    "change": int(change)
                })
                break
            elif threshold > 0 and change > threshold:  # Looking for rises
                time_min = (timestamps[i] - timestamps[0]) / 60
                changes.append({
                    "time_min": float(time_min),
                    "from_hr": int(hr[i]),
                    "to_hr": int(hr[j]),
                    "change": int(change)
                })
                break
            j += 1
    
    # Deduplicate (keep only distinct events)
    if len(changes) > 1:
        filtered = [changes[0]]
        for c in changes[1:]:
            if c["time_min"] - filtered[-1]["time_min"] > 1:  # At least 1 min apart
                filtered.append(c)
        return filtered[:5]  # Max 5 events
    
    return changes


def compute_stats(workout: Dict, hr: np.ndarray, timestamps: np.ndarray) -> Dict:
    """Compute summary statistics for a workout."""
    speed = np.array(workout.get('speed', []))
    altitude = np.array(workout.get('altitude', []))
    
    duration_min = (timestamps[-1] - timestamps[0]) / 60 if len(timestamps) > 1 else 0
    
    stats = {
        "duration_min": float(duration_min),
        "data_points": len(hr),
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "hr_min": int(np.min(hr)),
        "hr_max": int(np.max(hr)),
    }
    
    if len(speed) > 0:
        stats["speed_mean"] = float(np.mean(speed))
        stats["speed_max"] = float(np.max(speed))
        
        # Correlation
        if len(speed) == len(hr):
            valid_mask = (hr > 50) & (speed > 0)
            if np.sum(valid_mask) > 30:
                corr = np.corrcoef(speed[valid_mask], hr[valid_mask])[0, 1]
                if not np.isnan(corr):
                    stats["hr_speed_correlation"] = float(corr)
    
    if len(altitude) > 0:
        stats["altitude_min"] = float(np.min(altitude))
        stats["altitude_max"] = float(np.max(altitude))
        alt_diff = np.diff(altitude)
        stats["elevation_gain"] = float(np.sum(alt_diff[alt_diff > 0]))
    
    return stats


def make_decision(anomalies: List[Anomaly]) -> Tuple[Decision, List[str]]:
    """Make a decision based on detected anomalies."""
    
    # Hard exclusion codes
    exclude_codes = {
        "HR02_TOO_LOW", "HR03_TOO_HIGH", "SP02_NEGATIVE", 
        "TS03_NEGATIVE", "DI01_LENGTH_MISMATCH", "DI02_TOO_SHORT",
        "DI02_TOO_FEW_POINTS", "AL02_UNREALISTIC"
    }
    
    # Codes that require LLM review
    flag_codes = {
        "HR01_ZERO", "HR04_SPIKE", "HR05_FLATLINE", "HR06_LOW_VARIANCE",
        "HR07_SUDDEN_DROP", "HR08_SUDDEN_RISE", 
        "SP01_TOO_HIGH", "SP03_ZERO_MOVING", "SP04_SPIKE",
        "TS01_GAP_LARGE", "AL01_SPIKE",
        "PH01_OFFSET_SUSPECT"
    }
    
    flags = []
    
    for anomaly in anomalies:
        if anomaly.code in exclude_codes:
            return Decision.EXCLUDE, [anomaly.code]
        if anomaly.code in flag_codes:
            flags.append(anomaly.code)
    
    if flags:
        return Decision.FLAG, flags
    
    return Decision.PASS, []


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_workout(workout: Dict, line_number: int) -> WorkoutAnalysis:
    """Analyze a single workout for anomalies."""
    
    hr = np.array(workout.get('heart_rate', []))
    timestamps = np.array(workout.get('timestamp', []))
    speed = np.array(workout.get('speed', []))
    altitude = np.array(workout.get('altitude', []))
    
    # Check for missing essential data
    if len(hr) == 0:
        return WorkoutAnalysis(
            line_number=line_number,
            workout_id=workout.get('id'),
            user_id=workout.get('userId'),
            decision=Decision.EXCLUDE.value,
            anomalies=[asdict(Anomaly(
                code="DI04_MISSING_HR",
                description="No heart rate data",
                severity=Severity.HIGH.value
            ))],
            stats={},
            flags=["DI04_MISSING_HR"]
        )
    
    if len(timestamps) == 0:
        return WorkoutAnalysis(
            line_number=line_number,
            workout_id=workout.get('id'),
            user_id=workout.get('userId'),
            decision=Decision.EXCLUDE.value,
            anomalies=[asdict(Anomaly(
                code="DI04_MISSING_TS",
                description="No timestamp data",
                severity=Severity.HIGH.value
            ))],
            stats={},
            flags=["DI04_MISSING_TS"]
        )
    
    # Collect all anomalies
    all_anomalies = []
    all_anomalies.extend(detect_hr_anomalies(hr, timestamps))
    all_anomalies.extend(detect_speed_anomalies(speed, hr))
    all_anomalies.extend(detect_timestamp_anomalies(timestamps))
    all_anomalies.extend(detect_altitude_anomalies(altitude))
    all_anomalies.extend(detect_data_integrity_issues(workout, hr, timestamps))
    all_anomalies.extend(detect_offset_indicators(hr, speed, timestamps))
    
    # Make decision
    decision, flags = make_decision(all_anomalies)
    
    # Compute stats
    stats = compute_stats(workout, hr, timestamps)
    
    # Suggest action for flagged workouts
    suggested_action = None
    if decision == Decision.FLAG:
        if "PH01_OFFSET_SUSPECT" in flags or "HR07_SUDDEN_DROP" in flags:
            suggested_action = "LLM_REVIEW_OFFSET"
        else:
            suggested_action = "LLM_REVIEW_GENERAL"
    
    return WorkoutAnalysis(
        line_number=line_number,
        workout_id=workout.get('id'),
        user_id=workout.get('userId'),
        decision=decision.value,
        anomalies=[asdict(a) for a in all_anomalies],
        stats=stats,
        flags=flags,
        suggested_action=suggested_action
    )


# =============================================================================
# Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 1: Rule-based workout filtering')
    parser.add_argument('--input', '-i', type=str, 
                        default='DATA/raw/endomondoHR_proper-002.json',
                        help='Input data file')
    parser.add_argument('--index', '-x', type=str,
                        default='DATA/indices/running_all.txt',
                        help='Index file with line numbers to process')
    parser.add_argument('--output', '-o', type=str,
                        default='Preprocessing/stage1_output.json',
                        help='Output JSON file')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of workouts to process')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / args.input
    index_file = project_root / args.index
    output_file = project_root / args.output
    
    print(f"Stage 1: Rule-Based Filtering")
    print(f"=" * 60)
    print(f"Input: {input_file}")
    print(f"Index: {index_file}")
    print(f"Output: {output_file}")
    print()
    
    # Load index
    with open(index_file, 'r') as f:
        line_numbers = [int(line.strip()) for line in f if line.strip()]
    
    if args.limit:
        line_numbers = line_numbers[:args.limit]
    
    print(f"Processing {len(line_numbers)} workouts...")
    print()
    
    # Process workouts
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_file),
            "total_processed": 0,
            "pass_count": 0,
            "flag_count": 0,
            "exclude_count": 0,
            "config": {
                "HR_MIN_VALID": Config.HR_MIN_VALID,
                "HR_MAX_VALID": Config.HR_MAX_VALID,
                "HR_EXPECTED_MIN": Config.HR_EXPECTED_MIN,
                "HR_EXPECTED_MAX": Config.HR_EXPECTED_MAX,
                "SPEED_MAX_RUNNING": Config.SPEED_MAX_RUNNING,
                "MIN_DURATION_MINUTES": Config.MIN_DURATION_MINUTES,
            }
        },
        "workouts": []
    }
    
    pass_count = 0
    flag_count = 0
    exclude_count = 0
    error_count = 0
    
    for i, line_num in enumerate(line_numbers):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(line_numbers)}...")
        
        try:
            line = linecache.getline(str(input_file), line_num)
            if not line.strip():
                continue
            
            workout = ast.literal_eval(line.strip())
            analysis = analyze_workout(workout, line_num)
            
            results["workouts"].append(analysis.to_dict())
            
            if analysis.decision == "PASS":
                pass_count += 1
            elif analysis.decision == "FLAG":
                flag_count += 1
                if args.verbose:
                    print(f"  FLAG: Line {line_num}, ID {analysis.workout_id}, Flags: {analysis.flags}")
            else:
                exclude_count += 1
                if args.verbose:
                    print(f"  EXCLUDE: Line {line_num}, ID {analysis.workout_id}, Flags: {analysis.flags}")
        
        except Exception as e:
            error_count += 1
            results["workouts"].append({
                "line_number": line_num,
                "workout_id": None,
                "user_id": None,
                "decision": "EXCLUDE",
                "anomalies": [{"code": "DI06_PARSE_ERROR", "description": str(e), "severity": "HIGH"}],
                "stats": {},
                "flags": ["DI06_PARSE_ERROR"]
            })
    
    # Clear linecache
    linecache.clearcache()
    
    # Update metadata
    results["metadata"]["total_processed"] = len(line_numbers)
    results["metadata"]["pass_count"] = pass_count
    results["metadata"]["flag_count"] = flag_count
    results["metadata"]["exclude_count"] = exclude_count
    results["metadata"]["error_count"] = error_count
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Total processed: {len(line_numbers)}")
    print(f"  PASS:    {pass_count:6d} ({pass_count/len(line_numbers)*100:5.1f}%)")
    print(f"  FLAG:    {flag_count:6d} ({flag_count/len(line_numbers)*100:5.1f}%)")
    print(f"  EXCLUDE: {exclude_count:6d} ({exclude_count/len(line_numbers)*100:5.1f}%)")
    print(f"  ERRORS:  {error_count:6d} ({error_count/len(line_numbers)*100:5.1f}%)")
    print()
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
