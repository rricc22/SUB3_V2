#!/usr/bin/env python3
"""
Stage 2: LLM Validation for Flagged Workouts

Uses Groq's free API (Llama 3.1 70B) to:
1. Validate flagged workouts
2. Detect and quantify HR offset errors
3. Classify workout types

Author: OpenCode
Date: 2025-01-11
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse
import ast
import linecache

# Groq API
from groq import Groq

# =============================================================================
# Configuration
# =============================================================================

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # Set via environment variable
MODEL = "llama-3.3-70b-versatile"  # Best free model on Groq

# Rate limiting (Groq free tier: 14,400 req/day, 30 req/min)
REQUESTS_PER_MINUTE = 28  # Stay under 30
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # ~2.1 seconds

# Triage settings
BENIGN_FLAGS = {'HR04_SPIKE', 'HR08_SUDDEN_RISE', 'SP04_SPIKE', 'AL01_SPIKE', 'TS05_IRREGULAR', 'SP05_HR_MISMATCH'}
OFFSET_FLAGS = {'PH01_OFFSET_SUSPECT', 'HR07_SUDDEN_DROP'}


# =============================================================================
# LLM Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are an expert sports physiologist analyzing heart rate data from running workouts.

Your task is to:
1. Determine if the heart rate data is valid or has sensor errors
2. Detect HR offset errors (where HR is shifted by a constant value)
3. Classify the workout type

IMPORTANT CONTEXT:
- Normal running HR should be 150-190 BPM for effort
- Recovery/easy running: 130-155 BPM
- If HR shows 90-130 BPM during running speeds >10 km/h, suspect OFFSET ERROR
- Offset errors are common: the sensor reads 20-60 BPM too low

OFFSET CALCULATION:
- Calculate the actual offset needed: offset = expected_HR - observed_HR
- Example: if mean HR is 125 BPM but should be ~155 BPM, offset = +30 BPM
- DO NOT default to +20 BPM - calculate the actual value needed!

OFFSET TYPES:
1. FULL WORKOUT OFFSET: Same offset applies to entire workout
2. PARTIAL/ADAPTIVE OFFSET: Offset starts or ends at a specific time
   - Use when you see a sudden HR drop/rise mid-workout
   - Specify start_time_min and end_time_min for when offset applies

WORKOUT TYPES:
- INTENSIVE: Sustained high HR (170-190+ BPM), threshold/tempo runs
- INTERVALS: HR oscillates between high (160-185) and recovery (130-155)
- STEADY: Consistent moderate HR (145-170 BPM), easy-moderate runs
- RECOVERY: Low HR throughout (120-150 BPM), easy jogs

Respond ONLY with valid JSON, no other text."""

USER_PROMPT_TEMPLATE = """Analyze this running workout data:

WORKOUT ID: {workout_id}
DURATION: {duration_min:.1f} minutes
DATA POINTS: {data_points}

HR STATISTICS:
- Mean: {hr_mean:.0f} BPM
- Std: {hr_std:.1f} BPM
- Min: {hr_min} BPM
- Max: {hr_max} BPM

SPEED STATISTICS:
- Mean: {speed_mean:.1f} km/h
- Max: {speed_max:.1f} km/h

HR-SPEED CORRELATION: {correlation:.2f}

FLAGS DETECTED: {flags}

HR SAMPLES (every 2 min):
{hr_samples}

SPEED SAMPLES (every 2 min):
{speed_samples}

Based on this data, provide your analysis as JSON:
{{
    "decision": "KEEP" | "FIX" | "EXCLUDE",
    "confidence": 0.0-1.0,
    "workout_type": "INTENSIVE" | "INTERVALS" | "STEADY" | "RECOVERY" | "UNKNOWN",
    "has_offset_error": true | false,
    "estimated_offset": <number or null>,  // positive means add this to HR
    "offset_start_pct": <0-100 or null>,  // where offset starts (% of workout)
    "reasoning": "<brief explanation>"
}}

DECISION GUIDELINES:
- KEEP: Data looks valid, no correction needed
- FIX: Offset error detected, apply estimated_offset correction
- EXCLUDE: Data is too corrupted to salvage"""


# =============================================================================
# Helper Functions
# =============================================================================

def sample_array(arr: np.ndarray, interval_minutes: float, timestamps: np.ndarray) -> str:
    """Sample array at regular intervals for LLM context."""
    if len(arr) == 0:
        return "N/A"
    
    duration = (timestamps[-1] - timestamps[0]) / 60  # minutes
    samples = []
    
    for t in np.arange(0, duration, interval_minutes):
        # Find closest index to this time
        target_ts = timestamps[0] + t * 60
        idx = np.argmin(np.abs(timestamps - target_ts))
        samples.append(f"{t:.0f}min: {arr[idx]:.0f}" if isinstance(arr[idx], float) else f"{t:.0f}min: {arr[idx]}")
    
    return ", ".join(samples[:15])  # Max 15 samples


def prepare_workout_context(workout: Dict, analysis: Dict) -> Dict:
    """Prepare context for LLM from workout data and stage1 analysis."""
    hr = np.array(workout.get('heart_rate', []))
    speed = np.array(workout.get('speed', []))
    timestamps = np.array(workout.get('timestamp', []))
    
    stats = analysis.get('stats', {})
    
    return {
        'workout_id': workout.get('id', 'unknown'),
        'duration_min': stats.get('duration_min', 0),
        'data_points': len(hr),
        'hr_mean': stats.get('hr_mean', 0),
        'hr_std': stats.get('hr_std', 0),
        'hr_min': stats.get('hr_min', 0),
        'hr_max': stats.get('hr_max', 0),
        'speed_mean': stats.get('speed_mean', 0),
        'speed_max': stats.get('speed_max', 0),
        'correlation': stats.get('hr_speed_correlation', 0),
        'flags': ', '.join(analysis.get('flags', [])),
        'hr_samples': sample_array(hr, 2, timestamps) if len(hr) > 0 else "N/A",
        'speed_samples': sample_array(speed, 2, timestamps) if len(speed) > 0 else "N/A"
    }


def parse_llm_response(response_text: str) -> Optional[Dict]:
    """Parse LLM JSON response, handling common issues."""
    try:
        # Try direct parse
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return None


# =============================================================================
# Triage Functions
# =============================================================================

def triage_workout(analysis: Dict) -> str:
    """
    Triage flagged workouts to minimize LLM calls.
    
    Returns:
        'AUTO_PASS': Can pass without LLM (benign flags only)
        'LLM_OFFSET': Needs LLM for offset detection
        'LLM_REVIEW': Needs LLM for general review
    """
    flags = set(analysis.get('flags', []))
    
    # If only benign flags, auto-pass
    if flags.issubset(BENIGN_FLAGS):
        return 'AUTO_PASS'
    
    # If has offset-related flags, prioritize offset detection
    if flags & OFFSET_FLAGS:
        return 'LLM_OFFSET'
    
    return 'LLM_REVIEW'


# =============================================================================
# Main LLM Validation
# =============================================================================

class Stage2Validator:
    def __init__(self, api_key: str = GROQ_API_KEY):
        self.client = Groq(api_key=api_key)
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < DELAY_BETWEEN_REQUESTS:
            time.sleep(DELAY_BETWEEN_REQUESTS - elapsed)
        self.last_request_time = time.time()
    
    def validate_workout(self, workout: Dict, analysis: Dict) -> Dict:
        """Send workout to LLM for validation."""
        self._rate_limit()
        
        context = prepare_workout_context(workout, analysis)
        user_prompt = USER_PROMPT_TEMPLATE.format(**context)
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent responses
                max_tokens=500
            )
            
            self.request_count += 1
            response_text = response.choices[0].message.content
            
            parsed = parse_llm_response(response_text)
            if parsed:
                return {
                    'success': True,
                    'llm_response': parsed,
                    'raw_response': response_text
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to parse JSON',
                    'raw_response': response_text
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'raw_response': None
            }


# =============================================================================
# Main Processing
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2: LLM validation for flagged workouts')
    parser.add_argument('--input', '-i', type=str,
                        default='DATA/raw/endomondoHR_proper-002.json',
                        help='Raw data file')
    parser.add_argument('--stage1', '-s', type=str,
                        default='Preprocessing/stage1_full_output.json',
                        help='Stage 1 output file')
    parser.add_argument('--output', '-o', type=str,
                        default='Preprocessing/stage2_output.json',
                        help='Output file')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of LLM calls')
    parser.add_argument('--skip-triage', action='store_true',
                        help='Skip triage, send all flagged to LLM')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run - show what would be processed')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / args.input
    stage1_file = project_root / args.stage1
    output_file = project_root / args.output
    
    print(f"Stage 2: LLM Validation")
    print(f"=" * 60)
    print(f"Raw data: {input_file}")
    print(f"Stage 1: {stage1_file}")
    print(f"Output: {output_file}")
    print(f"Model: {MODEL}")
    print()
    
    # Load stage 1 results
    with open(stage1_file, 'r') as f:
        stage1_data = json.load(f)
    
    # Filter to flagged workouts only
    flagged = [w for w in stage1_data['workouts'] if w['decision'] == 'FLAG']
    print(f"Total flagged workouts: {len(flagged)}")
    
    # Triage
    triage_results = {'AUTO_PASS': [], 'LLM_OFFSET': [], 'LLM_REVIEW': []}
    for w in flagged:
        triage = triage_workout(w)
        triage_results[triage].append(w)
    
    print(f"\nTriage results:")
    print(f"  AUTO_PASS (benign flags): {len(triage_results['AUTO_PASS'])}")
    print(f"  LLM_OFFSET (offset suspects): {len(triage_results['LLM_OFFSET'])}")
    print(f"  LLM_REVIEW (other): {len(triage_results['LLM_REVIEW'])}")
    
    if args.skip_triage:
        to_process = flagged
    else:
        to_process = triage_results['LLM_OFFSET'] + triage_results['LLM_REVIEW']
    
    print(f"\nWorkouts to send to LLM: {len(to_process)}")
    
    if args.limit:
        to_process = to_process[:args.limit]
        print(f"Limited to: {len(to_process)}")
    
    if args.dry_run:
        print("\nDry run - exiting")
        return
    
    # Initialize validator
    validator = Stage2Validator()
    
    # Process results structure
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'stage1_file': str(stage1_file),
            'model': MODEL,
            'total_flagged': len(flagged),
            'auto_passed': len(triage_results['AUTO_PASS']),
            'llm_processed': 0,
            'llm_keep': 0,
            'llm_fix': 0,
            'llm_exclude': 0,
            'llm_errors': 0
        },
        'auto_passed': [
            {
                'line_number': w['line_number'],
                'workout_id': w['workout_id'],
                'decision': 'KEEP',
                'source': 'AUTO_PASS_TRIAGE',
                'original_flags': w['flags'],
                'workout_type': None,  # Will classify later
                'stats': w['stats']
            }
            for w in triage_results['AUTO_PASS']
        ],
        'llm_validated': []
    }
    
    # Process with LLM
    print(f"\nStarting LLM validation...")
    print(f"Rate limit: {REQUESTS_PER_MINUTE} req/min ({DELAY_BETWEEN_REQUESTS:.1f}s delay)")
    print()
    
    for i, analysis in enumerate(to_process):
        line_num = analysis['line_number']
        workout_id = analysis['workout_id']
        
        # Progress
        if (i + 1) % 10 == 0:
            eta_min = (len(to_process) - i - 1) * DELAY_BETWEEN_REQUESTS / 60
            print(f"  [{i+1}/{len(to_process)}] ETA: {eta_min:.0f} min | "
                  f"KEEP: {results['metadata']['llm_keep']} | "
                  f"FIX: {results['metadata']['llm_fix']} | "
                  f"EXCLUDE: {results['metadata']['llm_exclude']}")
        
        # Load raw workout data
        try:
            line = linecache.getline(str(input_file), line_num)
            workout = ast.literal_eval(line.strip())
        except Exception as e:
            results['llm_validated'].append({
                'line_number': line_num,
                'workout_id': workout_id,
                'decision': 'EXCLUDE',
                'source': 'PARSE_ERROR',
                'error': str(e)
            })
            results['metadata']['llm_errors'] += 1
            continue
        
        # Validate with LLM
        llm_result = validator.validate_workout(workout, analysis)
        results['metadata']['llm_processed'] += 1
        
        if llm_result['success']:
            llm_resp = llm_result['llm_response']
            decision = llm_resp.get('decision', 'EXCLUDE')
            
            entry = {
                'line_number': line_num,
                'workout_id': workout_id,
                'decision': decision,
                'source': 'LLM',
                'confidence': llm_resp.get('confidence', 0),
                'workout_type': llm_resp.get('workout_type', 'UNKNOWN'),
                'has_offset_error': llm_resp.get('has_offset_error', False),
                'estimated_offset': llm_resp.get('estimated_offset'),
                'offset_start_pct': llm_resp.get('offset_start_pct'),
                'reasoning': llm_resp.get('reasoning', ''),
                'original_flags': analysis['flags'],
                'stats': analysis['stats']
            }
            
            results['llm_validated'].append(entry)
            
            # Update counts
            if decision == 'KEEP':
                results['metadata']['llm_keep'] += 1
            elif decision == 'FIX':
                results['metadata']['llm_fix'] += 1
            else:
                results['metadata']['llm_exclude'] += 1
        else:
            results['llm_validated'].append({
                'line_number': line_num,
                'workout_id': workout_id,
                'decision': 'EXCLUDE',
                'source': 'LLM_ERROR',
                'error': llm_result.get('error', 'Unknown error'),
                'raw_response': llm_result.get('raw_response')
            })
            results['metadata']['llm_errors'] += 1
        
        # Save intermediate results every 100 workouts
        if (i + 1) % 100 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Clear linecache
    linecache.clearcache()
    
    # Final save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print()
    print(f"=" * 60)
    print(f"STAGE 2 SUMMARY")
    print(f"=" * 60)
    print(f"Total flagged: {len(flagged)}")
    print(f"Auto-passed (triage): {results['metadata']['auto_passed']}")
    print(f"LLM processed: {results['metadata']['llm_processed']}")
    print(f"  - KEEP: {results['metadata']['llm_keep']}")
    print(f"  - FIX: {results['metadata']['llm_fix']}")
    print(f"  - EXCLUDE: {results['metadata']['llm_exclude']}")
    print(f"  - Errors: {results['metadata']['llm_errors']}")
    print()
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
