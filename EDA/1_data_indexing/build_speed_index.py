#!/usr/bin/env python3
"""
Build byte-offset index for computed speeds JSONL file.

Creates a simple lookup file: line_num -> byte_offset
This allows O(1) random access to any workout's speed data.

Author: OpenCode
Date: 2025-01-11
"""

import json
from pathlib import Path
from time import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COMPUTED_SPEED_FILE = PROJECT_ROOT / "DATA" / "processed" / "running_computed_speed.jsonl"
INDEX_FILE = PROJECT_ROOT / "DATA" / "indices" / "computed_speed_offsets.idx"


def build_byte_offset_index():
    """Build index mapping line_num to byte offset in JSONL file."""
    
    print(f"Building byte-offset index for {COMPUTED_SPEED_FILE}...")
    start_time = time()
    
    offsets = {}
    
    with open(COMPUTED_SPEED_FILE, 'rb') as f:
        while True:
            byte_offset = f.tell()
            line = f.readline()
            
            if not line:
                break
            
            try:
                data = json.loads(line.decode('utf-8'))
                line_num = data['line_num']
                offsets[line_num] = byte_offset
            except (json.JSONDecodeError, KeyError):
                continue
            
            if len(offsets) % 10000 == 0:
                print(f"  Indexed {len(offsets):,} entries...")
    
    # Save index as binary format for fast loading
    # Format: line_num (4 bytes int) + byte_offset (8 bytes long)
    with open(INDEX_FILE, 'wb') as f:
        import struct
        for line_num, offset in sorted(offsets.items()):
            f.write(struct.pack('<IQ', line_num, offset))
    
    elapsed = time() - start_time
    
    print(f"\nIndex built in {elapsed:.1f} seconds")
    print(f"Total entries: {len(offsets):,}")
    print(f"Index file: {INDEX_FILE}")
    print(f"Index size: {INDEX_FILE.stat().st_size / 1024:.1f} KB")
    
    return offsets


if __name__ == "__main__":
    build_byte_offset_index()
