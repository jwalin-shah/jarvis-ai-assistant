#!/usr/bin/env python3
"""Minimal test to pinpoint where 5GB allocation happens."""
import sys
from pathlib import Path
import psutil

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

proc = psutil.Process()

print("=" * 80)
print("MEMORY TRACE: Loading embedder")
print("=" * 80)

mem = proc.memory_info()
print(f"\n[MAIN] START: RSS={mem.rss/1024/1024:.1f} MB, VMS={mem.vms/1024/1024:.1f} MB")

print("\n[MAIN] About to import embedding_adapter...")
from jarvis.embedding_adapter import get_embedder

mem = proc.memory_info()
print(f"[MAIN] After import: RSS={mem.rss/1024/1024:.1f} MB, VMS={mem.vms/1024/1024:.1f} MB")

print("\n[MAIN] About to call get_embedder()...")
embedder = get_embedder()

mem = proc.memory_info()
print(f"[MAIN] After get_embedder(): RSS={mem.rss/1024/1024:.1f} MB, VMS={mem.vms/1024/1024:.1f} MB")

print("\n[MAIN] About to encode test text...")
emb = embedder.encode(["test"])

mem = proc.memory_info()
print(f"[MAIN] After encode: RSS={mem.rss/1024/1024:.1f} MB, VMS={mem.vms/1024/1024:.1f} MB")

print(f"\n[MAIN] Embedding shape: {emb.shape}")
print("\n" + "=" * 80)
print("CHECK ACTIVITY MONITOR NOW - what's the total memory?")
print(f"psutil reports: RSS={mem.rss/1024/1024:.1f} MB, VMS={mem.vms/1024/1024/1024:.1f} GB")
print("=" * 80)

import time
print("\nWaiting 60 seconds so you can check Activity Monitor...")
print("Press Ctrl+C to exit early")
try:
    time.sleep(60)
except KeyboardInterrupt:
    print("\nInterrupted by user")
print("\nDone.")
