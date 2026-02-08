#!/usr/bin/env python3
"""Test memory usage of loading DailyDialog dataset."""
import psutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_mem(label):
    proc = psutil.Process()
    mem = proc.memory_info()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"{label}:")
    print(f"  Process RSS: {mem.rss/1024/1024:.1f} MB")
    print(f"  System: {vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB ({vm.percent:.1f}%), swap={swap.used/1024**3:.2f} GB")

print_mem("BEFORE import datasets")

from datasets import load_dataset

print_mem("\nAFTER import datasets")

print("\nLoading DailyDialog dataset...")
ds = load_dataset("OpenRL/daily_dialog", split="train")

print_mem(f"\nAFTER loading dataset ({len(ds)} dialogues)")

print("\nIterating to create examples...")
examples = []
for dialogue in ds:
    utterances = dialogue["dialog"]
    acts = dialogue["act"]

    for i in range(1, len(utterances)):
        examples.append({
            "text": utterances[i],
            "last_message": utterances[i-1],
            "label": str(acts[i]),
            "context": utterances[max(0, i-5):i],
        })

print_mem(f"\nAFTER creating {len(examples)} examples (dataset still in memory)")

del ds
import gc
gc.collect()

print_mem("\nAFTER deleting dataset")
