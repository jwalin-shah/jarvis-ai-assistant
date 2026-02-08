#!/usr/bin/env python3
"""Test memory usage of the filtered list (76k dicts with context)."""
import psutil

print("Creating 76k dicts (simulating filtered list)...")

proc = psutil.Process()
mem_start = proc.memory_info()
vm_start = psutil.virtual_memory()
swap_start = psutil.swap_memory()

print(f"BEFORE: RSS={mem_start.rss/1024/1024:.1f} MB")
print(f"  System: {vm_start.used/1024**3:.1f}/{vm_start.total/1024**3:.1f} GB ({vm_start.percent:.1f}%), swap={swap_start.used/1024**3:.2f} GB\n")

# Simulate the filtered list from prepare_dailydialog_data.py
filtered = []
for i in range(76000):
    filtered.append({
        'text': f'This is response message number {i} with some text that could be longer',
        'last_message': f'This is the last message before response {i}, could be even longer',
        'label': 'inform',
        'context': [
            f'Context message {j} for example {i} with more text to simulate real messages'
            for j in range(3)  # avg 3 context messages
        ],
        'source': 'dailydialog',
    })

    if (i + 1) % 20000 == 0:
        mem = proc.memory_info()
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        print(f"{i+1:6d} dicts: RSS={mem.rss/1024/1024:.1f} MB, system={vm.used/1024**3:.1f} GB ({vm.percent:.1f}%), swap={swap.used/1024**3:.2f} GB")

mem_end = proc.memory_info()
vm_end = psutil.virtual_memory()
swap_end = psutil.swap_memory()

print(f"\nAFTER 76k dicts: RSS={mem_end.rss/1024/1024:.1f} MB (+{(mem_end.rss-mem_start.rss)/1024/1024:.1f} MB)")
print(f"  System: {vm_end.used/1024**3:.1f}/{vm_end.total/1024**3:.1f} GB ({vm_end.percent:.1f}%), swap={swap_end.used/1024**3:.2f} GB")
print(f"  Swap increased by: {(swap_end.used - swap_start.used)/1024**2:.1f} MB")
