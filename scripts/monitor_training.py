#!/usr/bin/env python3
"""Monitor a running training process - shows memory, CPU, and runtime."""

import sys
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.utils.memory import get_memory_info, get_swap_info


def find_training_process():
    """Find the running training script process."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("train_category_svm.py" in arg for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def monitor_process(pid: int, interval: float = 5.0):
    """Monitor a process and log its state."""
    proc = psutil.Process(pid)
    start_time = time.time()

    print(f"Monitoring PID {pid}: {' '.join(proc.cmdline())}")
    print(f"Started at: {time.strftime('%H:%M:%S', time.localtime(proc.create_time()))}\n")

    print(f"{'Elapsed':>8s} {'CPU%':>6s} {'RAM(MB)':>9s} {'Swap(MB)':>10s} {'Status':>10s}")
    print("-" * 60)

    try:
        while proc.is_running():
            elapsed = time.time() - start_time
            mem = proc.memory_info()
            cpu = proc.cpu_percent(interval=1.0)
            swap = get_swap_info()

            status = proc.status()
            ram_mb = mem.rss / 1024**2

            print(
                f"{int(elapsed):>5d}s  {cpu:>5.1f}%  {ram_mb:>8.1f}  {swap['used_mb']:>9.1f}  {status:>10s}",
                flush=True,
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    except psutil.NoSuchProcess:
        print("\nProcess terminated")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
    else:
        proc = find_training_process()
        if proc is None:
            print("No training process found")
            sys.exit(1)
        pid = proc.pid

    monitor_process(pid)
