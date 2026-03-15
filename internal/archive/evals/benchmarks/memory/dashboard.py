"""Memory profiling dashboard for visualization.  # noqa: E501
  # noqa: E501
Provides utilities for:  # noqa: E501
- Collecting and displaying memory usage over time  # noqa: E501
- Generating text-based visualizations for terminal  # noqa: E501
- Exporting memory data for external analysis  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import json  # noqa: E501
import os  # noqa: E501
import time  # noqa: E501
from dataclasses import asdict, dataclass  # noqa: E402  # noqa: E501
from datetime import UTC, datetime  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

# noqa: E501
import psutil  # noqa: E501

# noqa: E501
from jarvis.metrics import MemorySampler, get_memory_sampler  # noqa: E402  # noqa: E501

  # noqa: E501
# Constants  # noqa: E501
BYTES_PER_MB = 1024 * 1024  # noqa: E501
BYTES_PER_GB = 1024**3  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class MemorySnapshot:  # noqa: E501
    """A comprehensive memory snapshot at a point in time."""  # noqa: E501
  # noqa: E501
    timestamp: str  # noqa: E501
    process_rss_mb: float  # noqa: E501
    process_vms_mb: float  # noqa: E501
    system_total_gb: float  # noqa: E501
    system_available_gb: float  # noqa: E501
    system_used_gb: float  # noqa: E501
    system_percent: float  # noqa: E501
    metal_gpu_mb: float  # noqa: E501
    gc_objects: int  # noqa: E501
  # noqa: E501
  # noqa: E501
def _get_metal_memory_mb() -> float:  # noqa: E501
    """Get Metal GPU memory if available."""  # noqa: E501
    try:  # noqa: E501
        import mlx.core as mx  # noqa: E501
  # noqa: E501
        return mx.metal.get_peak_memory() / BYTES_PER_MB  # noqa: E501
    except (ImportError, AttributeError):  # noqa: E501
        return 0.0  # noqa: E501
  # noqa: E501
  # noqa: E501
def _get_gc_objects() -> int:  # noqa: E501
    """Get count of tracked garbage collector objects."""  # noqa: E501
    import gc  # noqa: E501
  # noqa: E501
    return len(gc.get_objects())  # noqa: E501
  # noqa: E501
  # noqa: E501
def take_snapshot() -> MemorySnapshot:  # noqa: E501
    """Take a comprehensive memory snapshot.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        MemorySnapshot with current memory state  # noqa: E501
    """  # noqa: E501
    process = psutil.Process(os.getpid())  # noqa: E501
    mem_info = process.memory_info()  # noqa: E501
    system_mem = psutil.virtual_memory()  # noqa: E501
  # noqa: E501
    return MemorySnapshot(  # noqa: E501
        timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
        process_rss_mb=round(mem_info.rss / BYTES_PER_MB, 2),  # noqa: E501
        process_vms_mb=round(mem_info.vms / BYTES_PER_MB, 2),  # noqa: E501
        system_total_gb=round(system_mem.total / BYTES_PER_GB, 2),  # noqa: E501
        system_available_gb=round(system_mem.available / BYTES_PER_GB, 2),  # noqa: E501
        system_used_gb=round(system_mem.used / BYTES_PER_GB, 2),  # noqa: E501
        system_percent=round(system_mem.percent, 1),  # noqa: E501
        metal_gpu_mb=round(_get_metal_memory_mb(), 2),  # noqa: E501
        gc_objects=_get_gc_objects(),  # noqa: E501
    )  # noqa: E501
  # noqa: E501
  # noqa: E501
class MemoryDashboard:  # noqa: E501
    """Interactive memory dashboard for monitoring and visualization.  # noqa: E501
  # noqa: E501
    Provides text-based visualizations suitable for terminal output.  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    def __init__(self, sampler: MemorySampler | None = None) -> None:  # noqa: E501
        """Initialize the dashboard.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            sampler: MemorySampler to use (defaults to global singleton)  # noqa: E501
        """  # noqa: E501
        self._sampler = sampler or get_memory_sampler()  # noqa: E501
        self._snapshots: list[MemorySnapshot] = []  # noqa: E501
  # noqa: E501
    def start_monitoring(self, interval: float = 1.0) -> None:  # noqa: E501
        """Start background memory monitoring.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            interval: Sampling interval in seconds  # noqa: E501
        """  # noqa: E501
        self._sampler._interval = interval  # noqa: E501
        self._sampler.start()  # noqa: E501
  # noqa: E501
    def stop_monitoring(self) -> None:  # noqa: E501
        """Stop background memory monitoring."""  # noqa: E501
        self._sampler.stop()  # noqa: E501
  # noqa: E501
    def take_snapshot(self) -> MemorySnapshot:  # noqa: E501
        """Take and store a memory snapshot.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            The captured snapshot  # noqa: E501
        """  # noqa: E501
        snapshot = take_snapshot()  # noqa: E501
        self._snapshots.append(snapshot)  # noqa: E501
        return snapshot  # noqa: E501
  # noqa: E501
    def get_current_status(self) -> dict[str, Any]:  # noqa: E501
        """Get current memory status.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Dictionary with current memory state  # noqa: E501
        """  # noqa: E501
        snapshot = take_snapshot()  # noqa: E501
        return asdict(snapshot)  # noqa: E501
  # noqa: E501
    def render_ascii_chart(self, width: int = 60, height: int = 10) -> str:  # noqa: E501
        """Render an ASCII chart of memory usage over time.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            width: Chart width in characters  # noqa: E501
            height: Chart height in lines  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            ASCII art representation of memory usage  # noqa: E501
        """  # noqa: E501
        samples = self._sampler.get_samples()  # noqa: E501
  # noqa: E501
        if not samples:  # noqa: E501
            return "No memory samples available. Start monitoring first."  # noqa: E501
  # noqa: E501
        # Use last N samples that fit in width  # noqa: E501
        samples = samples[-width:]  # noqa: E501
        rss_values = [s.rss_mb for s in samples]  # noqa: E501
  # noqa: E501
        if not rss_values:  # noqa: E501
            return "No memory data to display."  # noqa: E501
  # noqa: E501
        min_val = min(rss_values)  # noqa: E501
        max_val = max(rss_values)  # noqa: E501
        range_val = max_val - min_val  # noqa: E501
  # noqa: E501
        if range_val == 0:  # noqa: E501
            range_val = 1  # Avoid division by zero  # noqa: E501
  # noqa: E501
        lines = []  # noqa: E501
  # noqa: E501
        # Header  # noqa: E501
        lines.append(f"Memory Usage (RSS MB) - Last {len(samples)} samples")  # noqa: E501
        lines.append(  # noqa: E501
            f"Max: {max_val:.1f} MB | Min: {min_val:.1f} MB | Current: {rss_values[-1]:.1f} MB"  # noqa: E501
        )  # noqa: E501
        lines.append("-" * width)  # noqa: E501
  # noqa: E501
        # Build the chart  # noqa: E501
        chart = [[" " for _ in range(len(rss_values))] for _ in range(height)]  # noqa: E501
  # noqa: E501
        for col, val in enumerate(rss_values):  # noqa: E501
            # Normalize to 0-height range  # noqa: E501
            normalized = (val - min_val) / range_val  # noqa: E501
            row = int(normalized * (height - 1))  # noqa: E501
            chart[height - 1 - row][col] = "*"  # noqa: E501
  # noqa: E501
        # Render with Y-axis labels  # noqa: E501
        for i, chart_row in enumerate(chart):  # noqa: E501
            y_val = max_val - (i / (height - 1)) * range_val  # noqa: E501
            label = f"{y_val:6.1f} |"  # noqa: E501
            lines.append(label + "".join(chart_row))  # noqa: E501
  # noqa: E501
        # X-axis  # noqa: E501
        lines.append(" " * 8 + "-" * len(rss_values))  # noqa: E501
  # noqa: E501
        return "\n".join(lines)  # noqa: E501
  # noqa: E501
    def render_summary(self) -> str:  # noqa: E501
        """Render a text summary of memory status.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Formatted text summary  # noqa: E501
        """  # noqa: E501
        snapshot = take_snapshot()  # noqa: E501
        stats = self._sampler.get_stats()  # noqa: E501
  # noqa: E501
        lines = [  # noqa: E501
            "=" * 50,  # noqa: E501
            "JARVIS Memory Dashboard",  # noqa: E501
            "=" * 50,  # noqa: E501
            "",  # noqa: E501
            "Current Status:",  # noqa: E501
            f"  Process RSS:     {snapshot.process_rss_mb:8.1f} MB",  # noqa: E501
            f"  Process VMS:     {snapshot.process_vms_mb:8.1f} MB",  # noqa: E501
            f"  Metal GPU:       {snapshot.metal_gpu_mb:8.1f} MB",  # noqa: E501
            f"  GC Objects:      {snapshot.gc_objects:8,}",  # noqa: E501
            "",  # noqa: E501
            "System Memory:",  # noqa: E501
            f"  Total:           {snapshot.system_total_gb:8.2f} GB",  # noqa: E501
            f"  Available:       {snapshot.system_available_gb:8.2f} GB",  # noqa: E501
            f"  Used:            {snapshot.system_used_gb:8.2f} GB ({snapshot.system_percent}%)",  # noqa: E501
            "",  # noqa: E501
            "Sampling Statistics:",  # noqa: E501
            f"  Sample Count:    {stats.get('sample_count', 0):8}",  # noqa: E501
            f"  Peak RSS:        {stats.get('peak_rss_mb', 0):8.1f} MB",  # noqa: E501
            f"  Average RSS:     {stats.get('avg_rss_mb', 0):8.1f} MB",  # noqa: E501
            f"  Min RSS:         {stats.get('min_rss_mb', 0):8.1f} MB",  # noqa: E501
            "",  # noqa: E501
            "=" * 50,  # noqa: E501
        ]  # noqa: E501
  # noqa: E501
        return "\n".join(lines)  # noqa: E501
  # noqa: E501
    def export_json(self, filepath: Path | str) -> None:  # noqa: E501
        """Export memory data to JSON file.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            filepath: Path to write JSON data  # noqa: E501
        """  # noqa: E501
        samples = self._sampler.get_samples()  # noqa: E501
        data = {  # noqa: E501
            "exported_at": datetime.now(UTC).isoformat(),  # noqa: E501
            "sample_count": len(samples),  # noqa: E501
            "stats": self._sampler.get_stats(),  # noqa: E501
            "current": asdict(take_snapshot()),  # noqa: E501
            "samples": [  # noqa: E501
                {  # noqa: E501
                    "timestamp": s.timestamp.isoformat(),  # noqa: E501
                    "rss_mb": s.rss_mb,  # noqa: E501
                    "vms_mb": s.vms_mb,  # noqa: E501
                    "available_gb": s.available_gb,  # noqa: E501
                }  # noqa: E501
                for s in samples  # noqa: E501
            ],  # noqa: E501
        }  # noqa: E501
  # noqa: E501
        with open(filepath, "w") as f:  # noqa: E501
            json.dump(data, f, indent=2)  # noqa: E501
  # noqa: E501
    def export_csv(self, filepath: Path | str) -> None:  # noqa: E501
        """Export memory samples to CSV file.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            filepath: Path to write CSV data  # noqa: E501
        """  # noqa: E501
        samples = self._sampler.get_samples()  # noqa: E501
  # noqa: E501
        with open(filepath, "w") as f:  # noqa: E501
            f.write("timestamp,rss_mb,vms_mb,percent,available_gb\n")  # noqa: E501
            for s in samples:  # noqa: E501
                f.write(  # noqa: E501
                    f"{s.timestamp.isoformat()},{s.rss_mb:.2f},{s.vms_mb:.2f},"  # noqa: E501
                    f"{s.percent:.2f},{s.available_gb:.2f}\n"  # noqa: E501
                )  # noqa: E501
  # noqa: E501
  # noqa: E501
def run_memory_watch(duration_seconds: int = 60, interval: float = 1.0) -> dict[str, Any]:  # noqa: E501
    """Run memory monitoring for a specified duration.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        duration_seconds: How long to monitor  # noqa: E501
        interval: Sampling interval  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Dictionary with memory statistics  # noqa: E501
    """  # noqa: E501
    dashboard = MemoryDashboard()  # noqa: E501
    dashboard.start_monitoring(interval=interval)  # noqa: E501
  # noqa: E501
    print(f"Monitoring memory for {duration_seconds} seconds...")  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        time.sleep(duration_seconds)  # noqa: E501
    finally:  # noqa: E501
        dashboard.stop_monitoring()  # noqa: E501
  # noqa: E501
    return dashboard._sampler.get_stats()  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> None:  # noqa: E501
    """Main entry point for dashboard CLI."""  # noqa: E501
    import argparse  # noqa: E501
  # noqa: E501
    parser = argparse.ArgumentParser(description="JARVIS Memory Dashboard")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--duration",  # noqa: E501
        type=int,  # noqa: E501
        default=30,  # noqa: E501
        help="Monitoring duration in seconds",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--interval",  # noqa: E501
        type=float,  # noqa: E501
        default=1.0,  # noqa: E501
        help="Sampling interval in seconds",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--export",  # noqa: E501
        type=str,  # noqa: E501
        help="Export data to file (JSON or CSV based on extension)",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    dashboard = MemoryDashboard()  # noqa: E501
    dashboard.start_monitoring(interval=args.interval)  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        print("Starting memory monitoring. Press Ctrl+C to stop.")  # noqa: E501
        print()  # noqa: E501
  # noqa: E501
        for _ in range(args.duration):  # noqa: E501
            # Clear screen and print summary  # noqa: E501
            print("\033[2J\033[H")  # ANSI clear screen  # noqa: E501
            print(dashboard.render_summary())  # noqa: E501
            print()  # noqa: E501
            print(dashboard.render_ascii_chart())  # noqa: E501
            time.sleep(1)  # noqa: E501
  # noqa: E501
    except KeyboardInterrupt:  # noqa: E501
        print("\nStopping...")  # noqa: E501
    finally:  # noqa: E501
        dashboard.stop_monitoring()  # noqa: E501
  # noqa: E501
        if args.export:  # noqa: E501
            export_path = Path(args.export)  # noqa: E501
            if export_path.suffix == ".csv":  # noqa: E501
                dashboard.export_csv(export_path)  # noqa: E501
            else:  # noqa: E501
                dashboard.export_json(export_path)  # noqa: E501
            print(f"Data exported to: {export_path}")  # noqa: E501
  # noqa: E501
        print("\nFinal Statistics:")  # noqa: E501
        print(dashboard.render_summary())  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    main()  # noqa: E501
