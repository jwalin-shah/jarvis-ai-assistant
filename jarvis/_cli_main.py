#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""JARVIS CLI - Developer tools for the JARVIS AI assistant.

Provides commands for server management, health monitoring, benchmarking,
and database operations. User-facing features are in the desktop app.

Usage:
    jarvis serve                     Start the API server for desktop app
    jarvis health                    Show system health status
    jarvis benchmark memory          Run performance benchmarks
    jarvis db stats                  Database operations

For comprehensive documentation, see docs/CLI_GUIDE.md

Shell Completion:
    To enable shell completion, install argcomplete and run:

    Bash:  eval "$(register-python-argcomplete jarvis)"
    Zsh:   eval "$(register-python-argcomplete jarvis)"
    Fish:  register-python-argcomplete --shell fish jarvis | source
"""

import argparse
import logging
import sys
from typing import NoReturn

# Optional argcomplete support for shell completion
try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from contracts.health import FeatureState
from core.health import get_degradation_controller, reset_degradation_controller
from core.memory import get_memory_controller, reset_memory_controller
from jarvis.db import get_db
from jarvis.errors import JarvisError
from jarvis.system import (
    FEATURE_CHAT,
    FEATURE_IMESSAGE,
    _check_imessage_access,
    initialize_system,
)

console = Console()
logger = logging.getLogger(__name__)


def _format_jarvis_error(error: JarvisError) -> None:
    """Format and display a JARVIS error with helpful suggestions."""
    from jarvis.errors import (
        ConfigurationError,
        ModelError,
        ResourceError,
        iMessageAccessError,
        iMessageError,
    )

    console.print(f"[red]Error: {error.message}[/red]")

    if isinstance(error, iMessageAccessError):
        if error.details.get("requires_permission"):
            instructions = error.details.get("permission_instructions", [])
            if instructions:
                console.print("\n[yellow]To fix this:[/yellow]")
                for i, instruction in enumerate(instructions, 1):
                    console.print(f"  {i}. {instruction}")
        else:
            console.print(
                "[yellow]Grant Full Disk Access in System Settings > Privacy & Security.[/yellow]"
            )
    elif isinstance(error, iMessageError):
        console.print("[yellow]Check that iMessage is accessible and try again.[/yellow]")
    elif isinstance(error, ModelError):
        if error.details.get("available_mb") and error.details.get("required_mb"):
            console.print(
                f"[yellow]Available: {error.details['available_mb']} MB, "
                f"Required: {error.details['required_mb']} MB[/yellow]"
            )
        console.print("[yellow]Try closing other applications to free memory.[/yellow]")
    elif isinstance(error, ResourceError):
        if error.details.get("resource_type") == "memory":
            console.print("[yellow]Close other applications to free up memory.[/yellow]")
        elif error.details.get("resource_type") == "disk":
            console.print("[yellow]Free up disk space and try again.[/yellow]")
    elif isinstance(error, ConfigurationError):
        if error.details.get("config_path"):
            console.print(f"[yellow]Config file: {error.details['config_path']}[/yellow]")
        console.print("[yellow]Try running 'jarvis health' to diagnose the issue.[/yellow]")

    logger.debug(
        "JarvisError details - code=%s, details=%s, cause=%s",
        error.code.value,
        error.details,
        error.cause,
    )


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


# =============================================================================
# Commands
# =============================================================================


def cmd_health(args: argparse.Namespace) -> int:
    """Display system health status."""
    console.print(Panel("[bold]JARVIS System Health[/bold]", title="Health Check"))

    # Memory status
    mem_controller = get_memory_controller()
    state = mem_controller.get_state()

    mem_table = Table(title="Memory Status")
    mem_table.add_column("Metric", style="bold")
    mem_table.add_column("Value")

    mem_table.add_row("Available Memory", f"{state.available_mb:.0f} MB")
    mem_table.add_row("Used Memory", f"{state.used_mb:.0f} MB")
    mem_table.add_row("Operating Mode", state.current_mode.value)
    mem_table.add_row("Pressure Level", state.pressure_level)
    mem_table.add_row("Model Loaded", "Yes" if state.model_loaded else "No")

    console.print(mem_table)
    console.print()

    # Feature health
    deg_controller = get_degradation_controller()
    health = deg_controller.get_health()

    feature_table = Table(title="Feature Status")
    feature_table.add_column("Feature", style="bold")
    feature_table.add_column("Status")
    feature_table.add_column("Details")

    status_colors = {
        FeatureState.HEALTHY: "green",
        FeatureState.DEGRADED: "yellow",
        FeatureState.FAILED: "red",
    }

    for feature_name, feature_state in health.items():
        color = status_colors.get(feature_state, "white")
        status = Text(feature_state.value, style=color)

        details = ""
        if feature_name == FEATURE_IMESSAGE:
            details = "Full Disk Access required" if not _check_imessage_access() else "OK"
        elif feature_name == FEATURE_CHAT:
            details = "OK"

        feature_table.add_row(feature_name, status, details)

    console.print(feature_table)
    console.print()

    # Model status
    try:
        from models import get_generator

        generator = get_generator()
        model_loaded = generator.is_loaded()

        model_table = Table(title="Model Status")
        model_table.add_column("Metric", style="bold")
        model_table.add_column("Value")

        model_table.add_row("Loaded", "Yes" if model_loaded else "No")
        if model_loaded:
            model_table.add_row("Memory Usage", f"{generator.get_memory_usage_mb():.0f} MB")

        console.print(model_table)
    except Exception as e:
        console.print(f"[yellow]Model status unavailable: {e}[/yellow]")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    benchmark_type = args.type
    output_file = args.output

    console.print(f"[bold]Running {benchmark_type} benchmark...[/bold]\n")

    benchmark_modules = {
        "memory": "evals.benchmarks.memory.run",
        "latency": "evals.benchmarks.latency.run",
        "hhem": "evals.benchmarks.hallucination.run",
    }

    if benchmark_type not in benchmark_modules:
        console.print(f"[red]Unknown benchmark type: {benchmark_type}[/red]")
        console.print(f"Available types: {', '.join(benchmark_modules.keys())}")
        return 1

    import subprocess

    cmd = [sys.executable, "-m", benchmark_modules[benchmark_type]]
    if output_file:
        cmd.extend(["--output", output_file])

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        console.print(f"[red]Error running benchmark: {e}[/red]")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Display version information."""
    from jarvis import __version__

    console.print(f"JARVIS AI Assistant v{__version__}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the API server."""
    import uvicorn

    host = args.host
    port = args.port
    reload = args.reload

    console.print(
        Panel(
            f"[bold green]Starting JARVIS API Server[/bold green]\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Reload: {'Enabled' if reload else 'Disabled'}",
            title="API Server",
        )
    )

    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
        return 0
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        return 1


def cmd_db(args: argparse.Namespace) -> int:
    """Handle database operations."""
    subcommand = args.db_command
    if subcommand is None:
        console.print("[red]Error: Please specify a db subcommand[/red]")
        console.print("Available: init, extract, stats, build-profiles, feedback, optimize")
        return 1

    if subcommand == "init":
        return _cmd_db_init(args)
    elif subcommand == "extract":
        return _cmd_db_extract(args)
    elif subcommand == "stats":
        return _cmd_db_stats(args)
    elif subcommand == "build-profiles":
        return _cmd_db_build_profiles(args)
    elif subcommand == "feedback":
        return _cmd_feedback(args)
    elif subcommand == "optimize":
        return _cmd_db_optimize(args)
    else:
        console.print(f"[red]Unknown db subcommand: {subcommand}[/red]")
        return 1


def _cmd_feedback(args: argparse.Namespace) -> int:
    """Display user feedback statistics."""
    from jarvis.eval.evaluation import get_feedback_store

    store = get_feedback_store()
    stats = store.get_stats()

    console.print(Panel("[bold]JARVIS User Feedback Statistics[/bold]", title="Feedback"))

    if not stats or stats.get("total_entries", 0) == 0:
        console.print("[yellow]No feedback recorded yet.[/yellow]")
        return 0

    table = Table(title="Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total Entries", str(stats["total_entries"]))
    table.add_row("Acceptance Rate", f"{stats.get('acceptance_rate', 0) * 100:.1f}%")
    table.add_row("Edit Rate", f"{stats.get('edit_rate', 0) * 100:.1f}%")

    console.print(table)

    # Actions breakdown
    actions_table = Table(title="Actions Breakdown")
    actions_table.add_column("Action", style="bold")
    actions_table.add_column("Count")

    actions = stats.get("actions", {})
    for action, count in actions.items():
        actions_table.add_row(action, str(count))

    console.print("\n", actions_table)

    if args.limit > 0:
        entries = store.list_feedback(limit=args.limit)
        if entries:
            console.print("\n[bold]Recent Feedback Entries:[/bold]")
            entries_table = Table()
            entries_table.add_column("Time", style="dim")
            entries_table.add_column("Action")
            entries_table.add_column("Suggestion")
            entries_table.add_column("Edited Text")

            for entry in entries:
                entries_table.add_row(
                    entry.timestamp.strftime("%Y-%m-%d %H:%M"),
                    entry.action.value,
                    entry.suggestion_text[:50] + "..."
                    if len(entry.suggestion_text) > 50
                    else entry.suggestion_text,
                    (entry.edited_text[:50] + "...")
                    if entry.edited_text and len(entry.edited_text) > 50
                    else (entry.edited_text or ""),
                )
            console.print(entries_table)

    return 0


def cmd_ner(args: argparse.Namespace) -> int:
    """Handle NER service management."""
    import os
    import signal
    import subprocess
    from pathlib import Path

    from jarvis.nlp.ner_client import PID_FILE, SOCKET_PATH, get_pid, is_service_running

    subcommand = args.ner_command
    if subcommand is None:
        console.print("[red]Error: Please specify a ner subcommand[/red]")
        console.print("Available: start, stop, status, setup")
        return 1

    ner_venv = Path.home() / ".jarvis" / "ner_venv"
    setup_script = Path(__file__).parent.parent / "scripts" / "setup_ner_venv.sh"
    server_script = Path(__file__).parent.parent / "scripts" / "ner_server.py"

    if subcommand == "setup":
        if not setup_script.exists():
            console.print(f"[red]Setup script not found: {setup_script}[/red]")
            return 1
        console.print("[bold]Setting up NER environment...[/bold]")
        result = subprocess.run(["bash", str(setup_script)], check=False)
        return result.returncode

    if subcommand == "status":
        if is_service_running():
            pid = get_pid()
            console.print(f"[green]NER service is running (PID: {pid})[/green]")
            console.print(f"Socket: {SOCKET_PATH}")
        else:
            console.print("[yellow]NER service is not running[/yellow]")
            if not ner_venv.exists():
                console.print("[dim]Run 'jarvis ner setup' to install the NER environment[/dim]")
            else:
                console.print("[dim]Run 'jarvis ner start' to start the service[/dim]")
        return 0

    if subcommand == "start":
        if is_service_running():
            pid = get_pid()
            console.print(f"[yellow]NER service is already running (PID: {pid})[/yellow]")
            return 0

        if not ner_venv.exists():
            console.print("[red]NER environment not found[/red]")
            console.print("Run 'jarvis ner setup' first")
            return 1

        python_path = ner_venv / "bin" / "python"
        if not python_path.exists():
            console.print(f"[red]Python not found in NER venv: {python_path}[/red]")
            return 1

        if not server_script.exists():
            console.print(f"[red]NER server script not found: {server_script}[/red]")
            return 1

        console.print("[bold]Starting NER service...[/bold]")
        # Start in background, logging stderr to file for debugging
        ner_log_path = Path.home() / ".jarvis" / "ner_server.log"
        ner_log_path.parent.mkdir(parents=True, exist_ok=True)
        ner_log_file = open(ner_log_path, "a")  # noqa: SIM115
        process = subprocess.Popen(
            [str(python_path), str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=ner_log_file,
            start_new_session=True,
        )

        # Wait briefly and check if it started
        import time

        time.sleep(1.0)

        if is_service_running():
            console.print(f"[green]NER service started (PID: {process.pid})[/green]")
            console.print(f"Socket: {SOCKET_PATH}")
            return 0
        else:
            console.print("[red]Failed to start NER service[/red]")
            console.print("Check logs or run server manually for debugging")
            return 1

    if subcommand == "stop":
        pid = get_pid()
        if pid is None:
            console.print("[yellow]NER service is not running[/yellow]")
            return 0

        console.print(f"[bold]Stopping NER service (PID: {pid})...[/bold]")
        try:
            os.kill(pid, signal.SIGTERM)
            console.print("[green]NER service stopped[/green]")
        except ProcessLookupError:
            console.print("[yellow]Process already terminated[/yellow]")
        except PermissionError:
            console.print(f"[red]Permission denied stopping PID {pid}[/red]")
            return 1

        # Clean up PID file
        if PID_FILE.exists():
            PID_FILE.unlink()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        return 0

    console.print(f"[red]Unknown ner subcommand: {subcommand}[/red]")
    return 1


def _cmd_db_init(args: argparse.Namespace) -> int:
    """Initialize the JARVIS database."""
    console.print("[bold]Initializing JARVIS database...[/bold]")

    db = get_db()

    if db.exists() and not args.force:
        console.print(f"[yellow]Database already exists at {db.db_path}[/yellow]")
        console.print("Use --force to reinitialize")
        return 0

    created = db.init_schema()

    if created:
        console.print(f"[green]Database created at {db.db_path}[/green]")
    else:
        console.print(f"[green]Database already up to date at {db.db_path}[/green]")

    return 0


def _cmd_db_extract(args: argparse.Namespace) -> int:
    """Extract topic segments from iMessage history into vec_chunks."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    db = get_db()
    created = db.init_schema()
    if created:
        console.print(f"[dim]Initialized database at {db.db_path}[/dim]")

    from integrations.imessage import ChatDBReader
    from jarvis.search.segment_ingest import ingest_and_extract_segments
    from jarvis.search.vec_search import get_vec_searcher
    from jarvis.topics.segment_labeler import reset_labeler
    from jarvis.topics.topic_segmenter import reset_segmenter

    # Reset singletons to force reload of refined logic/config
    reset_segmenter()
    reset_labeler()

    vec_searcher = get_vec_searcher(db)

    console.print("[bold]Segmenting conversations and indexing chunks...[/bold]\n")

    with ChatDBReader() as reader:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing conversations...", total=None)

            def progress_cb(current: int, total: int, chat_id: str) -> None:
                progress.update(
                    task,
                    description=f"Processing {current}/{total}: {chat_id[:30]}...",
                )

            stats = ingest_and_extract_segments(
                reader, db, vec_searcher, progress_cb, limit=args.limit, force=args.force
            )

    console.print("\n[bold green]Extraction complete![/bold green]")
    console.print(f"  Messages scanned: {stats['total_messages_scanned']}")
    console.print(f"  Conversations processed: {stats['conversations_processed']}")
    console.print(f"  Segments created: {stats['segments_created']}")
    console.print(f"  Segments indexed: {stats['segments_indexed']}")
    console.print(f"  Skipped (no response): {stats['segments_skipped_no_response']}")
    console.print(f"  Skipped (too short): {stats['segments_skipped_too_short']}")

    if stats.get("errors"):
        console.print(f"\n[yellow]Errors: {len(stats['errors'])}[/yellow]")

    # Optimize after bulk extraction
    db.optimize()
    console.print("[dim]Database optimized.[/dim]")

    return 0


def _cmd_db_optimize(args: argparse.Namespace) -> int:
    """Optimize the JARVIS database."""
    console.print("[bold]Optimizing JARVIS database...[/bold]")
    db = get_db()
    db.optimize()
    console.print("[green]Optimization complete (REINDEX, VACUUM).[/green]")
    return 0


def _cmd_db_build_profiles(args: argparse.Namespace) -> int:
    """Build contact profiles from conversation history."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    from integrations.imessage import ChatDBReader
    from jarvis.contacts.contact_profile import (
        ContactProfileBuilder,
        load_profile,
        save_profile,
        update_preference_tables,
    )

    builder = ContactProfileBuilder()
    built = 0
    skipped_existing = 0
    skipped_insufficient = 0
    errors = 0

    console.print("[bold]Building contact profiles from conversation history...[/bold]\n")

    with ChatDBReader() as reader:
        conversations = reader.get_conversations(limit=args.limit)
        console.print(f"Found {len(conversations)} conversations\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=None)

            for i, conv in enumerate(conversations):
                display = conv.display_name or ", ".join(conv.participants) or conv.chat_id
                progress.update(
                    task,
                    description=f"[{i + 1}/{len(conversations)}] {display[:40]}",
                )

                # Skip if profile exists and --force not set
                if not args.force and load_profile(conv.chat_id) is not None:
                    skipped_existing += 1
                    continue

                try:
                    messages = reader.get_messages(conv.chat_id, limit=10000)
                    if len(messages) < builder.min_messages:
                        skipped_insufficient += 1
                        continue

                    profile = builder.build_profile(
                        contact_id=conv.chat_id,
                        messages=messages,
                        contact_name=conv.display_name,
                    )
                    save_profile(profile)

                    # Update SQL preference tables
                    update_preference_tables(profile, messages)

                    built += 1
                except Exception as e:
                    logger.warning("Error building profile for %s: %s", conv.chat_id, e)
                    errors += 1
                except Exception as e:
                    logger.warning("Error building profile for %s: %s", conv.chat_id, e)
                    errors += 1

    console.print("\n[bold green]Profile building complete![/bold green]")
    console.print(f"  Built: {built}")
    console.print(f"  Skipped (existing): {skipped_existing}")
    console.print(f"  Skipped (insufficient messages): {skipped_insufficient}")
    if errors:
        console.print(f"  [yellow]Errors: {errors}[/yellow]")

    return 0


def _cmd_db_stats(args: argparse.Namespace) -> int:
    """Show database statistics."""
    db = get_db()

    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    stats = db.get_stats()

    console.print(Panel("[bold]JARVIS Database Statistics[/bold]", title="Stats"))

    overview = Table(title="Overview")
    overview.add_column("Metric", style="bold")
    overview.add_column("Count")

    overview.add_row("Contacts", str(stats["contacts"]))
    overview.add_row("Chunks (topic segments)", str(stats.get("chunks", 0)))

    console.print(overview)

    chunks_per_contact = stats.get("chunks_per_contact", [])
    if chunks_per_contact:
        console.print("\n[bold]Top Contacts by Chunks:[/bold]")
        for item in chunks_per_contact[:5]:
            if item["count"] > 0:
                console.print(f"  {item['name']}: {item['count']} chunks")

    try:
        with db.connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM vec_chunks")
            row = cursor.fetchone()
            vec_count = row["cnt"] if row else 0
        if vec_count > 0:
            console.print(f"\n[bold]Vec Index:[/bold] {vec_count} vectors")
        else:
            console.print("\n[dim]Vec index empty. Run 'jarvis db build-index'[/dim]")
    except Exception:
        console.print("\n[dim]Vec index not available.[/dim]")

    return 0


# =============================================================================
# Argument Parser
# =============================================================================


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter with better defaults."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=30, width=100)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all commands."""
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS AI Assistant - Developer Tools",
        formatter_class=HelpFormatter,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging")
    parser.add_argument("--version", action="store_true", help="show version and exit")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="show system health status",
        description="Display comprehensive system health information.",
        formatter_class=HelpFormatter,
    )
    health_parser.set_defaults(func=cmd_health)

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="run performance benchmarks",
        description="Run various performance benchmarks (memory, latency, hhem).",
        formatter_class=HelpFormatter,
    )
    bench_parser.add_argument(
        "type",
        choices=["memory", "latency", "hhem"],
        metavar="<type>",
        help="benchmark type: memory, latency, or hhem",
    )
    bench_parser.add_argument(
        "-o", "--output", metavar="<file>", help="output file for results (JSON)"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="show version information",
        formatter_class=HelpFormatter,
    )
    version_parser.set_defaults(func=cmd_version)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="start the API server",
        description="Start the FastAPI REST server for the desktop app.",
        formatter_class=HelpFormatter,
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", metavar="<addr>", help="host address (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "-p", "--port", type=int, default=8000, metavar="<port>", help="port (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="enable auto-reload for development"
    )
    serve_parser.set_defaults(func=cmd_serve)

    # Database command
    db_parser = subparsers.add_parser(
        "db",
        help="manage JARVIS database",
        description="Manage the JARVIS database (contacts, chunks, index).",
        formatter_class=HelpFormatter,
    )
    db_subparsers = db_parser.add_subparsers(dest="db_command")

    # db init
    db_init_parser = db_subparsers.add_parser("init", help="initialize the database")
    db_init_parser.add_argument("--force", action="store_true", help="force reinitialization")

    # db extract
    db_extract_parser = db_subparsers.add_parser(
        "extract", help="segment conversations and index chunks"
    )
    db_extract_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        metavar="<n>",
        help="max conversations to process",
    )
    db_extract_parser.add_argument(
        "--force", action="store_true", help="clear existing data and re-extract everything"
    )

    # db stats
    db_stats_parser = db_subparsers.add_parser("stats", help="show database statistics")
    db_stats_parser.add_argument("--gate-breakdown", dest="gate_breakdown", action="store_true")

    # db build-profiles
    db_build_profiles_parser = db_subparsers.add_parser(
        "build-profiles", help="build contact profiles from conversation history"
    )
    db_build_profiles_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        metavar="<n>",
        help="max conversations to process",
    )
    db_build_profiles_parser.add_argument(
        "--force", action="store_true", help="rebuild existing profiles"
    )

    # db feedback
    db_feedback_parser = db_subparsers.add_parser("feedback", help="show user feedback statistics")
    db_feedback_parser.add_argument(
        "-l", "--limit", type=int, default=5, help="number of recent entries to show"
    )

    # db optimize
    db_subparsers.add_parser("optimize", help="optimize the database (REINDEX, VACUUM)")

    db_parser.set_defaults(func=cmd_db)

    # ner command - manage NER service
    ner_parser = subparsers.add_parser(
        "ner",
        help="manage the NER (named entity recognition) service",
        description="Manage the spaCy NER service for entity extraction.",
    )
    ner_subparsers = ner_parser.add_subparsers(dest="ner_command")

    ner_subparsers.add_parser("start", help="start the NER service")
    ner_subparsers.add_parser("stop", help="stop the NER service")
    ner_subparsers.add_parser("status", help="check NER service status")
    ner_subparsers.add_parser("setup", help="set up NER environment (install spaCy)")

    ner_parser.set_defaults(func=cmd_ner)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()

    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)

    if args.version:
        return cmd_version(args)

    setup_logging(args.verbose)

    success, warnings = initialize_system()
    if not success:
        console.print("[red]Failed to initialize JARVIS system.[/red]")
        console.print("\n[dim]Troubleshooting tips:[/dim]")
        console.print("  1. Run 'python -m jarvis.setup' to validate environment")
        console.print("  2. Check system requirements (Apple Silicon, Python 3.11+)")
        console.print("  3. Run 'jarvis -v <command>' for debug logging")
        return 1

    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")

    if args.command is None:
        parser.print_help()
        return 0

    result: int = args.func(args)
    return result


def cleanup() -> None:
    """Clean up system resources."""
    try:
        reset_memory_controller()
        reset_degradation_controller()
    except Exception as e:
        logger.debug("Error resetting controllers during cleanup: %s", e)

    try:
        from models import reset_generator

        reset_generator()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Error resetting generator during cleanup: %s", e)


def run() -> NoReturn:
    """Entry point that handles cleanup and exit."""
    try:
        exit_code = main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        exit_code = 130
    except JarvisError as e:
        _format_jarvis_error(e)
        logger.exception("JARVIS error")
        exit_code = 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Unexpected error")
        exit_code = 1
    finally:
        cleanup()

    sys.exit(exit_code)


if __name__ == "__main__":
    run()
