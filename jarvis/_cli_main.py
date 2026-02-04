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
        "memory": "benchmarks.memory.run",
        "latency": "benchmarks.latency.run",
        "hhem": "benchmarks.hallucination.run",
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
        console.print(
            "Available: init, add-contact, list-contacts, extract, build-index, stats, cluster"
        )
        return 1

    if subcommand == "init":
        return _cmd_db_init(args)
    elif subcommand == "add-contact":
        return _cmd_db_add_contact(args)
    elif subcommand == "list-contacts":
        return _cmd_db_list_contacts(args)
    elif subcommand == "extract":
        return _cmd_db_extract(args)
    elif subcommand == "build-index":
        return _cmd_db_build_index(args)
    elif subcommand == "stats":
        return _cmd_db_stats(args)
    elif subcommand == "cluster":
        return _cmd_db_cluster(args)
    else:
        console.print(f"[red]Unknown db subcommand: {subcommand}[/red]")
        return 1


def cmd_ner(args: argparse.Namespace) -> int:
    """Handle NER service management."""
    import os
    import signal
    import subprocess
    from pathlib import Path

    from jarvis.ner_client import PID_FILE, SOCKET_PATH, get_pid, is_service_running

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
        # Start in background
        process = subprocess.Popen(
            [str(python_path), str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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


def _cmd_db_add_contact(args: argparse.Namespace) -> int:
    """Add or update a contact."""
    db = get_db()

    if not db.exists():
        db.init_schema()

    contact = db.add_contact(
        display_name=args.name,
        chat_id=args.chat_id,
        phone_or_email=args.phone,
        relationship=args.relationship,
        style_notes=args.style,
    )

    console.print("[green]Contact added/updated:[/green]")
    console.print(f"  Name: {contact.display_name}")
    if contact.relationship:
        console.print(f"  Relationship: {contact.relationship}")
    if contact.style_notes:
        console.print(f"  Style: {contact.style_notes}")
    if contact.chat_id:
        console.print(f"  Chat ID: {contact.chat_id}")

    return 0


def _cmd_db_list_contacts(args: argparse.Namespace) -> int:
    """List all contacts."""
    db = get_db()

    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    contacts = db.list_contacts(limit=args.limit)

    if not contacts:
        console.print("[dim]No contacts found. Add some with 'jarvis db add-contact'[/dim]")
        return 0

    table = Table(title=f"Contacts ({len(contacts)})")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Relationship")
    table.add_column("Style")
    table.add_column("Chat ID", style="dim")

    for c in contacts:
        table.add_row(
            str(c.id),
            c.display_name,
            c.relationship or "",
            c.style_notes or "",
            c.chat_id or "",
        )

    console.print(table)
    return 0


def _cmd_db_extract(args: argparse.Namespace) -> int:
    """Extract (trigger, response) pairs from iMessage history."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    if not _check_imessage_access():
        console.print(
            "[red]Cannot access iMessage. Grant Full Disk Access in "
            "System Settings > Privacy & Security.[/red]"
        )
        return 1

    db = get_db()
    if not db.exists():
        db.init_schema()
        console.print(f"[dim]Created database at {db.db_path}[/dim]")

    from integrations.imessage import ChatDBReader

    use_v2 = getattr(args, "v2", False)

    if use_v2:
        from jarvis.extract import ExchangeBuilderConfig, extract_all_pairs_v2

        config = ExchangeBuilderConfig(
            time_gap_boundary_minutes=getattr(args, "time_gap", 30.0),
            context_window_size=getattr(args, "context_size", 20),
            max_response_delay_hours=args.max_delay,
        )

        embedder = None
        skip_nli = getattr(args, "skip_nli", False)
        nli_model = None

        try:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            console.print(f"[dim]Loaded embedder for Gate B (backend: {embedder.backend})[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not load embedder: {e}[/yellow]")
            console.print("[yellow]Gate B will be skipped[/yellow]")

        if not skip_nli:
            try:
                from jarvis.validity_gate import load_nli_model

                nli_model = load_nli_model()
                if nli_model:
                    console.print("[dim]Loaded NLI model for Gate C[/dim]")
            except Exception as e:
                console.print(f"[yellow]Could not load NLI model: {e}[/yellow]")

        console.print(
            "[bold]Extracting pairs using v2 pipeline (exchange-based with gates)...[/bold]\n"
        )

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

                stats = extract_all_pairs_v2(
                    reader, db, config, embedder, nli_model, progress_cb, skip_nli
                )

        console.print("\n[bold green]Extraction complete (v2 pipeline)![/bold green]")
        console.print(f"  Messages scanned: {stats['total_messages_scanned']}")
        console.print(f"  Exchanges built: {stats['exchanges_built']}")
        console.print(f"  Conversations processed: {stats['conversations_processed']}")
        console.print(f"  Pairs added: {stats['pairs_added']}")
        console.print(f"  Duplicates skipped: {stats['pairs_skipped_duplicate']}")

        console.print("\n[bold]Validity Gate Results:[/bold]")
        console.print(f"  Gate A rejected: {stats['gate_a_rejected']}")
        console.print(f"  Gate B rejected: {stats['gate_b_rejected']}")
        console.print(f"  Gate C rejected: {stats['gate_c_rejected']}")
        console.print(f"  Final valid: {stats['final_valid']}")
        console.print(f"  Final invalid: {stats['final_invalid']}")
        console.print(f"  Final uncertain: {stats['final_uncertain']}")

        gate_a_reasons = stats.get("gate_a_reasons", {})
        if gate_a_reasons:
            console.print("\n[dim]Gate A rejection reasons:[/dim]")
            for reason, count in sorted(gate_a_reasons.items(), key=lambda x: -x[1]):
                console.print(f"  {reason}: {count}")

    else:
        from jarvis.extract import ExtractionConfig, extract_all_pairs

        config = ExtractionConfig(
            min_trigger_length=args.min_length,
            min_response_length=args.min_length,
            max_response_delay_hours=args.max_delay,
        )

        console.print("[bold]Extracting (trigger, response) pairs from iMessage...[/bold]\n")

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

                stats = extract_all_pairs(reader, db, config, progress_cb)

        console.print("\n[bold green]Extraction complete![/bold green]")
        console.print(f"  Messages scanned: {stats.get('total_messages_scanned', 'N/A')}")
        console.print(f"  Turns identified: {stats.get('turns_identified', 'N/A')}")
        console.print(f"  Conversations processed: {stats['conversations_processed']}")
        console.print(f"  Candidate pairs: {stats.get('candidate_pairs', 'N/A')}")
        console.print(f"  Pairs extracted: {stats['pairs_extracted']}")
        console.print(f"  Pairs added to database: {stats['pairs_added']}")
        console.print(f"  Duplicates skipped: {stats['pairs_skipped_duplicate']}")

        dropped = stats.get("dropped_by_reason", {})
        if dropped and any(v > 0 for v in dropped.values()):
            console.print("\n[dim]Dropped pairs by reason:[/dim]")
            for reason, count in dropped.items():
                if count > 0:
                    console.print(f"  {reason}: {count}")

    if stats.get("errors"):
        console.print(f"\n[yellow]Errors: {len(stats['errors'])}[/yellow]")

    return 0


def _cmd_db_build_index(args: argparse.Namespace) -> int:
    """Build FAISS index of triggers."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from jarvis.index import build_index_from_db

    db = get_db()
    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    pair_count = db.count_pairs()
    if pair_count == 0:
        console.print("[yellow]No pairs found. Run 'jarvis db extract' first.[/yellow]")
        return 1

    console.print(f"[bold]Building FAISS index for {pair_count} triggers...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)

        def progress_cb(stage: str, pct: float, msg: str) -> None:
            progress.update(task, description=msg)

        result = build_index_from_db(db, None, progress_cb)

    if not result["success"]:
        console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
        return 1

    console.print("\n[bold green]Index built successfully![/bold green]")
    console.print(f"  Triggers indexed: {result['pairs_indexed']}")
    console.print(f"  Embedding dimension: {result['dimension']}")
    console.print(f"  Index size: {result['index_size_bytes'] / 1024:.1f} KB")
    console.print(f"  Index path: {result['index_path']}")

    return 0


def _cmd_db_stats(args: argparse.Namespace) -> int:
    """Show database statistics."""
    from jarvis.index import get_index_stats

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
    overview.add_row("Pairs (total)", str(stats["pairs"]))
    overview.add_row("Pairs (quality >= 0.5)", str(stats.get("pairs_quality_gte_50", "N/A")))
    overview.add_row("Embeddings", str(stats["embeddings"]))

    console.print(overview)

    if stats["pairs_per_contact"]:
        console.print("\n[bold]Top Contacts by Pairs:[/bold]")
        for item in stats["pairs_per_contact"][:5]:
            if item["count"] > 0:
                console.print(f"  {item['name']}: {item['count']} pairs")

    if getattr(args, "gate_breakdown", False):
        gate_stats = db.get_gate_stats()
        if gate_stats.get("total_gated", 0) > 0:
            console.print("\n[bold]Validity Gate Breakdown:[/bold]")
            console.print(f"  Total gated pairs: {gate_stats['total_gated']}")
            console.print(f"  Valid: {gate_stats.get('status_valid', 0)}")
            console.print(f"  Invalid: {gate_stats.get('status_invalid', 0)}")
            console.print(f"  Uncertain: {gate_stats.get('status_uncertain', 0)}")

            if gate_stats.get("gate_a_rejected"):
                console.print(f"\n  Gate A rejected: {gate_stats['gate_a_rejected']}")
                if gate_stats.get("gate_a_reasons"):
                    console.print("  [dim]Gate A rejection reasons:[/dim]")
                    for reason, count in sorted(
                        gate_stats["gate_a_reasons"].items(), key=lambda x: -x[1]
                    ):
                        console.print(f"    {reason}: {count}")

            if gate_stats.get("gate_b_bands"):
                console.print("\n  [dim]Gate B bands:[/dim]")
                for band, count in gate_stats["gate_b_bands"].items():
                    console.print(f"    {band}: {count}")

            if gate_stats.get("gate_c_verdicts"):
                console.print("\n  [dim]Gate C verdicts:[/dim]")
                for verdict, count in gate_stats["gate_c_verdicts"].items():
                    console.print(f"    {verdict}: {count}")
        else:
            console.print("\n[dim]No pairs with gate data. Use --v2 extraction.[/dim]")

    index_stats = get_index_stats(db)
    if index_stats and index_stats.get("exists"):
        console.print("\n[bold]FAISS Index:[/bold]")
        console.print(f"  Version: {index_stats.get('version_id', 'N/A')}")
        console.print(f"  Model: {index_stats.get('model_name', 'N/A')}")
        console.print(f"  Vectors: {index_stats['num_vectors']}")
        console.print(f"  Dimension: {index_stats['dimension']}")
        console.print(f"  Size: {index_stats['size_bytes'] / 1024:.1f} KB")
        if index_stats.get("created_at"):
            console.print(f"  Created: {index_stats['created_at']}")
    else:
        console.print("\n[dim]FAISS index not built. Run 'jarvis db build-index'[/dim]")

    return 0


def _cmd_db_cluster(args: argparse.Namespace) -> int:
    """Run cluster analysis on message pairs."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from jarvis.clustering import run_cluster_analysis

    db = get_db()
    if not db.exists():
        console.print("[yellow]Database not initialized. Run 'jarvis db init' first.[/yellow]")
        return 1

    pair_count = db.count_pairs()
    if pair_count == 0:
        console.print("[yellow]No pairs found. Run 'jarvis db extract' first.[/yellow]")
        return 1

    n_clusters = args.n_clusters
    min_cluster_size = args.min_cluster_size

    console.print(f"[bold]Running cluster analysis on {pair_count} pairs...[/bold]")
    console.print(f"  Target clusters: {n_clusters}")
    console.print(f"  Minimum cluster size: {min_cluster_size}\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Computing embeddings and clustering...", total=None)

            stats = run_cluster_analysis(
                n_clusters=n_clusters,
                min_cluster_size=min_cluster_size,
            )

            progress.update(task, completed=True)

        if stats.clusters_created == 0:
            console.print("\n[yellow]No clusters created.[/yellow]")
            console.print(f"  Total pairs analyzed: {stats.total_pairs}")
            console.print("  Try reducing --min-cluster-size or increasing --n-clusters")
            return 0

        console.print("\n[bold green]Cluster analysis complete![/bold green]")
        console.print(f"  Total pairs analyzed: {stats.total_pairs}")
        console.print(f"  Pairs clustered: {stats.pairs_clustered}")
        console.print(f"  Clusters created: {stats.clusters_created}")
        console.print(f"  Noise (unclustered): {stats.noise_count}")
        console.print(f"  Largest cluster: {stats.largest_cluster_size} pairs")
        console.print(f"  Smallest cluster: {stats.smallest_cluster_size} pairs")
        console.print(f"  Average cluster size: {stats.avg_cluster_size:.1f} pairs")

        # Show cluster details
        clusters = db.list_clusters()
        if clusters:
            console.print("\n[bold]Clusters:[/bold]")
            for cluster in clusters[:10]:  # Show top 10
                console.print(f"  • {cluster.name}: {cluster.pair_count} pairs")
            if len(clusters) > 10:
                console.print(f"  ... and {len(clusters) - 10} more")

        return 0
    except Exception as e:
        console.print(f"\n[red]Error during cluster analysis: {e}[/red]")
        logger.exception("Cluster analysis failed")
        return 1


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
        description="Manage the JARVIS database (contacts, pairs, index).",
        formatter_class=HelpFormatter,
    )
    db_subparsers = db_parser.add_subparsers(dest="db_command")

    # db init
    db_init_parser = db_subparsers.add_parser("init", help="initialize the database")
    db_init_parser.add_argument("--force", action="store_true", help="force reinitialization")

    # db add-contact
    db_add_contact_parser = db_subparsers.add_parser("add-contact", help="add or update a contact")
    db_add_contact_parser.add_argument(
        "--name", required=True, metavar="<name>", help="contact name"
    )
    db_add_contact_parser.add_argument("--relationship", metavar="<type>", help="relationship type")
    db_add_contact_parser.add_argument("--style", metavar="<notes>", help="communication style")
    db_add_contact_parser.add_argument("--phone", metavar="<number>", help="phone or email")
    db_add_contact_parser.add_argument(
        "--chat-id", dest="chat_id", metavar="<id>", help="iMessage chat ID"
    )

    # db list-contacts
    db_list_contacts_parser = db_subparsers.add_parser("list-contacts", help="list all contacts")
    db_list_contacts_parser.add_argument("-l", "--limit", type=int, default=100, metavar="<n>")

    # db extract
    db_extract_parser = db_subparsers.add_parser("extract", help="extract pairs from iMessage")
    db_extract_parser.add_argument(
        "--min-length", dest="min_length", type=int, default=2, metavar="<n>"
    )
    db_extract_parser.add_argument(
        "--max-delay", dest="max_delay", type=float, default=1.0, metavar="<hours>"
    )
    db_extract_parser.add_argument(
        "--v2", action="store_true", help="use v2 extraction with validity gates"
    )
    db_extract_parser.add_argument(
        "--time-gap", dest="time_gap", type=float, default=30.0, metavar="<min>"
    )
    db_extract_parser.add_argument(
        "--context-size", dest="context_size", type=int, default=20, metavar="<n>"
    )
    db_extract_parser.add_argument("--skip-nli", dest="skip_nli", action="store_true")

    # db build-index
    db_subparsers.add_parser("build-index", help="build FAISS index")

    # db stats
    db_stats_parser = db_subparsers.add_parser("stats", help="show database statistics")
    db_stats_parser.add_argument("--gate-breakdown", dest="gate_breakdown", action="store_true")

    # db cluster
    db_cluster_parser = db_subparsers.add_parser(
        "cluster", help="run cluster analysis on message pairs (experimental)"
    )
    db_cluster_parser.add_argument(
        "--n-clusters",
        dest="n_clusters",
        type=int,
        default=5,
        metavar="<n>",
        help="number of clusters to create (default: 5)",
    )
    db_cluster_parser.add_argument(
        "--min-cluster-size",
        dest="min_cluster_size",
        type=int,
        default=10,
        metavar="<n>",
        help="minimum messages per cluster (default: 10)",
    )

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
