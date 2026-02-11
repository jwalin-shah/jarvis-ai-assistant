"""Unified CLI for JARVIS developer tools.

Uses Typer for modern, type-safe CLI with rich help output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jarvis.tools.registry import ToolRegistry, list_commands

# Rich console for pretty output
console = Console()

# Main app
app = typer.Typer(
    name="jarvis-tools",
    help="JARVIS developer tools for training, evaluation, and data processing.",
    rich_markup_mode="rich",
    add_completion=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    list_tools: bool = typer.Option(False, "--list", "-l", help="List all tools"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """JARVIS developer tools - unified CLI for ML training and data operations."""
    # Discover tools on startup
    ToolRegistry.discover()

    if version:
        from jarvis.tools import __version__

        console.print(f"jarvis-tools version {__version__}")
        raise typer.Exit()

    if list_tools:
        _list_tools_table()
        raise typer.Exit()

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


def _list_tools_table() -> None:
    """Display table of available tools."""
    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Version", style="dim")

    for name, desc, version in list_commands():
        table.add_row(name, desc, version)

    console.print(table)
    console.print("\nUse [bold]jarvis-tools <name> --help[/bold] for details.")


# Subcommand groups
train_app = typer.Typer(help="Model training commands")
eval_app = typer.Typer(help="Evaluation commands")
data_app = typer.Typer(help="Data pipeline commands")
label_app = typer.Typer(help="Labeling commands")
db_app = typer.Typer(help="Database commands")
check_app = typer.Typer(help="Validation and checking commands")
util_app = typer.Typer(help="Utility commands")

app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(label_app, name="label")
app.add_typer(db_app, name="db")
app.add_typer(check_app, name="check")
app.add_typer(util_app, name="util")


# ============================================================================
# TRAIN COMMANDS
# ============================================================================


@train_app.command("category")
def train_category(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing train.npz, test.npz, metadata.json",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: auto from config)",
    ),
    label_map: str = typer.Option(
        "4class",
        "--label-map",
        help="Label mapping: 4class (native) or 3class (merge directive+commissive)",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without training",
    ),
) -> None:
    """Train category classifier (LinearSVC)."""
    from jarvis.tools.commands.train import CategoryTrainer

    tool = CategoryTrainer(
        {
            "data_dir": data_dir,
            "output": output,
            "label_map": label_map,
            "seed": seed,
        }
    )

    if dry_run:
        result = tool.dry_run()
        _display_result(result)
        return

    with console.status("[bold green]Training category classifier..."):
        result = tool.run()

    _display_result(result)
    if not result:
        raise typer.Exit(1)


# ============================================================================
# EVAL COMMANDS
# ============================================================================


@eval_app.command("classifiers")
def eval_classifiers(
    stages: list[str] = typer.Option(
        ["mobilization", "category", "replyability"],
        "--stages",
        help="Stages to evaluate",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output JSON file for results",
    ),
) -> None:
    """Evaluate classifier stages."""
    console.print(f"Evaluating stages: {stages}")
    console.print("[yellow]Not yet implemented - use scripts/eval_classifiers.py[/yellow]")


# ============================================================================
# DATA COMMANDS
# ============================================================================


@data_app.command("build")
def data_build(
    dataset_type: str = typer.Argument(..., help="Dataset type: eval, goldset, train"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Limit number of records"),
) -> None:
    """Build evaluation or training datasets."""
    console.print(f"Building {dataset_type} dataset...")
    console.print("[yellow]Not yet implemented - use scripts/build_*.py[/yellow]")


@data_app.command("merge")
def data_merge(
    inputs: list[Path] = typer.Argument(..., help="Input files to merge"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file"),
    strategy: str = typer.Option(
        "concat", "--strategy", help="Merge strategy: concat, dedup, intersect"
    ),
) -> None:
    """Merge multiple datasets."""
    console.print(f"Merging {len(inputs)} files...")
    console.print("[yellow]Not yet implemented - use scripts/merge_goldsets.py[/yellow]")


# ============================================================================
# DB COMMANDS
# ============================================================================


@db_app.command("maintain")
def db_maintain(
    full: bool = typer.Option(
        False, "--full", help="Run full maintenance including VACUUM and backup"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
) -> None:
    """Run database maintenance tasks."""
    console.print("Running database maintenance...")
    console.print("[yellow]Not yet implemented - use scripts/db_maintenance.py[/yellow]")


@db_app.command("health")
def db_health(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Check database health status."""
    console.print("Checking database health...")
    console.print("[yellow]Not yet implemented - use scripts/db_maintenance.py --health[/yellow]")


# ============================================================================
# CHECK COMMANDS
# ============================================================================


@check_app.command("prompt")
def check_prompt(
    compare: bool = typer.Option(False, "--compare", help="Compare with previous version"),
) -> None:
    """Check prompt versions and drift."""
    console.print("Checking prompt versions...")
    console.print("[yellow]Not yet implemented - use scripts/check_prompt_version.py[/yellow]")


@check_app.command("contracts")
def check_contracts(
    tier: int = typer.Option(1, "--tier", help="Validation tier (1=critical, 4=all)"),
) -> None:
    """Validate contracts against implementations."""
    console.print("Validating contracts...")
    console.print("[yellow]Not yet implemented - use scripts/verify_contracts.py[/yellow]")


# ============================================================================
# UTIL COMMANDS
# ============================================================================


@util_app.command("codeowners")
def util_codeowners(
    path: Path | None = typer.Argument(None, help="Path to check ownership"),
) -> None:
    """Show code ownership information."""
    console.print("Code ownership...")
    console.print("[yellow]Not yet implemented - use scripts/code_owner.py[/yellow]")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _display_result(result) -> None:
    """Display a ToolResult with Rich formatting."""
    if result.success:
        console.print(
            Panel(
                f"[green]{result.message}[/green]",
                title="Success",
                border_style="green",
            )
        )

        if result.metrics:
            table = Table(title="Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
            console.print(table)

        if result.artifacts:
            console.print("\n[bold]Artifacts:[/bold]")
            for artifact in result.artifacts:
                console.print(f"  • {artifact}")
    else:
        console.print(
            Panel(
                f"[red]{result.message}[/red]",
                title="Error",
                border_style="red",
            )
        )

        if result.data and "errors" in result.data:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in result.data["errors"]:
                console.print(f"  • {error}")


def run() -> int:
    """Entry point for console script."""
    try:
        app()
        return 0
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(run())
