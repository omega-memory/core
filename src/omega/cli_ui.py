"""OMEGA CLI UI — Rich rendering helpers with graceful plain-text fallback."""

import os
from typing import Any, Dict, Optional, Sequence, Tuple

# Graceful import — fall back to plain text if Rich unavailable or NO_COLOR set
try:
    if os.environ.get("NO_COLOR"):
        raise ImportError("NO_COLOR")
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None  # type: ignore[assignment]


def print_header(title: str) -> None:
    """Print a styled header (Rich Panel or plain === title ===)."""
    if RICH_AVAILABLE:
        console.print(Panel(title, style="bold cyan", expand=False))
    else:
        print(f"\n=== {title} ===\n")


def print_section(title: str) -> None:
    """Print a section separator."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold]{title}[/bold]")
        console.print("─" * min(len(title) + 4, 60), style="dim")
    else:
        print(f"\n--- {title} ---")


def print_kv(pairs: Sequence[Tuple[str, Any]], indent: int = 2) -> None:
    """Print key-value pairs with colored keys or plain text."""
    prefix = " " * indent
    if RICH_AVAILABLE:
        for key, value in pairs:
            console.print(f"{prefix}[bold cyan]{key}:[/bold cyan] {value}")
    else:
        for key, value in pairs:
            print(f"{prefix}{key}: {value}")


def print_table(
    title: Optional[str],
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    styles: Optional[Sequence[Optional[str]]] = None,
) -> None:
    """Print a formatted table (Rich Table or aligned plain text)."""
    if RICH_AVAILABLE:
        table = Table(title=title, show_lines=False, pad_edge=True)
        for i, col in enumerate(columns):
            style = styles[i] if styles and i < len(styles) else None
            table.add_column(col, style=style)
        for row in rows:
            table.add_row(*(str(cell) for cell in row))
        console.print(table)
    else:
        if title:
            print(f"\n{title}")
        if not rows:
            print("  (empty)")
            return
        # Calculate column widths
        widths = [len(str(c)) for c in columns]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        # Header
        header = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns))
        print(f"  {header}")
        print(f"  {'  '.join('-' * w for w in widths)}")
        for row in rows:
            line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row) if i < len(widths))
            print(f"  {line}")


def print_bar_chart(
    items: Sequence[Tuple[str, int]],
    title: Optional[str] = None,
    total: Optional[int] = None,
) -> None:
    """Print a horizontal bar chart with colored blocks or ASCII #."""
    if total is None:
        total = sum(count for _, count in items)
    if total == 0:
        if title:
            print(f"  {title}: (no data)")
        return

    if RICH_AVAILABLE:
        table = Table(title=title, show_lines=False, pad_edge=True, show_header=True)
        table.add_column("Type", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("%", justify="right")
        table.add_column("", min_width=25)

        colors = ["cyan", "green", "yellow", "magenta", "blue", "red", "white"]
        for i, (label, count) in enumerate(items):
            pct = count / total * 100
            bar_len = int(pct / 2)
            color = colors[i % len(colors)]
            bar = Text("█" * bar_len, style=color)
            table.add_row(label, str(count), f"{pct:.1f}%", bar)
        console.print(table)
    else:
        if title:
            print(f"\n{title}")
        for label, count in items:
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"  {label:<20} {count:>5}  {pct:5.1f}%  {bar}")


_STATUS_SYMBOLS: Dict[str, Tuple[str, str]] = {
    "ok": ("  [bold green]✓[/bold green]", "  [OK]"),
    "fail": ("  [bold red]✗[/bold red]", "  [FAIL]"),
    "warn": ("  [bold yellow]![/bold yellow]", "  [WARN]"),
}


def print_status_line(status: str, msg: str) -> None:
    """Print a status line: green check / red X / yellow warning, or plain [OK]/[FAIL]/[WARN]."""
    rich_sym, plain_sym = _STATUS_SYMBOLS.get(status, ("  ?", "  [?]"))
    if RICH_AVAILABLE:
        console.print(f"{rich_sym} {msg}")
    else:
        print(f"{plain_sym} {msg}")


def print_summary(errors: int, warnings: int) -> None:
    """Print a final summary line."""
    if RICH_AVAILABLE:
        console.print("─" * 40, style="dim")
        if errors == 0 and warnings == 0:
            console.print("[bold green]All checks passed![/bold green]")
        elif errors == 0:
            console.print(f"[bold green]All checks passed[/bold green] with [yellow]{warnings} warning(s)[/yellow]")
        else:
            console.print(f"[bold red]{errors} error(s)[/bold red], [yellow]{warnings} warning(s)[/yellow]")
    else:
        print("=" * 40)
        if errors == 0 and warnings == 0:
            print("All checks passed!")
        elif errors == 0:
            print(f"All checks passed with {warnings} warning(s)")
        else:
            print(f"{errors} error(s), {warnings} warning(s)")
