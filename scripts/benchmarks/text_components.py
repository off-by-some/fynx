"""
Reactive Text Components - Fynx-powered terminal UI framework

Uses Fynx observables for reactive state management and automatic re-rendering.
Components automatically update when their reactive dependencies change.
"""

import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fynx import observable, reactive, transaction

# ============================================================================
# GLOBAL STATE - Reactive Console and UI State
# ============================================================================

# Global console observable - can be changed dynamically
console = observable(Console())

# Global UI state - reactive properties that components can subscribe to
ui_state = observable(
    {"theme": "default", "width": 80, "show_details": True, "color_scheme": "cyan"}
)


# ============================================================================
# UTILITY FUNCTIONS - Reactive versions
# ============================================================================


def get_theme():
    """Theme getter."""
    return ui_state.value["theme"]


def get_color_scheme():
    """Color scheme getter."""
    return ui_state.value["color_scheme"]


def get_width():
    """Width getter."""
    return ui_state.value["width"]


def should_show_details():
    """Detail visibility getter."""
    return ui_state.value["show_details"]


def format_number(num: float, precision: int = 0) -> str:
    """Format number with thousands separators."""
    return f"{num:,.{precision}f}"


def format_bytes(bytes_val: float, precision: int = 1) -> str:
    """Format bytes into human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.{precision}f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.{precision}f} TB"


def get_performance_color(ratio: float) -> str:
    """Get color based on performance ratio - reactive to color scheme."""
    scheme = get_color_scheme()
    if scheme == "green":
        if ratio >= 10:
            return "bright_green"
        elif ratio >= 5:
            return "green"
        elif ratio >= 2:
            return "yellow"
        else:
            return "dim"
    else:  # default cyan scheme
        if ratio >= 10:
            return "bright_green"
        elif ratio >= 5:
            return "green"
        elif ratio >= 2:
            return "yellow"
        else:
            return "dim"


def get_bar_char(intensity: float) -> str:
    """Get appropriate bar character based on intensity."""
    if intensity >= 0.875:
        return "█"
    elif intensity >= 0.625:
        return "▓"
    elif intensity >= 0.375:
        return "▒"
    elif intensity >= 0.125:
        return "░"
    else:
        return "·"


# ============================================================================
# REACTIVE COMPONENT SYSTEM
# ============================================================================


class ReactiveComponent:
    """Base class for components with reactive props."""

    def __init__(self, **props):
        self.props = observable(props)

    def render(self) -> str:
        """Render method - must be implemented by subclasses."""
        raise NotImplementedError

    @property
    def output(self) -> str:
        """Get rendered output."""
        return self.render()


def render_component(component: ReactiveComponent):
    """Render a reactive component to the console."""
    current_console = console.value
    output = component.output
    if output:
        current_console.print(output, end="")


# ============================================================================
# PRIMITIVE COMPONENTS
# ============================================================================


class TextComponent(ReactiveComponent):
    """Reactive text component."""

    def render(self) -> str:
        props = self.props.value
        content = props.get("children", "")
        style = props.get("style", "")
        bold = props.get("bold", False)
        dim = props.get("dim", False)

        styles = []
        if style:
            styles.append(style)
        if bold:
            styles.append("bold")
        if dim:
            styles.append("dim")

        style_str = " ".join(styles) if styles else None
        return f"[{style_str}]{content}[/{style_str}]" if style_str else content


class SpaceComponent(ReactiveComponent):
    """Reactive spacing component."""

    def render(self) -> str:
        props = self.props.value
        lines = props.get("lines", 1)
        return "\n" * lines


class DividerComponent(ReactiveComponent):
    """Reactive divider component."""

    def render(self) -> str:
        props = self.props.value
        char = props.get("char", "═")
        length = props.get("length", get_width())
        style = props.get("style", "cyan dim")
        return f"[{style}]{char * length}[/{style}]\n"


# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================


class HeaderComponent(ReactiveComponent):
    """Reactive header component."""

    def render(self) -> str:
        props = self.props.value
        title = props.get("title", "")
        subtitle = props.get("subtitle", "")
        metadata = props.get("metadata", {})

        lines = [""]

        # Title with decorative border
        border_line = "─" * (len(title) + 2)
        lines.append(f"[cyan]╭{border_line}╮[/cyan]")
        lines.append(f"[cyan]│ {title} │[/cyan]")
        lines.append(f"[cyan]╰{border_line}╯[/cyan]")

        if subtitle:
            lines.append(f"[dim]{subtitle}[/dim]")

        # Metadata badges
        if metadata:
            badges = []
            for key, value in metadata.items():
                badges.append(f"[cyan]●[/cyan] [bold]{key}:[/bold] {value}")
            lines.append(" | ".join(badges))

        lines.append("")
        return "\n".join(lines)


class SectionComponent(ReactiveComponent):
    """Reactive section component."""

    def render(self) -> str:
        props = self.props.value
        title = props.get("title", "")
        icon = props.get("icon", "▶")
        description = props.get("description", "")
        stats = props.get("stats", {})

        lines = ["", f"[bold cyan]{icon} {title}[/bold cyan]"]

        if description:
            lines.append(f"[dim]{description}[/dim]")

        if stats:
            stat_text = " | ".join(
                [f"[yellow]{k}:[/yellow] {v}" for k, v in stats.items()]
            )
            lines.append(stat_text)

        lines.append(f"[cyan dim]═{'═' * (get_width() - 1)}[/cyan dim]")
        lines.append("")

        return "\n".join(lines)


class CardComponent(ReactiveComponent):
    """Reactive card component."""

    def render(self) -> str:
        props = self.props.value
        title = props.get("title", "")
        content = props.get("content", "")
        metrics = props.get("metrics", {})
        border_style = props.get("border_style", "blue")

        rich_content = content

        if metrics:
            rich_content += "\n\n[dim]Key Metrics:[/dim]"
            for metric_name, metric_value in metrics.items():
                rich_content += (
                    f"\n  [cyan]●[/cyan] {metric_name}: [bold]{metric_value}[/bold]"
                )

        # Create panel representation
        panel_title = f"[bold]{title}[/bold]" if title else None
        return f"[bold {border_style}]{'─' * (len(title) + 4) if title else '─' * 20}[/bold {border_style}]\n{rich_content}\n[bold {border_style}]{'─' * (len(title) + 4) if title else '─' * 20}[/bold {border_style}]\n"


# ============================================================================
# DATA DISPLAY COMPONENTS
# ============================================================================


class PerformanceTableComponent(ReactiveComponent):
    """Reactive performance table component."""

    def render(self) -> str:
        props = self.props.value
        title = props.get("title", "")
        headers = props.get("headers", [])
        rows = props.get("rows", [])
        highlight_winners = props.get("highlight_winners", True)

        if not headers or not rows:
            return ""

        lines = []
        if title:
            lines.append(f"[bold]{title}[/bold]")
            lines.append("")

        # Simple table representation - in a real implementation this would be more sophisticated
        # For now, just create a text-based table
        col_widths = [
            max(len(str(row[i])) for row in [headers] + rows)
            for i in range(len(headers))
        ]

        # Header row
        header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        lines.append(f"[bold cyan]{header_line}[/bold cyan]")
        lines.append("[cyan]" + "─" * len(header_line) + "[/cyan]")

        # Data rows
        for row in rows:
            if highlight_winners and len(row) >= 5:
                winner = row[4]
                style = (
                    "green" if winner == "FynX" else "red" if winner != "Tie" else ""
                )
                row_line = " | ".join(
                    f"{str(cell):<{w}}" for cell, w in zip(row, col_widths)
                )
                if style:
                    row_line = f"[{style}]{row_line}[/{style}]"
                lines.append(row_line)
            else:
                row_line = " | ".join(
                    f"{str(cell):<{w}}" for cell, w in zip(row, col_widths)
                )
                lines.append(row_line)

        return "\n".join(lines) + "\n"


class TieredAnalysisComponent(ReactiveComponent):
    """Reactive tiered analysis component."""

    def render(self) -> str:
        props = self.props.value
        title = props.get("title", "")
        tiers = props.get("tiers", {})
        show_stats = props.get("show_stats", True)

        if not tiers:
            return ""

        lines = [
            f"[bold cyan]{title}[/bold cyan]",
            f"[cyan dim]═{'═' * (get_width() - 1)}[/cyan dim]",
            "",
        ]

        for tier_name, operations in tiers.items():
            if operations:
                # Calculate tier statistics
                ratios = [ratio for _, ratio in operations]
                if show_stats and len(ratios) > 1:
                    tier_header = (
                        f"[yellow bold]{tier_name}[/yellow bold] "
                        f"[dim]({len(operations)} operations, "
                        f"avg: {statistics.mean(ratios):.1f}x, "
                        f"max: {max(ratios):.1f}x)[/dim]"
                    )
                else:
                    tier_header = f"[yellow bold]{tier_name}[/yellow bold]"

                lines.append(tier_header)

                for op_name, ratio in operations:
                    # Create gradient bar
                    max_bar_length = 50
                    bar_length = min(int(ratio), max_bar_length)

                    # Use different characters for intensity
                    full_bars = bar_length
                    bar = "█" * full_bars

                    # Color based on performance level
                    bar_color = get_performance_color(ratio)

                    lines.append(
                        f"  [dim]•[/dim] {op_name:<40} "
                        f"[bold]{ratio:>6.1f}x[/bold] [{bar_color}]{bar}[/{bar_color}]"
                    )
                    lines.append("")

        return "\n".join(lines)


# ============================================================================
# COMPOSITE COMPONENTS
# ============================================================================


class ProgressIndicatorComponent(ReactiveComponent):
    """Reactive progress indicator component."""

    def render(self) -> str:
        props = self.props.value
        operation_name = props.get("operation_name", "")
        fynx_result = props.get("fynx_result")
        rxpy_result = props.get("rxpy_result")

        if not fynx_result or not rxpy_result:
            return ""

        fynx_ops = fynx_result.operations_per_second
        rxpy_ops = rxpy_result.operations_per_second
        rxpy_lib = rxpy_result.library

        # Check for DNF cases - true DNF only if no operations were completed
        fynx_dnf = fynx_result.operation_time == float("inf") and fynx_ops == 0
        rxpy_dnf = rxpy_result.operation_time == float("inf") and rxpy_ops == 0

        # Check for time-limit cases where results were obtained
        fynx_partial = fynx_result.operation_time == float("inf") and fynx_ops > 0
        rxpy_partial = rxpy_result.operation_time == float("inf") and rxpy_ops > 0

        if fynx_dnf and rxpy_dnf:
            return f"[yellow]🏁[/yellow] {operation_name:<40} [red]Both DNF[/red] [dim](neither completed)[/dim]\n"
        elif fynx_dnf:
            return f"[yellow]🏁[/yellow] {operation_name:<40} [red]FynX DNF[/red] [green]{rxpy_lib} completed[/green]\n"
        elif rxpy_dnf:
            return f"[yellow]🏁[/yellow] {operation_name:<40} [green]FynX completed[/green] [red]{rxpy_lib} DNF[/red]\n"

        # Check for partial results (time limit but got results)
        if fynx_partial and rxpy_partial:
            # Both hit time limit but got results - show comparison
            if fynx_ops > rxpy_ops:
                speedup = fynx_ops / rxpy_ops
                return (
                    f"[yellow]⏱️ [/yellow] {operation_name:<40} "
                    f"[bold green]FynX {speedup:>5.1f}x[/bold green] "
                    f"[dim]({format_number(fynx_ops)} vs {format_number(rxpy_ops)} ops/s)[/dim] [yellow](time limit)[/yellow]\n"
                )
            else:
                speedup = rxpy_ops / fynx_ops
                return (
                    f"[yellow]⏱️ [/yellow] {operation_name:<40} "
                    f"[bold blue]{rxpy_lib} {speedup:>5.1f}x[/bold blue] "
                    f"[dim]({format_number(rxpy_ops)} vs {format_number(fynx_ops)} ops/s)[/dim] [yellow](time limit)[/yellow]\n"
                )
        elif fynx_partial:
            return f"[yellow]⏱️ [/yellow] {operation_name:<40} [green]FynX: {format_number(fynx_ops)} ops/s[/green] [dim](time limit)[/dim] [green]{rxpy_lib} completed[/green]\n"
        elif rxpy_partial:
            return f"[yellow]⏱️ [/yellow] {operation_name:<40} [green]FynX completed[/green] [blue]{rxpy_lib}: {format_number(rxpy_ops)} ops/s[/blue] [dim](time limit)[/dim]\n"

        # Check for zero/negative ops/s
        if fynx_ops <= 0 and rxpy_ops <= 0:
            return f"[red]❌[/red] {operation_name:<40} [red]Both failed[/red] [dim](zero ops/s)[/dim]\n"
        elif fynx_ops <= 0:
            return f"[red]❌[/red] {operation_name:<40} [red]FynX failed[/red] [green]{rxpy_lib}: {format_number(rxpy_ops)} ops/s[/green]\n"
        elif rxpy_ops <= 0:
            return f"[red]❌[/red] {operation_name:<40} [green]FynX: {format_number(fynx_ops)} ops/s[/green] [red]{rxpy_lib} failed[/red]\n"

        if fynx_ops > rxpy_ops:
            speedup = fynx_ops / rxpy_ops
            icon = "✓" if speedup < 5 else "✓✓" if speedup < 10 else "✓✓✓"
            color = "green" if speedup < 5 else "bright_green"
            trophy = (
                "🏆 "
                if hasattr(fynx_result, "is_new_record") and fynx_result.is_new_record
                else ""
            )
            return (
                f"[{color}]{icon}[/{color}] {operation_name:<40} "
                f"[bold {color}]{trophy}FynX {speedup:>5.1f}x[/bold {color}] "
                f"[dim]({format_number(fynx_ops)} ops/s)[/dim]\n"
            )
        else:
            speedup = rxpy_ops / fynx_ops
            icon = "✗" if speedup < 5 else "✗✗" if speedup < 10 else "✗✗✗"
            color = "red" if speedup < 5 else "bright_red"
            trophy = (
                "🏆 "
                if hasattr(rxpy_result, "is_new_record") and rxpy_result.is_new_record
                else ""
            )
            return (
                f"[{color}]{icon}[/{color}] {operation_name:<40} "
                f"[bold {color}]{trophy}{rxpy_lib} {speedup:>5.1f}x[/bold {color}] "
                f"[dim]({format_number(rxpy_ops)} ops/s)[/dim]\n"
            )


class PerformanceSummaryComponent(ReactiveComponent):
    """Reactive performance summary component."""

    def render(self) -> str:
        props = self.props.value
        results = props.get("results", [])

        if not results:
            return ""

        fynx_wins = 0
        rxpy_wins = 0
        total_fynx_advantage = 0
        total_rxpy_advantage = 0
        rxpy_library_names = set()

        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                fynx_result = results[i]
                rxpy_result = results[i + 1]

                # Track the RxPY library name (could be RxPY or RxPY-Opt)
                rxpy_library_names.add(rxpy_result.library)

                fynx_ops = fynx_result.operations_per_second
                rxpy_ops = rxpy_result.operations_per_second

                if fynx_ops > 0 and rxpy_ops > 0:
                    if fynx_ops > rxpy_ops:
                        fynx_wins += 1
                        total_fynx_advantage += fynx_ops / rxpy_ops
                    else:
                        rxpy_wins += 1
                        total_rxpy_advantage += rxpy_ops / fynx_ops
                elif fynx_ops > rxpy_ops:
                    fynx_wins += 1
                    # Don't add to advantage if one is zero
                elif rxpy_ops > fynx_ops:
                    rxpy_wins += 1
                    # Don't add to advantage if one is zero
                # If equal (including both zero), don't count as win

        total_tests = fynx_wins + rxpy_wins
        # Use the most common RxPY library name for display, defaulting to "RxPY"
        rxpy_display_name = "RxPY"
        if rxpy_library_names:
            # If there's only one type, use it; otherwise use "RxPY" as generic
            if len(rxpy_library_names) == 1:
                rxpy_display_name = list(rxpy_library_names)[0]

        winner = (
            "FynX"
            if fynx_wins > rxpy_wins
            else rxpy_display_name if rxpy_wins > fynx_wins else "Tie"
        )
        winner_color = (
            "green" if winner == "FynX" else "blue" if "RxPY" in winner else "yellow"
        )

        # Calculate average advantages
        avg_fynx = total_fynx_advantage / fynx_wins if fynx_wins > 0 else 0
        avg_rxpy = total_rxpy_advantage / rxpy_wins if rxpy_wins > 0 else 0

        # Build content with metrics
        content = (
            f"[bold]Overall Winner: [{winner_color}]{winner}[/{winner_color}][/bold]\n\n"
            f"[green]FynX wins:[/green] {fynx_wins}/{total_tests} "
            f"[dim]({fynx_wins/total_tests*100:.0f}%)[/dim]\n"
            f"[blue]{rxpy_display_name} wins:[/blue] {rxpy_wins}/{total_tests} "
            f"[dim]({rxpy_wins/total_tests*100:.0f}%)[/dim]"
        )

        metrics = {
            "Avg FynX Advantage": f"{avg_fynx:.1f}x when winning",
            "Avg RxPY Advantage": f"{avg_rxpy:.1f}x when winning",
            "Total Tests": f"{total_tests} operations",
        }

        return CardComponent(
            title="🏆 Performance Summary", content=content, metrics=metrics
        ).output


# ============================================================================
# CONVENIENCE FUNCTIONS - Direct API (like React.createElement)
# ============================================================================


def Text(
    content: str = "", *, style: str = None, bold: bool = False, dim: bool = False
):
    """Create and render text component."""
    component = TextComponent(children=content, style=style, bold=bold, dim=dim)
    render_component(component)


def Space(lines: int = 1):
    """Create and render space component."""
    component = SpaceComponent(lines=lines)
    render_component(component)


def Divider(char: str = "═", length: int = 80, *, style: str = "cyan dim"):
    """Create and render divider component."""
    component = DividerComponent(char=char, length=length, style=style)
    render_component(component)


def Header(*, title: str, subtitle: str = None, metadata: Dict[str, Any] = None):
    """Create and render header component."""
    component = HeaderComponent(title=title, subtitle=subtitle, metadata=metadata or {})
    render_component(component)


def Section(
    *,
    title: str,
    icon: str = "▶",
    description: str = None,
    stats: Dict[str, Any] = None,
):
    """Create and render section component."""
    component = SectionComponent(
        title=title, icon=icon, description=description, stats=stats or {}
    )
    render_component(component)


def Card(
    *,
    title: str,
    content: str = None,
    metrics: Dict[str, Any] = None,
    border_style: str = "blue",
):
    """Create and render card component."""
    component = CardComponent(
        title=title,
        content=content or "",
        metrics=metrics or {},
        border_style=border_style,
    )
    render_component(component)


def PerformanceTable(
    *,
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    show_header: bool = True,
    highlight_winners: bool = True,
):
    """Create and render performance table component."""
    component = PerformanceTableComponent(
        title=title,
        headers=headers,
        rows=rows,
        show_header=show_header,
        highlight_winners=highlight_winners,
    )
    render_component(component)


def TieredAnalysis(
    *, title: str, tiers: Dict[str, List[Tuple[str, float]]], show_stats: bool = True
):
    """Create and render tiered analysis component."""
    component = TieredAnalysisComponent(title=title, tiers=tiers, show_stats=show_stats)
    render_component(component)


def ProgressIndicator(*, operation_name: str, fynx_result: Any, rxpy_result: Any):
    """Create and render progress indicator component."""
    component = ProgressIndicatorComponent(
        operation_name=operation_name, fynx_result=fynx_result, rxpy_result=rxpy_result
    )
    render_component(component)


def PerformanceSummary(*, results: List[Any]):
    """Create and render performance summary component."""
    component = PerformanceSummaryComponent(results=results)
    render_component(component)


# ============================================================================
# BENCHMARK RESULTS - Main composite component
# ============================================================================


class BenchmarkResultsComponent(ReactiveComponent):
    """Complete reactive benchmark results component."""

    def render(self) -> str:
        props = self.props.value
        config = props.get("config")
        results = props.get("results", [])
        execution_profiles = props.get("execution_profiles", {})
        elapsed_time = props.get("elapsed_time", 0)

        if not config or not results:
            return ""

        output_parts = []

        # Enhanced header
        header_comp = HeaderComponent(
            title="FynX vs RxPY Performance Comparison",
            subtitle="Comprehensive Reactive Programming Benchmark Analysis",
            metadata={
                "Time Limit": f"{config.time_limit * config.num_iterations}s total",
                "Iterations": config.num_iterations,
                "Total Tests": len(results) // 2 if results else 0,
            },
        )
        output_parts.append(header_comp.output)

        # Summary with enhanced metrics
        if results:
            summary_comp = PerformanceSummaryComponent(results=results)
            output_parts.append(summary_comp.output)

        # Performance tiers with statistics
        if results:
            operations = {}
            for i in range(0, len(results), 2):
                if i + 1 < len(results):
                    fynx_result = results[i]
                    rxpy_result = results[i + 1]
                    operations[fynx_result.operation] = {
                        "FynX": fynx_result,
                        rxpy_result.library: rxpy_result,
                    }

            # This would need more complex reactive logic for the tiers
            # For now, just show a placeholder
            if operations:
                section_comp = SectionComponent(
                    title="PERFORMANCE BY TIER",
                    icon="📊",
                    stats={"Operations": f"{len(operations)} total"},
                )
                output_parts.append(section_comp.output)

        # Footer with completion stats
        output_parts.append(f"[cyan dim]═{'═' * (get_width() - 1)}[/cyan dim]")
        output_parts.append(
            f"\n[bold green]✓[/bold green] Benchmark completed in "
            f"[bold]{elapsed_time:.1f}s[/bold] "
            f"[dim]({len(results)} total measurements)[/dim]\n"
        )

        return "".join(output_parts)


def BenchmarkResults(
    *,
    config: Any,
    results: List[Any],
    execution_profiles: Optional[Dict[str, Any]] = None,
    elapsed_time: Optional[float] = None,
):
    """Create and render complete benchmark results."""
    component = BenchmarkResultsComponent(
        config=config,
        results=results,
        execution_profiles=execution_profiles or {},
        elapsed_time=elapsed_time or 0,
    )
    render_component(component)


# ============================================================================
# UTILITY FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

# These functions are kept for compatibility but now use reactive components internally
