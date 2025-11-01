"""
FynX-TUI - Hyper-Optimized Reactive Terminal UI Framework

Built from the ground up with FynX's O(affected) reactivity.
Every component is a reactive node in the computation graph.
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from rich import box
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text as RichText

from fynx import Store, observable

# ============================================================================
# Reactive Component System
# ============================================================================


class Component(ABC):
    """
    Base component with reactive props.
    Components are pure functions: props in, Rich renderable out.
    """

    def __init__(self, **props):
        self.props = props

    @abstractmethod
    def render(self) -> RenderableType:
        """Transform props into Rich renderables"""
        pass

    def __rich_console__(self, console, options):
        """Rich console protocol - delegate to render()"""
        yield from console.render(self.render(), options)

    def __call__(self, **new_props) -> "Component":
        """Create new instance with updated props (immutable pattern)"""
        return self.__class__(**{**self.props, **new_props})


# ============================================================================
# Layout Components - Pure Transformations
# ============================================================================


class Box(Component):
    """Bordered container with optional styling"""

    def render(self) -> RenderableType:
        children = self.props.get("children", [])

        # Resolve all child components to Rich renderables
        renderables = []
        for child in children:
            if isinstance(child, Component):
                renderables.append(child.render())
            else:
                renderables.append(child)

        kwargs = {
            "title": self.props.get("title"),
            "border_style": self.props.get("border", "cyan"),
            "box": box.ROUNDED,
            "expand": self.props.get("expand", True),
            "padding": self.props.get("padding", (0, 1)),
        }

        # Only add style if it's not None
        if style := self.props.get("style"):
            kwargs["style"] = style

        return Panel(Group(*renderables), **kwargs)


class Row(Component):
    """Horizontal layout"""

    def render(self) -> RenderableType:
        children = self.props.get("children", [])

        renderables = []
        for child in children:
            if isinstance(child, Component):
                renderables.append(child.render())
            else:
                renderables.append(child)

        return Columns(renderables, equal=self.props.get("equal", False), expand=True)


class Col(Component):
    """Vertical layout"""

    def render(self) -> RenderableType:
        children = self.props.get("children", [])

        renderables = []
        for child in children:
            if isinstance(child, Component):
                renderables.append(child.render())
            else:
                renderables.append(child)

        return Group(*renderables)


# ============================================================================
# Text Components - Styled Content
# ============================================================================


class Text(Component):
    """Styled text with optional formatting"""

    def render(self) -> RenderableType:
        text = str(self.props.get("text", ""))

        # Build style from individual props
        style_parts = []
        if color := self.props.get("color"):
            style_parts.append(color)
        if bg := self.props.get("bg"):
            style_parts.append(f"on {bg}")
        if self.props.get("bold"):
            style_parts.append("bold")
        if self.props.get("dim"):
            style_parts.append("dim")
        if self.props.get("italic"):
            style_parts.append("italic")

        style = " ".join(style_parts) if style_parts else self.props.get("style")
        justify = self.props.get("justify")

        kwargs = {}
        if style:
            kwargs["style"] = style
        if justify:
            kwargs["justify"] = justify

        return RichText(text, **kwargs)


class H1(Component):
    """Large centered heading"""

    def render(self) -> RenderableType:
        text = str(self.props.get("text", ""))
        style = self.props.get("style", "bold cyan")
        return RichText(f"\n{text}\n", style=style, justify="center")


class Line(Component):
    """Horizontal divider"""

    def render(self) -> RenderableType:
        char = self.props.get("char", "─")
        width = self.props.get("width", 80)
        style = self.props.get("style", "dim")
        return RichText(char * width, style=style)


class Spacer(Component):
    """Vertical spacing"""

    def render(self) -> RenderableType:
        height = max(0, self.props.get("height", 1) - 1)
        return RichText("\n" * height)


class Tag(Component):
    """Colored badge/label"""

    def render(self) -> RenderableType:
        text = self.props.get("text", "")

        colors = {
            "blue": "white on blue",
            "green": "white on green",
            "yellow": "black on yellow",
            "red": "white on red",
            "cyan": "white on cyan",
            "magenta": "white on magenta",
        }

        color = self.props.get("color", "blue")
        style = colors.get(color, "white on blue")

        return RichText(f" {text} ", style=style)


# ============================================================================
# Reactive Components - Only re-render when dependencies change
# ============================================================================


class ReactiveComponent(Component):
    """
    Component that automatically re-renders when its observable dependencies change.

    Usage:
        class MyComponent(ReactiveComponent):
            def get_dependencies(self):
                return [some_observable, another_observable]

            def render(self):
                return Text(text=f"Value: {some_observable.value}")
    """

    def __init__(self, **props):
        super().__init__(**props)
        self._cached_render = None
        self._last_deps_values = None
        self._unsubscribers = []

        # Subscribe to dependencies
        deps = self.get_dependencies()
        for dep in deps:
            unsub = dep.subscribe(self._on_dependency_change)
            self._unsubscribers.append(unsub)

    def get_dependencies(self):
        """Override to return list of observables this component depends on"""
        return []

    def _on_dependency_change(self, _):
        """Called when any dependency changes - invalidate cache"""
        self._cached_render = None

    def render(self) -> RenderableType:
        """Render with caching - only re-render when dependencies change"""
        # Check if we need to re-render
        current_deps = tuple(dep.value for dep in self.get_dependencies())

        if self._cached_render is None or current_deps != self._last_deps_values:
            self._cached_render = self.render_component()
            self._last_deps_values = current_deps

        return self._cached_render

    def render_component(self) -> RenderableType:
        """Override this to implement your component's rendering logic"""
        raise NotImplementedError("Subclasses must implement render_component")

    def cleanup(self):
        """Clean up subscriptions"""
        for unsub in self._unsubscribers:
            unsub()
        self._unsubscribers.clear()


# ============================================================================
# Reactive Application Container
# ============================================================================


class ReactiveApp:
    """
    Application container that manages reactive rendering.

    Re-renders the entire component tree on every frame.
    Components are pure functions, so this is fast.
    """

    def __init__(self, root: Component, fps: int = 30):
        self.console = Console()
        self.root = root
        self.fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0  # 0 = unlimited FPS

        self._live = None
        self._running = False
        self._last_render_time = 0

        # Initialize performance store
        PerformanceStore.target_fps = fps
        PerformanceStore.frame_count = 0
        PerformanceStore.fps = 0.0
        PerformanceStore.avg_render_time = 0.0
        PerformanceStore.min_render_time = 0.0
        PerformanceStore.max_render_time = 0.0
        PerformanceStore.uptime = 0.0
        PerformanceStore.performance_ratio = 0.0

        # Performance tracking
        self._start_time = 0
        self._render_times = []
        self._fps_history = []

    def _render_root(self) -> RenderableType:
        """Render the entire component tree"""
        try:
            return self.root.render()
        except Exception as e:
            return RichText(f"[red]Render error: {e}[/red]")

    def start(self):
        """Start the reactive rendering loop"""
        self._running = True

        # Initial render
        initial = self._render_root()

        # Rich requires refresh_per_second > 0, so use at least 1 FPS for display
        # but physics can still run unlimited
        display_fps = max(1, self.fps)

        self._live = Live(
            initial,
            console=self.console,
            refresh_per_second=display_fps,
            screen=False,
        )

        # Start update thread
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()

        return self._live

    def _update_loop(self):
        """Continuously re-render at target FPS"""
        self._start_time = time.time()

        while self._running:
            frame_start = time.time()

            try:
                if self._live:
                    render_start = time.time()
                    renderable = self._render_root()
                    render_time = time.time() - render_start

                    self._live.update(renderable)
                    self._live.refresh()

                    # Track render time
                    self._render_times.append(render_time)
                    if len(self._render_times) > 60:  # Keep last 60 samples
                        self._render_times.pop(0)

            except Exception as e:
                pass  # Ignore errors, keep running

            # Calculate actual FPS
            frame_time = time.time() - frame_start
            actual_fps = 1.0 / frame_time if frame_time > 0 else 0

            self._fps_history.append(actual_fps)
            if len(self._fps_history) > 60:  # Keep last 60 samples
                self._fps_history.pop(0)

            # Update performance store reactively
            PerformanceStore.frame_count = PerformanceStore.frame_count.value + 1
            PerformanceStore.uptime = time.time() - self._start_time

            if self._fps_history:
                avg_fps = sum(self._fps_history) / len(self._fps_history)
                PerformanceStore.fps = avg_fps
                PerformanceStore.performance_ratio = (
                    avg_fps / self.fps if self.fps > 0 else 1.0
                )

            if self._render_times:
                avg_render = sum(self._render_times) / len(self._render_times) * 1000
                min_render = min(self._render_times) * 1000
                max_render = max(self._render_times) * 1000

                PerformanceStore.avg_render_time = avg_render
                PerformanceStore.min_render_time = min_render
                PerformanceStore.max_render_time = max_render

            # Sleep to maintain target FPS (skip sleep for unlimited FPS)
            if self.fps > 0:
                elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_time - elapsed)
                time.sleep(sleep_time)

    def stop(self):
        """Stop the rendering loop"""
        self._running = False


# ============================================================================
# Performance Store - Reactive performance tracking
# ============================================================================


class PerformanceStore(Store):
    """Reactive store for performance metrics"""

    # Real-time performance observables
    fps = observable(0.0)
    avg_render_time = observable(0.0)
    min_render_time = observable(0.0)
    max_render_time = observable(0.0)
    frame_count = observable(0)
    uptime = observable(0.0)
    target_fps = observable(30)
    performance_ratio = observable(0.0)

    # Derived observables
    fps_color = (performance_ratio + target_fps) >> (
        lambda ratio, target: (
            "cyan"
            if target == 0  # Unlimited FPS
            else "green" if ratio >= 0.95 else "yellow" if ratio >= 0.8 else "red"
        )
    )

    fps_display = (fps + target_fps) >> (
        lambda fps_val, target: (
            f"{fps_val:.1f} (unlimited)" if target == 0 else f"{fps_val:.1f}/{target}"
        )
    )


# ============================================================================
# Global App Reference
# ============================================================================

_current_app = None


def get_current_app():
    """Get the currently running app instance"""
    return _current_app


# ============================================================================
# Convenience Functions
# ============================================================================


def render(component: Component, fps: int = 30) -> ReactiveApp:
    """
    Create a reactive app from a root component.

    The app continuously re-renders at the target FPS.
    Set fps=0 for unlimited FPS (as fast as possible).
    Default fps=30, but demo uses fps=0 (unlimited) by default.
    """
    global _current_app
    app = ReactiveApp(component, fps=fps)
    _current_app = app
    return app


# ============================================================================
# Helper: Derived Observables
# ============================================================================


def combine(*observables):
    """
    Combine multiple observables using FynX's + operator.

    Returns a combined observable that updates when any input changes.
    """
    if not observables:
        raise ValueError("Must provide at least one observable")

    result = observables[0]
    for obs in observables[1:]:
        result = result + obs

    return result


class PerformanceStats(Component):
    """Display real-time performance statistics from reactive store"""

    def render(self):
        fps_text = PerformanceStore.fps_display.value
        fps_color = PerformanceStore.fps_color.value
        avg_render = PerformanceStore.avg_render_time.value
        min_render = PerformanceStore.min_render_time.value
        max_render = PerformanceStore.max_render_time.value
        frame_count = PerformanceStore.frame_count.value
        uptime = PerformanceStore.uptime.value

        return Col(
            children=[
                Row(
                    equal=True,
                    children=[
                        Box(
                            title="🎯 FPS",
                            border="blue",
                            padding=(0, 1),
                            children=[Text(text=fps_text, color=fps_color, bold=True)],
                        ),
                        Box(
                            title="⚡ Render",
                            border="green",
                            padding=(0, 1),
                            children=[
                                Text(
                                    text=f"Avg: {avg_render:.1f}ms",
                                    color="green",
                                    bold=True,
                                ),
                                Text(text=f"Min: {min_render:.1f}ms", color="green"),
                                Text(text=f"Max: {max_render:.1f}ms", color="green"),
                            ],
                        ),
                        Box(
                            title="📊 Frames",
                            border="yellow",
                            padding=(0, 1),
                            children=[
                                Text(text=f"{frame_count:,}", color="yellow", bold=True)
                            ],
                        ),
                        Box(
                            title="⏱️ Uptime",
                            border="magenta",
                            padding=(0, 1),
                            children=[
                                Text(text=f"{uptime:.1f}s", color="magenta", bold=True)
                            ],
                        ),
                    ],
                )
            ]
        ).render()


def derive(observable_or_combined, transform_fn):
    """
    Create a derived observable using FynX's >> operator.

    Usage:
        derived = derive(some_observable, lambda x: x * 2)
        derived = derive(combine(obs1, obs2), lambda a, b: a + b)
    """
    return observable_or_combined >> transform_fn
