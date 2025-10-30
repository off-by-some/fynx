"""
FynX Conway's Game of Life - Feature-Rich, Elegantly Reactive

A complete Game of Life implementation where EVERYTHING is reactive math!
The entire simulation is a pure computation graph.

Features:
- Auto-evolving from random primordial soup
- Population statistics as derived observables
- Stability detection (oscillators, still lifes)
- Pattern recognition in the reactive graph
- Density analysis
- Generational coloring with age tracking
- Population history sparkline
- Auto-restart when stable

Interactive controls shown (keyboard input not implemented in this demo)
All computation is in the reactive graph - the UI just observes!
"""

import math
import random
import time

from fynx import Store, observable

from .tui import (
    H1,
    Box,
    Col,
    Component,
    Line,
    PerformanceStats,
    ReactiveComponent,
    RichText,
    Row,
    Spacer,
    Text,
    render,
)

# ============================================================================
# Conway's Game of Life - Pure Reactive Mathematics
# ============================================================================


class Life(Store):
    """All Game of Life state as reactive observables"""

    # Core state - dict of (x, y) -> age
    cells = observable({})
    generation = observable(0)

    # Control state (for visual indicators only)
    paused = observable(False)
    step_mode = observable(False)

    # Grid dimensions
    width = 100
    height = 50

    # === DERIVED OBSERVABLES - Pure Reactive Math ===

    # Population count
    population = cells >> (lambda c: len(c))

    # Average age of cells
    avg_age = cells >> (lambda c: sum(age for age in c.values()) / len(c) if c else 0)

    # Max age (oldest cell)
    max_age = cells >> (lambda c: max(c.values()) if c else 0)

    # Population density (cells per 100 units)
    density = (cells + observable(width) + observable(height)) >> (
        lambda c, w, h: (len(c) / (w * h)) * 100 if w * h > 0 else 0
    )

    # Bounding box of live cells
    bounds = cells >> (
        lambda c: (
            (
                min(x for x, y in c.keys()),
                min(y for x, y in c.keys()),
                max(x for x, y in c.keys()),
                max(y for x, y in c.keys()),
            )
            if c
            else (0, 0, 0, 0)
        )
    )

    # Spread (how large is the live area)
    spread = bounds >> (
        lambda b: (
            math.sqrt((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) if b[2] > b[0] else 0
        )
    )

    # Population history for tracking
    pop_history = observable([])

    # Stability detection - are we oscillating or stable?
    is_stable = pop_history >> (
        lambda h: (
            len(h) > 20 and len(set(h[-10:])) <= 3  # Last 10 gens have ≤3 unique values
        )
    )

    # Pattern type detection (heuristic)
    pattern_type = (population + spread + is_stable) >> (
        lambda p, s, stable: (
            "Still Life"
            if stable and s < 5
            else (
                "Oscillator"
                if stable and s < 20
                else (
                    "Spaceship"
                    if p > 10 and s > 20 and not stable
                    else "Growing" if p > 100 else "Chaos"
                )
            )
        )
    )

    # Birth/death rates (computed from last generation)
    births = observable(0)
    deaths = observable(0)

    # Birth/death ratio
    growth_rate = (births + deaths) >> (
        lambda b, d: (b - d) / max(d, 1) if d > 0 else 0
    )

    # Entropy measure (how chaotic is the pattern)
    entropy = (cells + avg_age) >> (
        lambda c, avg: (
            sum(abs(age - avg) for age in c.values()) / len(c) if c and avg > 0 else 0
        )
    )


# ============================================================================
# Pure Functional Game of Life Logic
# ============================================================================


def get_neighbors(x, y):
    """Get list of neighbor coordinates"""
    return [
        (x - 1, y - 1),
        (x, y - 1),
        (x + 1, y - 1),
        (x - 1, y),
        (x + 1, y),
        (x - 1, y + 1),
        (x, y + 1),
        (x + 1, y + 1),
    ]


def count_live_neighbors(cells, x, y):
    """Count live neighbors for a cell"""
    return sum(1 for nx, ny in get_neighbors(x, y) if (nx, ny) in cells)


def step_life():
    """Advance one generation - pure functional transformation"""
    current_cells = Life.cells.value
    new_cells = {}

    # Track births and deaths
    birth_count = 0
    death_count = 0

    # Consider all live cells and their neighbors
    cells_to_check = set(current_cells.keys())
    for x, y in list(current_cells.keys()):
        cells_to_check.update(get_neighbors(x, y))

    # Apply Conway's rules
    for x, y in cells_to_check:
        alive = (x, y) in current_cells
        neighbors = count_live_neighbors(current_cells, x, y)

        if alive:
            if neighbors in (2, 3):
                # Survival
                age = current_cells[(x, y)]
                new_cells[(x, y)] = age + 1
            else:
                # Death
                death_count += 1
        else:
            if neighbors == 3:
                # Birth
                new_cells[(x, y)] = 0
                birth_count += 1

    # Update all reactive state
    Life.cells = new_cells
    Life.generation = Life.generation.value + 1
    Life.births = birth_count
    Life.deaths = death_count

    # Update population history
    history = Life.pop_history.value[-100:] if Life.pop_history.value else []
    history.append(len(new_cells))
    Life.pop_history = history


def spawn_random_soup(density=0.3):
    """Create random initial state"""
    cells = {}
    center_x = Life.width // 2
    center_y = Life.height // 2
    radius = min(Life.width, Life.height) // 3

    # Spawn in a circular region
    for _ in range(int(Life.width * Life.height * density)):
        angle = random.random() * 2 * math.pi
        r = random.random() * radius
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        cells[(x, y)] = 0

    Life.cells = cells
    Life.generation = 0
    Life.pop_history = []
    Life.births = 0
    Life.deaths = 0
    Life.paused = False
    Life.step_mode = False


# Control functions (would be used with keyboard input)
# def clear_grid():
#     Life.cells = {}
#     Life.generation = 0
#     Life.pop_history = []
#     Life.births = 0
#     Life.deaths = 0
#
# def toggle_pause():
#     Life.paused = not Life.paused.value
#
# def toggle_step_mode():
#     Life.step_mode = not Life.step_mode.value
#
# def request_step():
#     if Life.paused.value:
#         Life.step_requested = True


# ============================================================================
# Reactive UI Components
# ============================================================================


class LifeGrid(ReactiveComponent):
    """Render the Game of Life grid with generational coloring"""

    def get_dependencies(self):
        return [Life.cells, Life.bounds]

    def render_component(self):
        cells = Life.cells.value
        bounds = Life.bounds.value
        w = self.props.get("width", 100)
        h = self.props.get("height", 50)

        # Auto-center viewport on the action
        if bounds[2] > bounds[0]:
            center_x = (bounds[0] + bounds[2]) // 2
            center_y = (bounds[1] + bounds[3]) // 2
        else:
            center_x = Life.width // 2
            center_y = Life.height // 2

        view_x = center_x - w // 2
        view_y = center_y - h // 2

        # Render viewport
        lines = []
        for y in range(h):
            line = RichText()
            for x in range(w):
                world_x = x + view_x
                world_y = y + view_y

                if (world_x, world_y) in cells:
                    age = cells[(world_x, world_y)]
                    # Color based on age - gradient from new to old
                    if age == 0:
                        char, color = "█", "bright_white"
                    elif age < 2:
                        char, color = "█", "white"
                    elif age < 5:
                        char, color = "▓", "bright_cyan"
                    elif age < 10:
                        char, color = "▓", "cyan"
                    elif age < 20:
                        char, color = "▒", "blue"
                    else:
                        char, color = "░", "dim"
                else:
                    char, color = " ", None

                line.append(char, style=color)
            lines.append(line)

        return Col(children=lines).render()


class PopulationStats(ReactiveComponent):
    """Display population statistics - all derived observables!"""

    def get_dependencies(self):
        return [Life.population, Life.generation, Life.avg_age, Life.max_age]

    def render_component(self):
        pop = Life.population.value
        gen = Life.generation.value
        avg_age = Life.avg_age.value
        max_age = Life.max_age.value

        return Row(
            equal=True,
            children=[
                Box(
                    title="👥 Population",
                    border="cyan",
                    padding=(0, 1),
                    children=[Text(text=f"{pop:,}", color="cyan", bold=True)],
                ),
                Box(
                    title="🧬 Generation",
                    border="green",
                    padding=(0, 1),
                    children=[Text(text=f"{gen:,}", color="green", bold=True)],
                ),
                Box(
                    title="📊 Avg Age",
                    border="yellow",
                    padding=(0, 1),
                    children=[Text(text=f"{avg_age:.1f}", color="yellow", bold=True)],
                ),
                Box(
                    title="👴 Max Age",
                    border="magenta",
                    padding=(0, 1),
                    children=[Text(text=f"{max_age}", color="magenta", bold=True)],
                ),
            ],
        ).render()


class DynamicsStats(ReactiveComponent):
    """Display dynamics statistics - births, deaths, growth"""

    def get_dependencies(self):
        return [Life.births, Life.deaths, Life.growth_rate, Life.density]

    def render_component(self):
        births = Life.births.value
        deaths = Life.deaths.value
        growth = Life.growth_rate.value
        density = Life.density.value

        growth_color = "green" if growth > 0 else "red" if growth < 0 else "yellow"

        return Row(
            equal=True,
            children=[
                Box(
                    title="🐣 Births",
                    border="green",
                    padding=(0, 1),
                    children=[Text(text=f"{births}", color="green", bold=True)],
                ),
                Box(
                    title="💀 Deaths",
                    border="red",
                    padding=(0, 1),
                    children=[Text(text=f"{deaths}", color="red", bold=True)],
                ),
                Box(
                    title="📈 Growth",
                    border=growth_color,
                    padding=(0, 1),
                    children=[
                        Text(text=f"{growth:+.2f}", color=growth_color, bold=True)
                    ],
                ),
                Box(
                    title="🌡️ Density",
                    border="cyan",
                    padding=(0, 1),
                    children=[Text(text=f"{density:.1f}%", color="cyan", bold=True)],
                ),
            ],
        ).render()


class PatternAnalysis(ReactiveComponent):
    """Pattern recognition and analysis"""

    def get_dependencies(self):
        return [Life.pattern_type, Life.spread, Life.entropy, Life.is_stable]

    def render_component(self):
        pattern = Life.pattern_type.value
        spread = Life.spread.value
        entropy = Life.entropy.value
        stable = Life.is_stable.value

        pattern_colors = {
            "Still Life": "blue",
            "Oscillator": "cyan",
            "Spaceship": "green",
            "Growing": "yellow",
            "Chaos": "magenta",
        }

        color = pattern_colors.get(pattern, "white")
        status = "🔒 STABLE" if stable else "🌀 EVOLVING"
        status_color = "blue" if stable else "green"

        return Row(
            equal=True,
            children=[
                Box(
                    title="🔬 Pattern",
                    border=color,
                    padding=(0, 1),
                    children=[Text(text=pattern, color=color, bold=True)],
                ),
                Box(
                    title="📏 Spread",
                    border="yellow",
                    padding=(0, 1),
                    children=[Text(text=f"{spread:.1f}", color="yellow", bold=True)],
                ),
                Box(
                    title="🌊 Entropy",
                    border="magenta",
                    padding=(0, 1),
                    children=[Text(text=f"{entropy:.2f}", color="magenta", bold=True)],
                ),
                Box(
                    title="Status",
                    border=status_color,
                    padding=(0, 1),
                    children=[Text(text=status, color=status_color, bold=True)],
                ),
            ],
        ).render()


class PopulationGraph(ReactiveComponent):
    """Mini population history graph"""

    def get_dependencies(self):
        return [Life.pop_history]

    def render_component(self):
        history = Life.pop_history.value
        if not history or len(history) < 2:
            return Text(text="📈 Population: (initializing...)", color="dim").render()

        # Sparkline
        max_pop = max(history[-60:]) if history else 1
        min_pop = min(history[-60:]) if history else 0

        chars = "▁▂▃▄▅▆▇█"
        graph = ""
        for pop in history[-60:]:  # Last 60 generations
            if max_pop > min_pop:
                normalized = (pop - min_pop) / (max_pop - min_pop)
            else:
                normalized = 0.5
            idx = int(normalized * (len(chars) - 1))
            graph += chars[idx]

        return Text(text=f"📈 Population History: {graph}", color="cyan").render()


class LifeControls(ReactiveComponent):
    """Interactive controls for the Game of Life"""

    def get_dependencies(self):
        return [Life.generation, Life.is_stable, Life.paused, Life.step_mode]

    def render_component(self):
        gen = Life.generation.value
        stable = Life.is_stable.value
        paused = Life.paused.value
        step_mode = Life.step_mode.value

        # Show what controls would do (keyboard input not implemented in this demo)
        play_pause = "▶️  RESUME" if paused else "⏸️  PAUSE"
        step_text = "S: Step" if step_mode else "S: Step Mode"

        status_text = (
            "🔒 STABLE" if stable else ("⏸️ PAUSED" if paused else "🌀 RUNNING")
        )
        status_color = "yellow" if stable else ("red" if paused else "green")

        return Row(
            equal=True,
            children=[
                Box(
                    title="🎮 Controls",
                    border="blue",
                    padding=(0, 1),
                    children=[
                        Text(text=f"SPACE: {play_pause}", color="cyan"),
                        Text(text=step_text, color="green"),
                        Text(text="C: Clear", color="yellow"),
                        Text(text="R: Random", color="magenta"),
                    ],
                ),
                Box(
                    title="⚙️ Settings",
                    border="green",
                    padding=(0, 1),
                    children=[
                        Text(text="↑↓: Speed", color="cyan"),
                        Text(text="+/-: Zoom", color="green"),
                        Text(text="P: Patterns", color="yellow"),
                    ],
                ),
                Box(
                    title="📊 Status",
                    border=status_color,
                    padding=(0, 1),
                    children=[
                        Text(text=f"Gen: {gen:,}", color="cyan", bold=True),
                        Text(text=status_text, color=status_color),
                    ],
                ),
            ],
        ).render()


class LifeScene(ReactiveComponent):
    """Main Game of Life scene"""

    def get_dependencies(self):
        return [Life.generation, Life.is_stable]

    def render_component(self):
        from rich.console import Console

        console = Console()
        width, height = console.size

        # Calculate grid size
        grid_width = max(60, width - 10)
        grid_height = max(20, int(height * 0.40))

        Life.width = grid_width
        Life.height = grid_height

        stable = Life.is_stable.value

        return Box(
            title="🧬 FynX Conway's Game of Life - Pure Reactive Mathematics",
            border="green",
            padding=(1, 2),
            children=[
                H1(text="Everything is a Derived Observable"),
                Spacer(height=1),
                LifeControls(),
                Spacer(height=1),
                PopulationStats(),
                Spacer(height=1),
                DynamicsStats(),
                Spacer(height=1),
                PatternAnalysis(),
                Spacer(height=1),
                Box(
                    title="🌍 Universe (Auto-centered)",
                    border="cyan",
                    padding=(0, 1),
                    children=[LifeGrid(width=grid_width, height=grid_height)],
                ),
                Spacer(height=1),
                PopulationGraph(),
                Spacer(height=1),
                PerformanceStats(),
                Spacer(height=1),
                Box(
                    title="⚡ Reactive Features",
                    border="yellow",
                    padding=(0, 1),
                    children=[
                        Text(
                            text="✨ All stats computed from reactive observables",
                            color="yellow",
                        ),
                        Text(
                            text="🎯 Pattern detection runs in computation graph",
                            color="cyan",
                        ),
                        Text(
                            text="🔄 Auto-restarts when pattern stabilizes",
                            color="green",
                        ),
                        Text(
                            text="📊 Birth/death rates, entropy, density - all derived!",
                            color="magenta",
                        ),
                    ],
                ),
                Line(width=grid_width),
                Text(
                    text=(
                        "Stable pattern detected! Restarting..."
                        if stable
                        else "Evolution in progress..."
                    ),
                    color="yellow" if stable else "green",
                    italic=True,
                ),
            ],
        ).render()


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    print("🧬 FynX Conway's Game of Life")
    print("Pure reactive mathematics - everything is a derived observable!\n")

    # Start with random primordial soup!
    spawn_random_soup(density=0.65)

    import os

    fps = 24 if os.environ.get("LIMITED_FPS") else 0

    app = LifeScene()
    renderer = render(app, fps=fps)

    last_step = 0
    step_delay = 100  # ms between generations (global for keyboard control)

    with renderer.start():
        try:
            while True:
                current_time = time.time() * 1000

                # Step at controlled speed (auto-running for demo)
                should_step = False
                if current_time - last_step >= step_delay:
                    should_step = True

                if should_step:
                    step_life()
                    last_step = current_time

                    # Auto-restart if stable (after 200 generations minimum)
                    if Life.is_stable.value and Life.generation.value > 200:
                        print(
                            f"\n🔒 Stable pattern detected at generation {Life.generation.value}"
                        )
                        print(f"   Pattern type: {Life.pattern_type.value}")
                        print(f"   Final population: {Life.population.value}")
                        print("\n🌱 Spawning new primordial soup...\n")
                        time.sleep(2)
                        spawn_random_soup(density=random.uniform(0.2, 0.35))

                time.sleep(0.016)  # ~60 FPS UI updates

        except KeyboardInterrupt:
            print(f"\n✨ Simulated {Life.generation.value:,} generations!")
            print(f"👥 Final population: {Life.population.value:,}")
            print(f"🔬 Pattern type: {Life.pattern_type.value}")
            print(f"📊 Final entropy: {Life.entropy.value:.2f}")
            print("\nEvery stat was a pure derived observable! 🚀")
