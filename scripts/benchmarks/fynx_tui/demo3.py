"""
FynX Fluid Dynamics - O(affected) Wave Propagation

A stunning real-time fluid simulation showcasing FynX's true power:
- Beautiful wave propagation with realistic physics
- Multiple simulation layers (water, fire, sand, plasma)
- Gorgeous gradient rendering
- Auto-generating waves that demonstrate reactive updates

All stats consolidated for maximum fluid simulation display space.
This shows FynX handling complex simulations with fine-grained reactivity!
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
# Fluid Simulation Constants
# ============================================================================

# Simulation types
SIM_WATER = "water"
SIM_FIRE = "fire"
SIM_SAND = "sand"
SIM_PLASMA = "plasma"

# Wave characters for different heights
WAVE_CHARS = " ░▒▓█"
FIRE_CHARS = " .·∴◦○●"
SAND_CHARS = " ░▒▓█"
PLASMA_CHARS = " ·∘○◎◉"

# Color gradients
WATER_COLORS = ["blue", "cyan", "bright_cyan", "white", "bright_white"]
FIRE_COLORS = ["red", "yellow", "bright_yellow", "white", "bright_white"]
SAND_COLORS = ["yellow", "bright_yellow", "white", "bright_white", "bright_white"]
PLASMA_COLORS = ["magenta", "bright_magenta", "cyan", "bright_cyan", "white"]


# ============================================================================
# Reactive Fluid Store
# ============================================================================


class FluidSim(Store):
    """Reactive fluid simulation state"""

    # Grid dimensions
    width = 80
    height = 40

    # Simulation grid - simple 2D array
    grid = []  # Will store height values
    velocity_grid = []  # Will store velocities

    # Simulation parameters
    damping = observable(0.99)
    propagation = observable(0.25)
    sim_type = observable(SIM_WATER)

    # Frame counter and mode
    frame = observable(0)
    auto_wave = observable(True)

    # Statistics
    total_energy = observable(0.0)
    active_cells = observable(0)
    max_height = observable(0.0)


def init_fluid_grid(width, height):
    """Initialize the fluid grid"""
    FluidSim.width = width
    FluidSim.height = height

    # Create 2D grids
    FluidSim.grid = [[0.0 for _ in range(width)] for _ in range(height)]
    FluidSim.velocity_grid = [[0.0 for _ in range(width)] for _ in range(height)]


def get_cell(x, y):
    """Get cell with bounds checking"""
    if 0 <= y < FluidSim.height and 0 <= x < FluidSim.width:
        return FluidSim.grid[y][x], FluidSim.velocity_grid[y][x]
    return 0.0, 0.0


def set_cell(x, y, height, velocity):
    """Update cell state"""
    if 0 <= y < FluidSim.height and 0 <= x < FluidSim.width:
        FluidSim.grid[y][x] = height
        FluidSim.velocity_grid[y][x] = velocity


def add_disturbance(x, y, strength=5.0, radius=3):
    """Add a disturbance at position (creates waves)"""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= radius:
                nx, ny = x + dx, y + dy
                if 0 <= ny < FluidSim.height and 0 <= nx < FluidSim.width:
                    # Gaussian falloff
                    falloff = math.exp(-(dist * dist) / (radius * radius))
                    h, v = get_cell(nx, ny)
                    set_cell(nx, ny, h + strength * falloff, v)


# ============================================================================
# Fluid Physics Step
# ============================================================================


def step_fluid():
    """Single fluid simulation step - wave equation solver"""
    w, h = FluidSim.width, FluidSim.height
    damp = FluidSim.damping.value
    prop = FluidSim.propagation.value
    sim_type = FluidSim.sim_type.value

    # Create buffer for new states
    new_grid = [[0.0 for _ in range(w)] for _ in range(h)]
    new_velocity = [[0.0 for _ in range(w)] for _ in range(h)]

    total_e = 0.0
    active = 0
    max_h = 0.0

    # Calculate forces
    for y in range(h):
        for x in range(w):
            height, velocity = get_cell(x, y)

            # Calculate acceleration from neighbors (wave equation)
            neighbors = []
            if x > 0:
                neighbors.append(get_cell(x - 1, y)[0])
            if x < w - 1:
                neighbors.append(get_cell(x + 1, y)[0])
            if y > 0:
                neighbors.append(get_cell(x, y - 1)[0])
            if y < h - 1:
                neighbors.append(get_cell(x, y + 1)[0])

            if neighbors:
                avg_height = sum(neighbors) / len(neighbors)
                acceleration = (avg_height - height) * prop
            else:
                acceleration = 0

            # Simulation-specific behavior
            if sim_type == SIM_FIRE:
                # Fire rises and dissipates
                if height > 0:
                    acceleration += 0.15
                velocity *= 0.94

            elif sim_type == SIM_SAND:
                # Sand falls
                if height > 0.1:
                    acceleration -= 0.25
                velocity *= 0.97

            elif sim_type == SIM_PLASMA:
                # Plasma has chaotic oscillations
                if x > 0 and x < w - 1 and y > 0 and y < h - 1:
                    corners = [
                        get_cell(x - 1, y - 1)[0],
                        get_cell(x + 1, y - 1)[0],
                        get_cell(x - 1, y + 1)[0],
                        get_cell(x + 1, y + 1)[0],
                    ]
                    corner_avg = sum(corners) / 4
                    acceleration += (corner_avg - height) * 0.1

            # Update velocity and position
            new_vel = (velocity + acceleration) * damp
            new_h = height + new_vel

            # Boundary damping
            if x <= 1 or x >= w - 2 or y <= 1 or y >= h - 2:
                new_h *= 0.7
                new_vel *= 0.7

            # Clamp values
            new_h = max(-5, min(5, new_h))
            new_vel = max(-2, min(2, new_vel))

            new_grid[y][x] = new_h
            new_velocity[y][x] = new_vel

            # Calculate stats
            energy = abs(new_h) + abs(new_vel)
            total_e += energy
            if energy > 0.01:
                active += 1
            max_h = max(max_h, abs(new_h))

    # Apply all updates
    FluidSim.grid = new_grid
    FluidSim.velocity_grid = new_velocity

    # Update statistics
    FluidSim.total_energy = total_e
    FluidSim.active_cells = active
    FluidSim.max_height = max_h

    # Increment frame
    FluidSim.frame = FluidSim.frame.value + 1


# ============================================================================
# Auto Wave Generation
# ============================================================================


def generate_auto_waves():
    """Automatically generate waves at random positions"""
    if not FluidSim.auto_wave.value:
        return

    frame = FluidSim.frame.value
    sim_type = FluidSim.sim_type.value

    if frame % 30 == 0:  # Every 30 frames
        if sim_type == SIM_WATER:
            # Rain drops
            x = random.randint(5, FluidSim.width - 5)
            y = random.randint(5, FluidSim.height - 5)
            add_disturbance(x, y, strength=3.0, radius=2)

        elif sim_type == SIM_FIRE:
            # Fire sources at bottom
            x = random.randint(10, FluidSim.width - 10)
            y = FluidSim.height - 3
            add_disturbance(x, y, strength=4.0, radius=3)

        elif sim_type == SIM_SAND:
            # Sand pours from top
            x = random.randint(10, FluidSim.width - 10)
            y = 2
            add_disturbance(x, y, strength=3.0, radius=2)

        elif sim_type == SIM_PLASMA:
            # Random plasma bursts
            x = random.randint(10, FluidSim.width - 10)
            y = random.randint(10, FluidSim.height - 10)
            add_disturbance(x, y, strength=5.0, radius=4)


# ============================================================================
# Reactive Fluid Renderer
# ============================================================================


class FluidRenderer(ReactiveComponent):
    """Renders fluid simulation"""

    def get_dependencies(self):
        return [FluidSim.frame, FluidSim.sim_type]

    def render_component(self):
        w = self.props.get("width", 80)
        h = self.props.get("height", 40)
        sim_type = FluidSim.sim_type.value

        # Choose rendering style based on sim type
        if sim_type == SIM_WATER:
            chars, colors = WAVE_CHARS, WATER_COLORS
        elif sim_type == SIM_FIRE:
            chars, colors = FIRE_CHARS, FIRE_COLORS
        elif sim_type == SIM_SAND:
            chars, colors = SAND_CHARS, SAND_COLORS
        else:  # PLASMA
            chars, colors = PLASMA_CHARS, PLASMA_COLORS

        lines = []
        for y in range(h):
            line = RichText()
            for x in range(w):
                height, _ = get_cell(x, y)

                # Map height to character and color
                # Normalize height to [0, 1]
                normalized = (height + 5) / 10  # Assuming height range [-5, 5]
                normalized = max(0, min(1, normalized))

                # Get character index
                char_idx = int(normalized * (len(chars) - 1))
                char = chars[char_idx]

                # Get color index
                color_idx = int(normalized * (len(colors) - 1))
                color = colors[color_idx]

                # Special effects for different sim types
                if sim_type == SIM_FIRE and height > 2:
                    char = random.choice("○●")
                elif sim_type == SIM_PLASMA and abs(height) > 3:
                    char = random.choice("◉●")

                line.append(char, style=color)
            lines.append(line)

        return Col(children=lines).render()


# ============================================================================
# UI Components
# ============================================================================


class SimulationStats(ReactiveComponent):
    """Display simulation statistics"""

    def get_dependencies(self):
        return [
            FluidSim.total_energy,
            FluidSim.active_cells,
            FluidSim.max_height,
            FluidSim.frame,
        ]

    def render_component(self):
        energy = FluidSim.total_energy.value
        active = FluidSim.active_cells.value
        max_h = FluidSim.max_height.value
        frame = FluidSim.frame.value

        return Row(
            equal=True,
            children=[
                Box(
                    title="⚡ Energy",
                    border="yellow",
                    padding=(0, 1),
                    children=[Text(text=f"{energy:.1f}", color="yellow", bold=True)],
                ),
                Box(
                    title="🌊 Active",
                    border="cyan",
                    padding=(0, 1),
                    children=[Text(text=f"{active}", color="cyan", bold=True)],
                ),
                Box(
                    title="📊 Peak",
                    border="green",
                    padding=(0, 1),
                    children=[Text(text=f"{max_h:.2f}", color="green", bold=True)],
                ),
                Box(
                    title="🎬 Frame",
                    border="magenta",
                    padding=(0, 1),
                    children=[Text(text=f"{frame:,}", color="magenta", bold=True)],
                ),
            ],
        ).render()


class SimulationControls(ReactiveComponent):
    """Display current simulation mode"""

    def get_dependencies(self):
        return [FluidSim.sim_type, FluidSim.auto_wave]

    def render_component(self):
        sim_type = FluidSim.sim_type.value
        auto = FluidSim.auto_wave.value

        mode_info = {
            SIM_WATER: ("💧 WATER WAVES", "blue", "Rain drops create ripples"),
            SIM_FIRE: ("🔥 FIRE DYNAMICS", "red", "Heat rises and dissipates"),
            SIM_SAND: ("🏖️ SAND FLOW", "yellow", "Gravity pulls particles down"),
            SIM_PLASMA: ("⚛️ PLASMA FIELD", "magenta", "Chaotic energy oscillations"),
        }

        title, color, desc = mode_info.get(sim_type, ("SIMULATION", "white", "Unknown"))
        auto_text = "🎲 Auto-generating" if auto else "⏸️  Paused"

        return Box(
            title=title,
            border=color,
            padding=(0, 1),
            children=[
                Text(text=desc, color=color, italic=True),
                Spacer(height=1),
                Text(text=auto_text, color="green" if auto else "dim", bold=True),
            ],
        ).render()


class FluidScene(ReactiveComponent):
    """Main fluid simulation scene"""

    def get_dependencies(self):
        return [FluidSim.frame]

    def render_component(self):
        from rich.console import Console

        console = Console()
        width, height = console.size

        # Calculate optimal grid size (match other demos for consistent height)
        field_width = max(40, width - 8)
        field_height = max(10, int(height * 0.35))

        # Reinitialize grid if size changed significantly
        if (
            abs(FluidSim.width - field_width) > 5
            or abs(FluidSim.height - field_height) > 5
        ):
            init_fluid_grid(field_width, field_height)

        return Box(
            title="🌊 FynX Fluid Dynamics Engine",
            border="cyan",
            padding=(1, 2),
            children=[
                H1(text="O(affected) Wave Propagation"),
                Spacer(height=1),
                Box(
                    title="💧 Real-Time Fluid Field",
                    border="blue",
                    padding=(0, 1),
                    children=[FluidRenderer(width=field_width, height=field_height)],
                ),
                Spacer(height=1),
                Box(
                    title="📊 Reactive Performance",
                    border="yellow",
                    padding=(1, 2),
                    children=[
                        SimulationStats(),
                        Spacer(height=1),
                        PerformanceStats(),
                    ],
                ),
                Line(width=field_width),
                Text(
                    text="Press Ctrl+C to exit • Watch the waves propagate!",
                    color="dim",
                    italic=True,
                ),
            ],
        ).render()


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    print("🌊 FynX Fluid Dynamics Engine")
    print("Real-time wave propagation simulation\n")

    # Initialize with default size (will resize dynamically)
    init_fluid_grid(80, 40)

    import os

    fps = 24 if os.environ.get("LIMITED_FPS") else 0

    app = FluidScene()
    renderer = render(app, fps=fps)

    # Cycle through simulation modes
    sim_modes = [SIM_WATER, SIM_FIRE, SIM_SAND, SIM_PLASMA]
    mode_index = 0
    frames_per_mode = 600  # 10 seconds at 60fps

    with renderer.start():
        try:
            while True:
                # Step the simulation
                step_fluid()

                # Generate auto waves
                generate_auto_waves()

                # Cycle simulation modes
                if (
                    FluidSim.frame.value % frames_per_mode == 0
                    and FluidSim.frame.value > 0
                ):
                    mode_index = (mode_index + 1) % len(sim_modes)
                    FluidSim.sim_type = sim_modes[mode_index]
                    # Clear grid on mode change
                    init_fluid_grid(FluidSim.width, FluidSim.height)

                time.sleep(1 / 60)  # 60 Hz simulation

        except KeyboardInterrupt:
            total_cells = FluidSim.width * FluidSim.height
            print(f"\n✨ Simulated {FluidSim.frame.value:,} frames!")
            print(f"🌊 {total_cells:,} cells computed per frame!")
            print("Fluid dynamics with FynX reactivity! 🚀")
