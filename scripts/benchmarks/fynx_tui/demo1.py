"""
FynX Particle Physics - Truly Reactive O(affected) Demo

Everything is reactive! Even performance tracking uses FynX observables and derived values.
Dynamic physics with moving attractor (yellow ⚫), wind forces, and random impulses prevent predictable oscillation.

Components re-render only when their dependencies change:
- ParticleField → particles, center, & attractor position
- StatsPanel → physics stats (energy, speed, center, collisions)
- Dashboard → frame count & particle count
- PerformanceStats → FPS, render times, frames, uptime (all from PerformanceStore)

Runs at unlimited FPS by default to showcase FynX performance. Set LIMITED_FPS=1 to cap at 24 FPS.
FynX ensures only affected computations run - true fine-grained reactivity!
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
# Pure FynX Dataflow Functions - All physics logic in named functions!
# ============================================================================


def compute_particle_forces(ps, rr, rf, ax, ay, astr, wphase, wstr, grav):
    """Compute forces for all particles - FynX computed variable"""
    return [
        compute_single_particle_forces(p, ps, rr, rf, ax, ay, astr, wphase, wstr, grav)
        for p in ps
    ]


def compute_single_particle_forces(
    p,
    all_particles,
    repel_radius,
    repel_force,
    attractor_x,
    attractor_y,
    attractor_strength,
    wind_phase,
    wind_strength,
    gravity,
):
    """Compute forces acting on a single particle"""
    fx, fy = 0.0, 0.0

    # Gravity
    fy += gravity * p["mass"] * 0.1

    # Particle repulsion
    for other in all_particles:
        if other["id"] == p["id"]:
            continue
        dx, dy = p["x"] - other["x"], p["y"] - other["y"]
        dist = math.sqrt(dx * dx + dy * dy)
        if 0.1 < dist < repel_radius:
            force = repel_force / (dist * dist)
            fx += (dx / dist) * force
            fy += (dy / dist) * force

    # Attractor force
    dx_attractor = attractor_x - p["x"]
    dy_attractor = attractor_y - p["y"]
    dist_attractor = math.sqrt(dx_attractor**2 + dy_attractor**2)
    if dist_attractor > 1:
        attractor_force = attractor_strength / (dist_attractor**1.5)
        fx += (dx_attractor / dist_attractor) * attractor_force
        fy += (dy_attractor / dist_attractor) * attractor_force

    # Wind force
    wind_x = math.sin(wind_phase + p["x"] * 0.1) * wind_strength * 1.5
    wind_y = math.cos(wind_phase + p["y"] * 0.1) * wind_strength * 0.3
    wind_turbulence = math.sin(wind_phase * 2.1 + p["id"] * 0.7) * wind_strength * 0.8
    fx += wind_x + wind_turbulence
    fy += wind_y

    return (fx, fy)


def update_particle_velocities(ps, forces, drag_val):
    """Update particle velocities with forces and drag - FynX computed variable"""
    return [
        {**p, "vx": (p["vx"] + fx) * drag_val, "vy": (p["vy"] + fy) * drag_val}
        for p, (fx, fy) in zip(ps, forces)
    ]


def update_particle_positions(ps):
    """Update particle positions - FynX computed variable"""
    return [{**p, "x": p["x"] + p["vx"], "y": p["y"] + p["vy"]} for p in ps]


def handle_particle_collisions(ps, w, h, damp):
    """Handle boundary collisions - FynX computed variable"""
    result = []
    for p in ps:
        p_copy = p.copy()
        p_copy["hit"] = False

        min_x, max_x = 2, w - 2
        min_y, max_y = 2, h - 2

        if p["x"] < min_x:
            p_copy["x"], p_copy["vx"], p_copy["hit"] = min_x, abs(p["vx"]) * damp, True
            p_copy["vx"] += random.uniform(0.1, 0.3)
        elif p["x"] > max_x:
            p_copy["x"], p_copy["vx"], p_copy["hit"] = max_x, -abs(p["vx"]) * damp, True
            p_copy["vx"] -= random.uniform(0.1, 0.3)

        if p["y"] < min_y:
            p_copy["y"], p_copy["vy"], p_copy["hit"] = min_y, abs(p["vy"]) * damp, True
            p_copy["vx"] += random.uniform(-0.5, 0.5)
        elif p["y"] > max_y:
            p_copy["y"], p_copy["vy"], p_copy["hit"] = max_y, -abs(p["vy"]) * damp, True
            p_copy["vy"] -= random.uniform(0.2, 0.8)
            p_copy["vx"] += random.uniform(-0.3, 0.3)

        result.append(p_copy)
    return result


def update_particle_properties(ps):
    """Update trails and derived properties - FynX computed variable"""
    result = []
    for p in ps:
        p_copy = p.copy()

        # Update trail
        velocity = math.sqrt(p["vx"] ** 2 + p["vy"] ** 2)
        p_copy["trail"] = [(int(p["x"]), int(p["y"]), 0, velocity)] + [
            (x, y, age + 1, vel) for x, y, age, vel in p["trail"] if age < 12
        ]

        # Calculate energy
        p_copy["energy"] = 0.5 * p["mass"] * velocity**2

        # Dynamic color based on energy and velocity
        energy_norm = min(1.0, p["energy"] / 20.0)
        velocity_norm = min(1.0, velocity / 5.0)

        if energy_norm > 0.7:
            p_copy["color"] = random.choice(
                ["bright_red", "bright_yellow", "bright_cyan", "bright_magenta"]
            )
        elif velocity_norm > 0.6:
            p_copy["color"] = random.choice(
                ["bright_blue", "cyan", "blue", "bright_cyan"]
            )
        # Otherwise keep existing color

        result.append(p_copy)
    return result


# ============================================================================
# Particle System
# ============================================================================

# ============================================================================
# Enhanced Particle Rendering Constants
# ============================================================================

PARTICLE_CHARS = "  · ∘ ○ ◎ ◉ ◆ ✦ ✦ ✦ ◆ ◉ ◎ ○ ∘ ·  "
PARTICLE_COLORS = [
    "dim",
    "grey50",
    "grey70",
    "white",
    "bright_white",
    "bright_yellow",
    "yellow",
    "bright_red",
    "red",
    "bright_magenta",
]

TRAIL_CHARS = [" ", "ˑ", "·", "◦", "○"]
TRAIL_COLORS = ["dim", "grey50", "grey70", "white", "bright_white"]


# ============================================================================
# Pure Reactive Physics Store
# ============================================================================


class Physics(Store):
    """All physics state as FynX observables - truly reactive!"""

    # Core time state (only this changes!)
    frame = observable(0)

    # Physics parameters (adjusted for more horizontal movement)
    gravity = observable(0.2)  # Reduced gravity
    damping = observable(0.8)  # Stronger damping on bounces
    drag = observable(0.99)  # Slightly more drag
    repel_radius = observable(8.0)
    repel_force = observable(0.5)

    # Dynamic force field (moving attractor)
    attractor_strength = observable(0.02)

    # Environmental forces
    wind_strength = observable(0.01)

    # Boundaries (set by UI)
    width = observable(70)
    height = observable(21)

    # Visual effects
    collision_flash = observable(0.0)

    # ============================================================================
    # Pure FynX Derived Physics - All computed reactively!
    # ============================================================================

    # Initial particle state (will be set once)
    _initial_particles = observable([])

    # Dynamic attractor position (derived from frame)
    attractor_phase = frame >> (lambda f: f * 0.05)
    attractor_x = (width + attractor_phase) >> (
        lambda w, phase: w / 2 + math.cos(phase) * (w / 4)
    )
    attractor_y = (height + attractor_phase) >> (
        lambda h, phase: h / 2 + math.sin(phase * 0.7) * (h / 3)
    )

    # Wind phase (derived from frame)
    wind_phase = frame >> (lambda f: f * 0.02)

    # Physics state - will be set by step_physics
    _particle_state = observable([])
    particles = _particle_state  # Public alias

    # Step 1: Compute forces for all particles
    particle_forces = (
        particles
        + repel_radius
        + repel_force
        + attractor_x
        + attractor_y
        + attractor_strength
        + wind_phase
        + wind_strength
        + gravity
    ) >> compute_particle_forces

    # Step 2: Update velocities with forces and drag
    particle_velocities = (
        particles + particle_forces + drag
    ) >> update_particle_velocities

    # Step 3: Update positions
    particle_positions = particle_velocities >> update_particle_positions

    # Step 4: Handle boundary collisions
    particle_collisions = (
        particle_positions + width + height + damping
    ) >> handle_particle_collisions

    # Step 5: Update trails and derived properties (this is the NEXT state)
    next_particles = particle_collisions >> update_particle_properties

    # Derived observables - FynX auto-memoizes with >>
    energy = (particles + gravity) >> (
        lambda ps, g: (
            sum(
                0.5 * p["mass"] * (p["vx"] ** 2 + p["vy"] ** 2)
                + p["mass"] * g * (25 - p["y"])
                for p in ps
            )
            if ps
            else 0.0
        )
    )

    speed = particles >> (
        lambda ps: (
            sum(math.sqrt(p["vx"] ** 2 + p["vy"] ** 2) for p in ps) / len(ps)
            if ps
            else 0.0
        )
    )

    center = particles >> (
        lambda ps: (
            (
                sum(p["x"] * p["mass"] for p in ps) / sum(p["mass"] for p in ps),
                sum(p["y"] * p["mass"] for p in ps) / sum(p["mass"] for p in ps),
            )
            if ps
            else (0.0, 0.0)
        )
    )

    collisions = particles >> (lambda ps: sum(1 for p in ps if p.get("hit", False)))

    particle_count = particles >> (lambda ps: len(ps))


# ============================================================================
# Particle System
# ============================================================================

# ============================================================================
# Enhanced Particle Rendering Constants
# ============================================================================

PARTICLE_CHARS = "  · ∘ ○ ◎ ◉ ◆ ✦ ✦ ✦ ◆ ◉ ◎ ○ ∘ ·  "
PARTICLE_COLORS = [
    "dim",
    "grey50",
    "grey70",
    "white",
    "bright_white",
    "bright_yellow",
    "yellow",
    "bright_red",
    "red",
    "bright_magenta",
]

TRAIL_CHARS = [" ", "ˑ", "·", "◦", "○"]
TRAIL_COLORS = ["dim", "grey50", "grey70", "white", "bright_white"]


def spawn_particles(count=15):
    """Initialize particle swarm with enhanced properties"""
    w, h = Physics.width.value, Physics.height.value

    initial_particles = [
        {
            "id": i,
            "x": random.uniform(5, w - 5),
            "y": random.uniform(3, h - 3),
            "vx": random.uniform(-3, 3),
            "vy": random.uniform(-2, 2),
            "mass": random.uniform(0.8, 1.5),
            "radius": random.uniform(0.8, 1.2),
            "color": random.choice(
                [
                    "bright_red",
                    "red",
                    "bright_yellow",
                    "yellow",
                    "bright_green",
                    "green",
                    "bright_cyan",
                    "cyan",
                    "bright_blue",
                    "blue",
                    "bright_magenta",
                    "magenta",
                    "orange1",
                    "dark_orange",
                    "yellow",
                    "green",
                    "blue",
                    "bright_blue",
                    "purple",
                    "magenta",
                    "bright_magenta",
                    "red",
                ]
            ),
            "trail": [],  # Now: list of (x, y, age, velocity)
            "hit": False,
            "energy": 0.0,  # Kinetic energy
            "last_hit_frame": -100,  # For collision flash timing
        }
        for i in range(count)
    ]

    Physics._particle_state.set(initial_particles)


# FynX Physics Step
def step_physics():
    """Advance physics by one frame using FynX dataflow"""
    current_particles = Physics.particles.value
    current_frame = Physics.frame.value

    # The physics computation happens automatically through the derived observables!
    # next_particles computes the next state from the current particles
    next_state = Physics.next_particles.value

    if next_state:
        # Update particle state with the computed next state
        Physics._particle_state.set(next_state)

    # Track collisions for flash effects
    new_collisions = sum(
        1
        for p in Physics.particles.value
        if p["hit"] and current_frame - p.get("last_hit_frame", -100) > 5
    )
    if new_collisions > 0:
        Physics.collision_flash = min(1.0, Physics.collision_flash.value + 0.3)
    else:
        Physics.collision_flash = max(0, Physics.collision_flash.value - 0.05)

    # Advance frame - this triggers recomputation of all derived physics!
    Physics.frame = current_frame + 1


# ============================================================================
# Pure UI Components
# ============================================================================


class ParticleField(ReactiveComponent):
    """Render enhanced particle field with glow, trails, and atmosphere"""

    def get_dependencies(self):
        return [
            Physics.particles,
            Physics.center,
            Physics.attractor_x,
            Physics.attractor_y,
            Physics.frame,
        ]

    def render_component(self):
        # Expensive computation - only runs when dependencies actually change!
        particles = Physics.particles.value
        center_x, center_y = Physics.center.value
        attractor_x = Physics.attractor_x.value
        attractor_y = Physics.attractor_y.value
        frame = Physics.frame.value
        w = self.props.get("width", 70)
        h = self.props.get("height", 21)

        # Initialize grid with background atmosphere
        grid = [[(" ", None) for _ in range(w)] for _ in range(h)]

        # Add subtle animated starfield background
        for y in range(h):
            for x in range(w):
                if random.random() < 0.003:  # Sparse stars
                    star_phase = (frame + x + y) % 60
                    star_brightness = abs(math.sin(star_phase * 0.1))
                    if star_brightness > 0.7:
                        grid[y][x] = ("·", "dim")

        # Add pulsing gravity well effect around center of mass
        for y in range(h):
            for x in range(w):
                dist_from_center = math.hypot(x - center_x, y - center_y)
                if dist_from_center < 12:
                    pulse = 0.3 + 0.2 * math.sin(frame * 0.15 + dist_from_center * 0.3)
                    if random.random() < pulse / (dist_from_center + 1) * 0.1:
                        grid[y][x] = ("ˑ", "bright_cyan")

        # Render particle trails first (background layer)
        for p in particles:
            for tx, ty, age, velocity in p["trail"]:
                if 0 <= tx < w and 0 <= ty < h:
                    fade = 1.0 - (age / 12.0)
                    trail_idx = min(len(TRAIL_CHARS) - 1, int(fade * len(TRAIL_CHARS)))
                    if trail_idx > 0:  # Skip empty space
                        char = TRAIL_CHARS[trail_idx]
                        # Trail color based on particle's current color with fading
                        base_color = p["color"]
                        # Dim the color based on trail age
                        if "bright_" in base_color:
                            trail_color = (
                                base_color.replace("bright_", "")
                                if fade < 0.5
                                else base_color
                            )
                        else:
                            trail_color = "dim" if fade < 0.3 else base_color
                        grid[ty][tx] = (char, trail_color)

        # Render particles (foreground layer)
        for p in particles:
            px, py = int(p["x"]), int(p["y"])
            if 0 <= px < w and 0 <= py < h:
                # Energy and velocity based rendering
                velocity = math.sqrt(p["vx"] ** 2 + p["vy"] ** 2)
                energy_norm = min(1.0, p["energy"] / 15.0)
                velocity_norm = min(1.0, velocity / 4.0)
                intensity = (energy_norm + velocity_norm) / 2.0

                # Collision flash effect
                if p["hit"] and frame - p.get("last_hit_frame", -100) < 3:
                    intensity = min(1.0, intensity + 0.5)

                particle_idx = int(intensity * (len(PARTICLE_CHARS) - 1))
                char = PARTICLE_CHARS[particle_idx]

                # Dynamic color based on energy and collision state
                if p["hit"] and frame - p.get("last_hit_frame", -100) < 5:
                    color = "bright_white" if frame % 2 == 0 else "bright_yellow"
                else:
                    color_idx = min(
                        len(PARTICLE_COLORS) - 1, int(intensity * len(PARTICLE_COLORS))
                    )
                    color = PARTICLE_COLORS[color_idx]

                grid[py][px] = (char, color)

        # Render special markers
        cx, cy = int(center_x), int(center_y)
        if 0 <= cx < w and 0 <= cy < h:
            grid[cy][cx] = ("+", "bright_white")

        ax, ay = int(attractor_x), int(attractor_y)
        if 0 <= ax < w and 0 <= ay < h:
            grid[ay][ax] = ("⚫", "bright_yellow")

        # Convert grid to RichText lines
        lines = []
        for y in range(h):
            line = RichText()
            for x in range(w):
                char, color = grid[y][x]
                line.append(char, style=color)
            lines.append(line)

        return Col(children=lines).render()


class StatsPanel(ReactiveComponent):
    """Enhanced stats display with animated borders and sparklines"""

    def get_dependencies(self):
        return [
            Physics.energy,
            Physics.speed,
            Physics.center,
            Physics.collisions,
            Physics.collision_flash,
            Physics.frame,
        ]

    def render_component(self):
        # Only re-compute when stats actually change!
        energy = Physics.energy.value
        speed = Physics.speed.value
        cx, cy = Physics.center.value
        hits = Physics.collisions.value
        collision_flash = Physics.collision_flash.value
        frame = Physics.frame.value

        # Animated border cycling
        border_phase = (frame // 15) % 4
        border_chars = ["─", "╸", "╺", "━"]
        animated_border = border_chars[border_phase]

        # Energy sparkline (keep last 10 values)
        if not hasattr(self, "energy_history"):
            self.energy_history = []
        self.energy_history.append(energy)
        self.energy_history = self.energy_history[-10:]
        sparkline = "".join(
            "▁▂▃▄▅▆▇█"[min(7, int(v / 20))] for v in self.energy_history
        )

        # Collision flash effect on borders
        energy_border = "bright_yellow" if collision_flash > 0.1 else "yellow"
        hits_border = "bright_red" if collision_flash > 0.1 else "red"

        return Row(
            equal=True,
            children=[
                Box(
                    title="Energy",
                    border=energy_border,
                    padding=(0, 1),
                    children=[
                        Text(text=f"{energy:.1f} J", color="yellow", bold=True),
                        Text(text=sparkline, color="dim"),
                    ],
                ),
                Box(
                    title="Speed",
                    border="green",
                    padding=(0, 1),
                    children=[Text(text=f"{speed:.2f} u/s", color="green", bold=True)],
                ),
                Box(
                    title="Center",
                    border="magenta",
                    padding=(0, 1),
                    children=[
                        Text(text=f"({cx:.0f}, {cy:.0f})", color="magenta", bold=True)
                    ],
                ),
                Box(
                    title="Hits",
                    border=hits_border,
                    padding=(0, 1),
                    children=[Text(text=str(hits), color="red", bold=True)],
                ),
            ],
        ).render()


class Dashboard(ReactiveComponent):
    """Enhanced main dashboard with animated borders and atmosphere"""

    def get_dependencies(self):
        return [Physics.frame, Physics.particle_count, Physics.collision_flash]

    def render_component(self):
        # Only re-compute layout when frame or particle count changes!
        frame = Physics.frame.value
        count = Physics.particle_count.value
        collision_flash = Physics.collision_flash.value

        # Calculate field dimensions with constant display height
        from rich.console import Console

        console = Console()
        console_width, console_height = console.size

        # Keep both physics and display height constant for absolutely no jolting
        physics_height = Physics.height.value  # Constant at 21
        field_width = max(40, console_width - 8)
        field_height = (
            physics_height + 1
        )  # Constant display height: physics height + border

        # Update physics boundaries (only width is reactive, height stays constant)
        Physics.width = field_width - 2
        # Physics.height stays constant at 21

        # Animated main border
        main_border_phase = (frame // 20) % 4
        main_border_chars = ["━", "╸", "╺", "─"]
        main_border = main_border_chars[main_border_phase]

        # Flash main border on collisions
        main_border_style = "bright_cyan" if collision_flash > 0.1 else "cyan"
        sim_border_style = "bright_blue" if collision_flash > 0.1 else "blue"
        perf_border_style = "bright_green" if collision_flash > 0.1 else "green"

        return Box(
            title="FynX Particle Physics - Terminal Art",
            border=main_border_style,
            padding=(1, 2),
            children=[
                H1(text="O(affected) Reactive Computation"),
                Spacer(height=1),
                Box(
                    title=f"Simulation ({count} particles, frame {frame:,})",
                    border=sim_border_style,
                    padding=(0, 1),
                    children=[ParticleField(width=field_width, height=field_height)],
                ),
                Spacer(height=1),
                Box(
                    title="FynX O(affected) Performance",
                    border=perf_border_style,
                    padding=(1, 2),
                    children=[
                        StatsPanel(),
                        Spacer(height=1),
                        PerformanceStats(),
                    ],
                ),
                Line(width=field_width),
                Text(
                    text="Press Ctrl+C to exit - Watch the magic",
                    color="dim",
                    italic=True,
                ),
            ],
        ).render()


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    print("FynX Particle Physics - Terminal Art")
    print("Enhanced with glow, trails, atmosphere & reactive magic\n")

    spawn_particles(15)

    import os

    # Default to unlimited FPS, cap to 24 if LIMITED_FPS env var is set
    fps = 24 if os.environ.get("LIMITED_FPS") else 0

    app = Dashboard()
    renderer = render(app, fps=fps)

    # Track collisions for bell sound
    last_collision_flash = 0.0

    with renderer.start():
        try:
            while True:
                # Run physics steps
                for _ in range(3):
                    step_physics()

                # Terminal bell on new collisions
                current_flash = Physics.collision_flash.value
                if current_flash > 0.1 and last_collision_flash <= 0.1:
                    print("\a", end="", flush=True)  # Terminal bell
                last_collision_flash = current_flash

                time.sleep(1 / 20)

        except KeyboardInterrupt:
            print(f"\n{Physics.frame.value:,} frames rendered!")
            print("Terminal art created with FynX reactive magic!")
            print("Terminal art created with FynX reactive magic!")
