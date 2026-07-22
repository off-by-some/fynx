#!/usr/bin/env python3
"""
FynX Gravity Loom
=================

A one-file terminal particle sandbox that makes FynX's reactive graph visible.

The demo keeps a strict split:

* The imperative shell owns terminal I/O, keyboard input, time, and particles.
* ``Controls`` contains the tiny source observables that input is allowed to
  mutate.
* ``build_reactive_graph`` derives force fields, colors, status data, and
  diagnostics from those sources with FynX operators.

The graph intentionally forms a diamond:

    phase -> shared -> sin -> orbit -> force field
                     └ cos ┘        └ render state

The status line reports how many times the joined ``orbit`` node recomputed for
each source update. A healthy graph should stay near 1.00, not 2.00.

Controls
--------
q / Esc     quit
Space       pause
r           reseed particles
1 / 2 / 3   galaxy / vortex / binary modes
[ or ,      decrease gravity
] or . or / increase gravity
- / +       decrease / increase particle count
t           toggle trails
c           toggle color
Arrow keys  steer the field

Install:
    python -m pip install fynx

Run:
    python examples/gravity.py
"""

from __future__ import annotations

import math
import os
import random
import select
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeAlias

try:
    from fynx import Observable, Store, observable, reactive
except ImportError as error:
    raise SystemExit(
        "FynX is required.\n\n" "Install it with:\n" "    python -m pip install fynx"
    ) from error


# Braille dots for local coordinates:
# (0,0)=1 (0,1)=2 (0,2)=3 (0,3)=7
# (1,0)=4 (1,1)=5 (1,2)=6 (1,3)=8
BRAILLE = ((1, 2, 4, 64), (8, 16, 32, 128))
RESET = "\x1b[0m"

TARGET_FPS = 60.0
DEFAULT_GRAVITY = 0.105
DEFAULT_PARTICLES = 5_000
PARTICLE_STEP = 1_000
MIN_PARTICLES = 1_000
MAX_PARTICLES = 20_000
MIN_GRAVITY = 0.01
MAX_GRAVITY = 0.5
GRAVITY_DOWN = 0.84
GRAVITY_UP = 1.19
MAX_STEER = 0.7
STEER_STEP = 0.05
SOFTENING = 0.015

Point: TypeAlias = tuple[int, int]
Attractor: TypeAlias = tuple[float, float]


class Mode(Enum):
    """Simulation modes selected by the numeric keys."""

    GALAXY = 1
    VORTEX = 2
    BINARY = 3


MODE_BY_KEY = {
    "1": Mode.GALAXY,
    "2": Mode.VORTEX,
    "3": Mode.BINARY,
}


@dataclass(slots=True)
class Particle:
    """Mutable particle state owned by the simulation shell."""

    x: float
    y: float
    vx: float
    vy: float
    age: float


@dataclass(frozen=True, slots=True)
class ForceField:
    """Derived physics policy for one animation frame."""

    ax: float
    ay: float
    bx: float
    by: float
    gravity: float
    swirl: float
    drag: float

    @property
    def attractors(self) -> tuple[Attractor, Attractor]:
        return (self.ax, self.ay), (self.bx, self.by)


@dataclass(frozen=True, slots=True)
class ModeProfile:
    """Mode-specific constants used by the reactive force and color graph."""

    name: str
    gravity_scale: float
    swirl: float
    drag: float
    ax_scale: float
    ay_scale: float
    bx_scale: float
    by_scale: float
    colors: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class RenderState:
    """Derived terminal styling for one animation frame."""

    particle_colors: tuple[int, ...]
    trail_color: int
    attractor_color: int
    status_color: int
    color_enabled: bool
    trails_enabled: bool


@dataclass(frozen=True, slots=True)
class FrameModel:
    """Single derived snapshot consumed by rendering and status formatting."""

    field: ForceField
    render: RenderState
    profile: ModeProfile
    gravity: float
    particle_target: int
    running: bool
    effects_enabled: bool


@dataclass(slots=True)
class GraphCounters:
    """
    Instrument the diamond graph.

    These counters are intentionally stateful: their only purpose is to expose
    recomputation behavior in the status line.
    """

    shared: int = 0
    diamond: int = 0

    def count_shared(self, value: float) -> float:
        self.shared += 1
        return value

    def count_diamond(self, sine: float, cosine: float) -> tuple[float, float]:
        self.diamond += 1
        return sine, cosine


@dataclass(frozen=True, slots=True)
class ReactiveGraph:
    """Named handles for the FynX graph built from ``Controls``."""

    frame: Observable[FrameModel]
    field: Observable[ForceField]
    render: Observable[RenderState]
    profile: Observable[ModeProfile]
    running: Observable[bool]
    effects_enabled: Observable[bool]
    counters: GraphCounters


@dataclass(slots=True)
class Telemetry:
    """Imperative frame metrics kept outside the reactive graph."""

    propagation_ms: float = 0.0
    smooth_fps: float = TARGET_FPS
    diamond_ratio: float = 1.0
    last_diamond: int = 0

    def record_graph_update(self, counters: GraphCounters, elapsed_ms: float) -> None:
        self.propagation_ms = elapsed_ms
        delta = counters.diamond - self.last_diamond
        self.last_diamond = counters.diamond
        self.diamond_ratio = self.diamond_ratio * 0.9 + delta * 0.1

    def record_frame_time(self, elapsed: float) -> None:
        if elapsed > 0:
            self.smooth_fps = self.smooth_fps * 0.92 + (1.0 / elapsed) * 0.08


@dataclass(slots=True)
class FrameBuffer:
    """Flat Braille framebuffer; avoids per-row lists in the hot render path."""

    cols: int
    rows: int
    width: int = field(init=False)
    height: int = field(init=False)
    cells: list[int] = field(init=False)
    colors: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.cols <= 0 or self.rows <= 0:
            raise ValueError("FrameBuffer dimensions must be positive")

        self.width = self.cols * 2
        self.height = self.rows * 4
        self.cells = [0] * (self.cols * self.rows)
        self.colors = [0] * (self.cols * self.rows)

    def plot(self, px: int, py: int, color: int) -> None:
        cx = px // 2
        cy = py // 4
        if 0 <= cx < self.cols and 0 <= cy < self.rows:
            index = cy * self.cols + cx
            self.cells[index] |= BRAILLE[px & 1][py & 3]
            self.colors[index] = color

    def text(self, color_enabled: bool) -> str:
        return self.colored_text() if color_enabled else self.plain_text()

    def colored_text(self) -> str:
        lines: list[str] = []
        for row_start in range(0, len(self.cells), self.cols):
            current_color = -1
            chars: list[str] = []
            for index in range(row_start, row_start + self.cols):
                dots = self.cells[index]
                if not dots:
                    chars.append(" ")
                    continue

                color = self.colors[index]
                if color != current_color:
                    chars.append(f"\x1b[38;5;{color}m")
                    current_color = color
                chars.append(chr(0x2800 + dots))

            if current_color != -1:
                chars.append(RESET)
            lines.append("".join(chars))

        return "\n".join(lines)

    def plain_text(self) -> str:
        return "\n".join(
            "".join(
                chr(0x2800 + dots) if dots else " "
                for dots in self.cells[row_start : row_start + self.cols]
            )
            for row_start in range(0, len(self.cells), self.cols)
        )


class Controls(Store):
    """
    Source observables for the demo.

    Keyboard input mutates only these values. Everything else in the simulation
    view is derived by FynX in ``build_reactive_graph``.
    """

    phase = observable(0.0)
    gravity = observable(DEFAULT_GRAVITY)
    mode = observable(Mode.GALAXY)
    steer_x = observable(0.0)
    steer_y = observable(0.0)
    color_enabled = observable(True)
    trails_enabled = observable(True)
    particle_target = observable(DEFAULT_PARTICLES)
    paused = observable(False)

    @classmethod
    def reset(cls) -> None:
        """Restore the public controls to their documented defaults."""

        cls.phase.set(0.0)
        cls.gravity.set(DEFAULT_GRAVITY)
        cls.mode.set(Mode.GALAXY)
        cls.steer_x.set(0.0)
        cls.steer_y.set(0.0)
        cls.color_enabled.set(True)
        cls.trails_enabled.set(True)
        cls.particle_target.set(DEFAULT_PARTICLES)
        cls.paused.set(False)

    @classmethod
    def advance_to_frame(cls, frame_no: int) -> None:
        if frame_no < 0:
            raise ValueError("frame number must be non-negative")
        cls.phase.set(frame_no / TARGET_FPS)

    @classmethod
    def set_mode(cls, mode: Mode) -> None:
        cls.mode.set(mode)

    @classmethod
    def scale_gravity(cls, factor: float) -> None:
        if factor <= 0:
            raise ValueError("gravity scale factor must be positive")
        cls.gravity.set(clamp(cls.gravity.value * factor, MIN_GRAVITY, MAX_GRAVITY))

    @classmethod
    def change_particles(cls, delta: int) -> None:
        cls.particle_target.set(
            int(clamp(cls.particle_target.value + delta, MIN_PARTICLES, MAX_PARTICLES))
        )

    @classmethod
    def steer(cls, dx: float, dy: float) -> None:
        cls.steer_x.set(clamp(cls.steer_x.value + dx, -MAX_STEER, MAX_STEER))
        cls.steer_y.set(clamp(cls.steer_y.value + dy, -MAX_STEER, MAX_STEER))

    @classmethod
    def toggle_pause(cls) -> None:
        cls.paused.set(not cls.paused.value)

    @classmethod
    def toggle_color(cls) -> None:
        cls.color_enabled.set(not cls.color_enabled.value)

    @classmethod
    def toggle_trails(cls) -> None:
        cls.trails_enabled.set(not cls.trails_enabled.value)


class ParticleCloud:
    """Owns particle allocation so source-state changes can resize it safely."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.particles: list[Particle] = []

    def reseed(self, count: int) -> None:
        if count < 0:
            raise ValueError("particle count must be non-negative")
        self.particles = seed_particles(count, self._rng)

    def resize_to(self, count: int) -> None:
        if count < 0:
            raise ValueError("particle count must be non-negative")

        current = len(self.particles)
        if count > current:
            self.particles.extend(seed_particles(count - current, self._rng))
        elif count < current:
            del self.particles[count:]


class Terminal:
    """ANSI terminal manager with non-blocking Unix keyboard input."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old: Optional[Any] = None

    def __enter__(self) -> "Terminal":
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise SystemExit("Run this script in an interactive terminal.")
        if os.name == "nt":
            raise SystemExit(
                "This one-file build targets macOS/Linux terminals. "
                "On Windows, run it inside WSL."
            )

        import termios
        import tty

        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        sys.stdout.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self.old is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()

    def keys(self) -> list[str]:
        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            char = os.read(self.fd, 1).decode(errors="ignore")
            if char == "\x1b":
                keys.append(self._escape_key())
            else:
                keys.append(char)
        return keys

    def _escape_key(self) -> str:
        if not select.select([sys.stdin], [], [], 0.002)[0]:
            return "ESC"

        sequence = os.read(self.fd, 2).decode(errors="ignore")
        return {
            "[A": "UP",
            "[B": "DOWN",
            "[C": "RIGHT",
            "[D": "LEFT",
        }.get(sequence, "ESC")

    @staticmethod
    def draw(text: str) -> None:
        sys.stdout.write("\x1b[H\x1b[J" + text)
        sys.stdout.flush()


@dataclass(frozen=True, slots=True)
class InputResult:
    """Outcome of a single key press."""

    quit: bool = False
    clear_trails: bool = False


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        raise ValueError("clamp lower bound must be <= upper bound")
    return min(high, max(low, value))


def seed_particles(count: int, rng: random.Random) -> list[Particle]:
    """Seed a rotating disk, which quickly blooms into orbital structure."""

    if count < 0:
        raise ValueError("particle count must be non-negative")
    return [new_particle(rng) for _ in range(count)]


def new_particle(rng: random.Random) -> Particle:
    angle = rng.random() * math.tau
    radius = math.sqrt(rng.random()) * 0.93 + 0.02
    speed = 0.17 + 0.18 * (1.0 - radius)
    return Particle(
        x=math.cos(angle) * radius,
        y=math.sin(angle) * radius * 0.55,
        vx=-math.sin(angle) * speed + rng.uniform(-0.018, 0.018),
        vy=math.cos(angle) * speed * 1.7 + rng.uniform(-0.018, 0.018),
        age=rng.random() * 10.0,
    )


def recycle_particle(particle: Particle, rng: random.Random) -> None:
    """Move an escaped particle back to the outer disk without allocating."""

    angle = rng.random() * math.tau
    radius = rng.uniform(0.65, 1.0)
    particle.x = math.cos(angle) * radius
    particle.y = math.sin(angle) * radius * 0.55
    particle.vx = -math.sin(angle) * 0.2
    particle.vy = math.cos(angle) * 0.34
    particle.age = 0.0


def integrate(
    particles: list[Particle],
    dt: float,
    field: ForceField,
    rng: random.Random,
) -> None:
    """Advance particles through two softened gravity wells plus a curl field."""

    if dt < 0:
        raise ValueError("dt must be non-negative")

    damp = max(0.0, 1.0 - field.drag * dt)
    sqrt = math.sqrt

    for particle in particles:
        dx = field.ax - particle.x
        dy = field.ay - particle.y
        r2 = dx * dx + dy * dy + SOFTENING
        inv = field.gravity / (r2 * sqrt(r2))
        fx = dx * inv
        fy = dy * inv

        dx = field.bx - particle.x
        dy = field.by - particle.y
        r2 = dx * dx + dy * dy + SOFTENING
        inv = -0.72 * field.gravity / (r2 * sqrt(r2))
        fx += dx * inv
        fy += dy * inv

        fx += -particle.y * field.swirl
        fy += particle.x * field.swirl

        particle.vx = (particle.vx + fx * dt) * damp
        particle.vy = (particle.vy + fy * dt) * damp
        particle.x += particle.vx * dt
        particle.y += particle.vy * dt
        particle.age += dt

        if (
            particle.x * particle.x + particle.y * particle.y > 5.0
            or particle.age > 35.0
        ):
            recycle_particle(particle, rng)


def build_reactive_graph() -> ReactiveGraph:
    """
    Build the FynX graph for force policy, visual style, and diagnostics.

    ``Controls`` are the only source observables. The graph uses:
    * ``>>`` for transformations
    * ``+`` for value joins
    * ``~`` for the derived running state
    * ``|`` for a total "any visual effect enabled" boolean
    """

    counters = GraphCounters()

    shared_phase = Controls.phase >> counters.count_shared
    sine = shared_phase >> math.sin
    cosine = shared_phase >> math.cos
    orbit = (sine + cosine) >> counters.count_diamond

    profile = Controls.mode >> profile_for_mode
    running = ~Controls.paused
    effects_enabled = Controls.color_enabled | Controls.trails_enabled

    field = (
        orbit + profile + Controls.gravity + Controls.steer_x + Controls.steer_y
    ) >> (
        lambda sc, mode_profile, gravity, steer_x, steer_y: build_field(
            sine=sc[0],
            cosine=sc[1],
            profile=mode_profile,
            gravity=gravity,
            steer_x=steer_x,
            steer_y=steer_y,
        )
    )

    render = (
        orbit
        + profile
        + Controls.gravity
        + Controls.color_enabled
        + Controls.trails_enabled
    ) >> (
        lambda sc, mode_profile, gravity, color, trails: build_render_state(
            sine=sc[0],
            cosine=sc[1],
            profile=mode_profile,
            gravity=gravity,
            color_enabled=color,
            trails_enabled=trails,
        )
    )

    frame = (
        field
        + render
        + profile
        + Controls.gravity
        + Controls.particle_target
        + running
        + effects_enabled
    ) >> (
        lambda force_field, render_state, mode_profile, gravity, particle_target, is_running, has_effects: FrameModel(
            field=force_field,
            render=render_state,
            profile=mode_profile,
            gravity=gravity,
            particle_target=particle_target,
            running=is_running,
            effects_enabled=has_effects,
        )
    )

    return ReactiveGraph(
        frame=frame,
        field=field,
        render=render,
        profile=profile,
        running=running,
        effects_enabled=effects_enabled,
        counters=counters,
    )


def build_field(
    sine: float,
    cosine: float,
    profile: ModeProfile,
    gravity: float,
    steer_x: float,
    steer_y: float,
) -> ForceField:
    """Pure derived policy: modes change the entire force field coherently."""

    return ForceField(
        ax=profile.ax_scale * cosine + steer_x,
        ay=profile.ay_scale * sine + steer_y,
        bx=profile.bx_scale * cosine + steer_x,
        by=profile.by_scale * sine + steer_y,
        gravity=gravity * profile.gravity_scale,
        swirl=profile.swirl,
        drag=profile.drag,
    )


def profile_for_mode(mode: Mode) -> ModeProfile:
    """Map validated mode input to a stable simulation profile."""

    if mode is Mode.GALAXY:
        return ModeProfile(
            name="GALAXY",
            gravity_scale=1.00,
            swirl=0.020,
            drag=0.42,
            ax_scale=0.30,
            ay_scale=0.18,
            bx_scale=-0.36,
            by_scale=-0.22,
            colors=(45, 51, 87, 123, 159),
        )

    if mode is Mode.VORTEX:
        return ModeProfile(
            name="VORTEX",
            gravity_scale=0.34,
            swirl=0.115,
            drag=0.70,
            ax_scale=0.10,
            ay_scale=0.10,
            bx_scale=0.00,
            by_scale=0.00,
            colors=(83, 119, 155, 191, 227),
        )

    if mode is Mode.BINARY:
        return ModeProfile(
            name="BINARY",
            gravity_scale=1.25,
            swirl=-0.012,
            drag=0.30,
            ax_scale=0.42,
            ay_scale=0.28,
            bx_scale=-0.42,
            by_scale=-0.28,
            colors=(197, 203, 209, 215, 221),
        )

    raise ValueError(f"Unsupported mode: {mode!r}")


def build_render_state(
    sine: float,
    cosine: float,
    profile: ModeProfile,
    gravity: float,
    color_enabled: bool,
    trails_enabled: bool,
) -> RenderState:
    """Derive terminal styling from simulation state, not ad hoc globals."""

    pulse = int((sine + 1.0) * 1.5 + gravity * 8.0) % len(profile.colors)
    colors = profile.colors[pulse:] + profile.colors[:pulse]
    trail_color = 240 + int((cosine + 1.0) * 3.0)
    return RenderState(
        particle_colors=colors,
        trail_color=trail_color,
        attractor_color=231,
        status_color=colors[-1],
        color_enabled=color_enabled,
        trails_enabled=trails_enabled,
    )


def render(
    particles: list[Particle],
    old_points: list[Point],
    cols: int,
    rows: int,
    model: FrameModel,
) -> tuple[str, list[Point]]:
    """Render particles and the two attractors into a Braille framebuffer."""

    buffer = FrameBuffer(cols, rows)
    points: list[Point] = []
    particle_colors = model.render.particle_colors
    color_count = len(particle_colors)

    for particle in particles:
        px = int((particle.x * 0.48 + 0.5) * (buffer.width - 1))
        py = int((particle.y * 0.48 + 0.5) * (buffer.height - 1))
        points.append((px, py))
        color = particle_colors[
            int((particle.age * 7.0 + particle.y * 5.0) % color_count)
        ]
        buffer.plot(px, py, color)

    if model.render.trails_enabled:
        for px, py in old_points:
            buffer.plot(px, py, model.render.trail_color)

    for ax, ay in model.field.attractors:
        px = int((ax * 0.48 + 0.5) * (buffer.width - 1))
        py = int((ay * 0.48 + 0.5) * (buffer.height - 1))
        for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            buffer.plot(px + dx, py + dy, model.render.attractor_color)

    return buffer.text(model.render.color_enabled), points


def status_text(model: FrameModel, telemetry: Telemetry, particle_count: int) -> str:
    """Format the compact dashboard shown under the canvas."""

    color_prefix = (
        f"\x1b[38;5;{model.render.status_color}m" if model.render.color_enabled else ""
    )
    color_suffix = RESET if model.render.color_enabled else ""
    run_flag = "run" if model.running else "paused"
    trail_flag = "trail" if model.render.trails_enabled else "clean"
    color_flag = "color" if model.render.color_enabled else "mono"
    effects_flag = "fx" if model.effects_enabled else "plain"

    return (
        f"{color_prefix} FynX Gravity Loom  {model.profile.name}  {run_flag}  "
        f"{particle_count:,}/{model.particle_target:,} particles  "
        f"{telemetry.smooth_fps:4.0f} fps  {trail_flag}/{color_flag}/{effects_flag}\n"
        f" reactive frame {telemetry.propagation_ms:6.3f} ms  "
        f"diamond recomputes/update {telemetry.diamond_ratio:4.2f}  "
        f"gravity {model.gravity:.3f}{color_suffix}\n"
        " 1/2/3 mode  arrows steer  [,/]/. gravity  +/- particles  "
        "c color  t trails  space pause  r reset  q quit"
    )


def handle_key(key: str, cloud: ParticleCloud) -> InputResult:
    """Apply one key press to the source observables or particle cloud."""

    if key in ("q", "Q", "ESC"):
        return InputResult(quit=True)

    if key == " ":
        Controls.toggle_pause()
    elif key in ("r", "R"):
        cloud.reseed(Controls.particle_target.value)
        return InputResult(clear_trails=True)
    elif key in MODE_BY_KEY:
        Controls.set_mode(MODE_BY_KEY[key])
    elif key in ("[", ","):
        Controls.scale_gravity(GRAVITY_DOWN)
    elif key in ("]", ".", "/"):
        Controls.scale_gravity(GRAVITY_UP)
    elif key in ("t", "T"):
        Controls.toggle_trails()
    elif key in ("c", "C"):
        Controls.toggle_color()
    elif key == "+":
        Controls.change_particles(PARTICLE_STEP)
    elif key == "-":
        Controls.change_particles(-PARTICLE_STEP)
    elif key == "LEFT":
        Controls.steer(-STEER_STEP, 0.0)
    elif key == "RIGHT":
        Controls.steer(STEER_STEP, 0.0)
    elif key == "UP":
        Controls.steer(0.0, -STEER_STEP)
    elif key == "DOWN":
        Controls.steer(0.0, STEER_STEP)

    return InputResult()


def terminal_canvas_size() -> tuple[int, int]:
    size = shutil.get_terminal_size((100, 32))
    return max(20, size.columns), max(8, size.lines - 4)


def main() -> None:
    Controls.reset()
    rng = random.Random()
    graph = build_reactive_graph()
    telemetry = Telemetry()
    cloud = ParticleCloud(rng)
    old_points: list[Point] = []
    frame_no = 0

    @reactive(Controls.particle_target)
    def keep_particle_count(target: int) -> None:
        cloud.resize_to(target)

    # Touch the model once so telemetry starts after the initial graph realization.
    _ = graph.frame.value
    telemetry.last_diamond = graph.counters.diamond

    with Terminal() as terminal:
        previous = time.perf_counter()

        while True:
            frame_start = time.perf_counter()
            dt = min(frame_start - previous, 0.05)
            previous = frame_start

            for key in terminal.keys():
                result = handle_key(key, cloud)
                if result.quit:
                    return
                if result.clear_trails:
                    old_points = []

            model = graph.frame.value
            if model.running:
                before = time.perf_counter()
                frame_no += 1
                Controls.advance_to_frame(frame_no)
                model = graph.frame.value
                telemetry.record_graph_update(
                    graph.counters,
                    (time.perf_counter() - before) * 1000.0,
                )
                integrate(cloud.particles, dt, model.field, rng)

            cols, canvas_rows = terminal_canvas_size()
            picture, old_points = render(
                cloud.particles,
                old_points,
                cols,
                canvas_rows,
                model,
            )
            terminal.draw(
                picture + "\n" + status_text(model, telemetry, len(cloud.particles))
            )

            elapsed = time.perf_counter() - frame_start
            delay = (1.0 / TARGET_FPS) - elapsed
            if delay > 0:
                time.sleep(delay)
            telemetry.record_frame_time(time.perf_counter() - frame_start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
