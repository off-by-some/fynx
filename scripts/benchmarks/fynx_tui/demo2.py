"""
FynX 3D Wireframe Engine - O(affected) Reactive Graphics

A fully reactive 3D rendering system with:
- Real-time 3D rotation (Euler angles)
- Perspective projection
- Dynamic lighting system
- Particle effects synchronized to rotation
- Geometric shapes: cube, pyramid, diamond
- All powered by FynX's fine-grained reactivity!

All performance and status stats consolidated for maximum 3D render space.
Every visual element is a derived observable - rotation changes propagate
through the entire rendering pipeline with zero wasted computation.
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
# 3D Math Utilities
# ============================================================================


def rotate_x(point, angle):
    """Rotate point around X axis"""
    x, y, z = point
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x, y * cos_a - z * sin_a, y * sin_a + z * cos_a)


def rotate_y(point, angle):
    """Rotate point around Y axis"""
    x, y, z = point
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a)


def rotate_z(point, angle):
    """Rotate point around Z axis"""
    x, y, z = point
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)


def project(point, width, height, fov=None, viewer_distance=3):
    """Project 3D point to 2D screen coordinates"""
    if fov is None:
        # Auto-scale FOV based on canvas height to ensure shapes fit
        fov = height * 0.8  # Reduced from 1.2 to prevent over-scaling

    x, y, z = point
    factor = fov / (viewer_distance + z)
    x_proj = x * factor + width / 2
    y_proj = -y * factor + height / 2

    return (int(x_proj), int(y_proj), z)


# ============================================================================
# Shape Generation
# ============================================================================


def get_shape_vertices(shape_type, scale):
    """Generate vertices for different shapes"""
    s = scale

    if shape_type == "cube":
        return [
            (-s, -s, -s),
            (s, -s, -s),
            (s, s, -s),
            (-s, s, -s),  # Back face
            (-s, -s, s),
            (s, -s, s),
            (s, s, s),
            (-s, s, s),  # Front face
        ]

    elif shape_type == "pyramid":
        return [
            (-s, -s, -s),
            (s, -s, -s),
            (s, -s, s),
            (-s, -s, s),  # Base
            (0, s * 1.5, 0),  # Apex
        ]

    elif shape_type == "star":
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            # Outer points
            points.append((math.cos(angle) * s, math.sin(angle) * s, 0))
            # Inner points
            inner_angle = angle + math.pi / 5
            points.append(
                (math.cos(inner_angle) * s * 0.4, math.sin(inner_angle) * s * 0.4, 0)
            )
        # Add depth
        return points + [(x, y, z + s * 0.5) for x, y, z in points]

    elif shape_type == "diamond":
        return [
            (0, s * 1.5, 0),  # Top
            (-s, 0, -s),
            (s, 0, -s),
            (s, 0, s),
            (-s, 0, s),  # Middle ring
            (0, -s * 1.5, 0),  # Bottom
        ]

    return []  # Default empty


def get_shape_edges(shape_type):
    """Get edge connections for each shape"""
    if shape_type == "cube":
        return [
            # Back face
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Front face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Connecting edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

    elif shape_type == "pyramid":
        return [
            # Base
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Edges to apex
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        ]

    elif shape_type == "star":
        edges = []
        # Base layer: connect outer points (pentagon)
        outer_points = [0, 2, 4, 6, 8]  # Every other point starting at 0
        for i in range(5):
            edges.append((outer_points[i], outer_points[(i + 1) % 5]))

        # Base layer: connect inner points (pentagon)
        inner_points = [1, 3, 5, 7, 9]  # Every other point starting at 1
        for i in range(5):
            edges.append((inner_points[i], inner_points[(i + 1) % 5]))

        # Base layer: connect outer to adjacent inner points
        for i in range(5):
            outer_idx = outer_points[i]
            inner_idx = inner_points[i]
            edges.append((outer_idx, inner_idx))

        # Connect front to back (all points)
        for i in range(10):
            edges.append((i, i + 10))

        return edges

    elif shape_type == "diamond":
        return [
            # Top to middle ring
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            # Middle ring
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            # Middle to bottom
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
        ]

    return []


# ============================================================================
# Reactive 3D Graphics Store
# ============================================================================


class Graphics3D(Store):
    """All 3D graphics state as FynX observables"""

    # Rotation angles (animated over time)
    angle_x = observable(0.0)
    angle_y = observable(0.0)
    angle_z = observable(0.0)

    # Rotation speeds
    speed_x = observable(0.02)
    speed_y = observable(0.03)
    speed_z = observable(0.01)

    # Scene parameters
    scale = observable(1.1)  # Base scale for good visibility
    shape_type = observable("cube")  # 'cube', 'pyramid', 'diamond'

    # Lighting
    light_angle = observable(0.0)
    light_intensity = observable(1.0)

    # Particle system
    particle_count = observable(20)
    particle_speed = observable(0.05)

    # Display settings (reactive)
    width = observable(60)
    height = observable(30)

    # Frame counter
    frame = observable(0)

    # Derived: Light position (rotates around scene)
    light_pos = light_angle >> (
        lambda angle: (
            math.cos(angle) * 3,
            math.sin(angle) * 2,
            math.sin(angle * 0.5) * 2,
        )
    )

    # Derived: Effective scale based on canvas size
    effective_scale = (scale + width + height) >> (
        lambda s, w, h: min(s, min(w, h) / 40)  # Scale down for smaller canvases
    )

    # Derived: Shape vertices based on shape_type and effective scale
    base_vertices = (shape_type + effective_scale) >> (
        lambda shape, s: get_shape_vertices(shape, s)
    )

    # Derived: Rotated vertices
    rotated_vertices = (base_vertices + angle_x + angle_y + angle_z) >> (
        lambda verts, ax, ay, az: [
            rotate_z(rotate_y(rotate_x(v, ax), ay), az) for v in verts
        ]
    )

    # Note: width and height are updated dynamically by Scene3D component
    # The projection will use whatever width/height values are current


# ============================================================================
# Particle System
# ============================================================================


class Particle:
    def __init__(self):
        self.reset()

    def reset(self):
        angle = random.uniform(0, 2 * math.pi)
        self.x = math.cos(angle) * 2
        self.y = math.sin(angle) * 2
        self.z = random.uniform(-1, 1)
        self.vx = math.cos(angle) * 0.02
        self.vy = math.sin(angle) * 0.02
        self.vz = random.uniform(-0.01, 0.01)
        self.life = 1.0
        self.color = random.choice(
            ["cyan", "magenta", "yellow", "green", "blue", "red"]
        )

    def update(self, speed_multiplier=1.0):
        self.x += self.vx * speed_multiplier
        self.y += self.vy * speed_multiplier
        self.z += self.vz * speed_multiplier
        self.life -= 0.01

        if self.life <= 0 or abs(self.x) > 4 or abs(self.y) > 4:
            self.reset()


PARTICLES = [Particle() for _ in range(30)]


# ============================================================================
# Animation Loop
# ============================================================================


def animate_scene():
    """Update all animations - called every frame"""
    # Rotate shape
    Graphics3D.angle_x = Graphics3D.angle_x.value + Graphics3D.speed_x.value
    Graphics3D.angle_y = Graphics3D.angle_y.value + Graphics3D.speed_y.value
    Graphics3D.angle_z = Graphics3D.angle_z.value + Graphics3D.speed_z.value

    # Rotate light
    Graphics3D.light_angle = Graphics3D.light_angle.value + 0.05

    # Update particles
    speed = Graphics3D.particle_speed.value
    for particle in PARTICLES:
        particle.update(speed)

    # Increment frame
    Graphics3D.frame = Graphics3D.frame.value + 1

    # Cycle shapes every 300 frames
    if Graphics3D.frame.value % 300 == 0:
        shapes = ["cube", "pyramid", "diamond"]  # Removed star
        current = Graphics3D.shape_type.value
        if current == "star":
            # If currently showing star, switch to cube
            Graphics3D.shape_type = "cube"
        else:
            next_idx = (shapes.index(current) + 1) % len(shapes)
            Graphics3D.shape_type = shapes[next_idx]


# ============================================================================
# Reactive 3D Renderer
# ============================================================================


class WireframeRenderer(ReactiveComponent):
    """Renders 3D wireframe - recomputes when rotation, shape, or dimensions change"""

    def get_dependencies(self):
        return [
            Graphics3D.rotated_vertices,
            Graphics3D.shape_type,
            Graphics3D.light_pos,
            Graphics3D.width,
            Graphics3D.height,
        ]

    def render_component(self):
        rotated_verts = Graphics3D.rotated_vertices.value
        shape = Graphics3D.shape_type.value
        light = Graphics3D.light_pos.value

        # Use canvas dimensions from props (should match Graphics3D dimensions)
        w = self.props.get("width", 60)
        h = self.props.get("height", 30)

        # Project vertices using canvas dimensions
        verts = [project(v, w, h) for v in rotated_verts]

        # Create canvas
        canvas = [[" " for _ in range(w)] for _ in range(h)]
        depth_buffer = [[float("inf") for _ in range(w)] for _ in range(h)]
        colors = [[None for _ in range(w)] for _ in range(h)]

        # Draw particles first (background)
        for p in PARTICLES:
            if p.life > 0:
                x_proj, y_proj, z_proj = project((p.x, p.y, p.z), w, h)
                if 0 <= x_proj < w and 0 <= y_proj < h:
                    intensity = p.life
                    char = "·" if intensity > 0.5 else "•" if intensity > 0.3 else "∘"
                    if z_proj < depth_buffer[y_proj][x_proj]:
                        canvas[y_proj][x_proj] = char
                        depth_buffer[y_proj][x_proj] = z_proj
                        colors[y_proj][x_proj] = p.color

        # Draw wireframe edges
        edges = get_shape_edges(shape)

        for edge in edges:
            if edge[0] < len(verts) and edge[1] < len(verts):
                p1, p2 = verts[edge[0]], verts[edge[1]]

                # Calculate edge brightness based on distance from light
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                mid_z = (p1[2] + p2[2]) / 2

                # Distance from light (in 3D space before projection)
                dx = mid_x - w / 2
                dy = mid_y - h / 2
                dz = mid_z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz) / 20
                brightness = max(0.3, 1.0 - dist)

                # Choose character based on brightness
                if brightness > 0.8:
                    char = "█"
                    color = "bright_white"
                elif brightness > 0.6:
                    char = "▓"
                    color = "white"
                elif brightness > 0.4:
                    char = "▒"
                    color = "cyan"
                else:
                    char = "░"
                    color = "blue"

                # Draw line
                draw_line(canvas, depth_buffer, colors, p1, p2, char, color)

        # Draw vertices as highlights
        for vert in verts:
            x, y, z = vert
            if 0 <= x < w and 0 <= y < h:
                if z < depth_buffer[y][x]:
                    canvas[y][x] = "◉"
                    colors[y][x] = "yellow"
                    depth_buffer[y][x] = z

        # Convert canvas to Rich Text objects
        lines = []
        for y in range(h):
            line = RichText()
            for x in range(w):
                char = canvas[y][x]
                color = colors[y][x]
                line.append(char, style=color)
            lines.append(line)

        return Col(children=lines).render()


def draw_line(canvas, depth_buffer, colors, p1, p2, char, color):
    """Bresenham's line algorithm with depth buffering"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    steps = max(dx, dy)
    if steps == 0:
        return

    x, y = x1, y1

    while True:
        # Interpolate depth
        if steps > 0:
            t = math.sqrt((x - x1) ** 2 + (y - y1) ** 2) / steps
            z = z1 + (z2 - z1) * t
        else:
            z = z1

        # Draw pixel with depth testing
        if 0 <= x < len(canvas[0]) and 0 <= y < len(canvas):
            if z < depth_buffer[y][x]:
                canvas[y][x] = char
                depth_buffer[y][x] = z
                colors[y][x] = color

        if x == x2 and y == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


# ============================================================================
# UI Components
# ============================================================================


class ShapeInfo(ReactiveComponent):
    """Display current shape info"""

    def get_dependencies(self):
        return [Graphics3D.shape_type, Graphics3D.frame]

    def render_component(self):
        shape = Graphics3D.shape_type.value
        frame = Graphics3D.frame.value

        shape_names = {
            "cube": "📦 CUBE",
            "pyramid": "🔺 PYRAMID",
            "diamond": "💎 DIAMOND",
        }

        return Box(
            title=shape_names.get(shape, shape.upper()),
            border="cyan",
            padding=(0, 1),
            children=[Text(text=f"Frame: {frame:,}", color="cyan", bold=True)],
        ).render()


class RotationStats(ReactiveComponent):
    """Display rotation angles"""

    def get_dependencies(self):
        return [Graphics3D.angle_x, Graphics3D.angle_y, Graphics3D.angle_z]

    def render_component(self):
        ax = Graphics3D.angle_x.value % (2 * math.pi)
        ay = Graphics3D.angle_y.value % (2 * math.pi)
        az = Graphics3D.angle_z.value % (2 * math.pi)

        return Row(
            equal=True,
            children=[
                Box(
                    title="🔄 X-Axis",
                    border="red",
                    padding=(0, 1),
                    children=[
                        Text(text=f"{math.degrees(ax):.0f}°", color="red", bold=True)
                    ],
                ),
                Box(
                    title="🔄 Y-Axis",
                    border="green",
                    padding=(0, 1),
                    children=[
                        Text(text=f"{math.degrees(ay):.0f}°", color="green", bold=True)
                    ],
                ),
                Box(
                    title="🔄 Z-Axis",
                    border="blue",
                    padding=(0, 1),
                    children=[
                        Text(text=f"{math.degrees(az):.0f}°", color="blue", bold=True)
                    ],
                ),
            ],
        ).render()


class Scene3D(ReactiveComponent):
    """Main 3D scene component"""

    def get_dependencies(self):
        return [Graphics3D.frame]

    def render_component(self):
        from rich.console import Console

        console = Console()
        width, height = console.size

        field_width = max(40, width - 8)
        field_height = max(12, int(height * 0.40))

        Graphics3D.width = field_width - 2
        Graphics3D.height = field_height - 1

        return Box(
            title="🌌 FynX 3D Wireframe Engine",
            border="magenta",
            padding=(1, 2),
            children=[
                H1(text="O(affected) 3D Graphics Pipeline"),
                Spacer(height=1),
                Box(
                    title="🎬 Real-Time Render",
                    border="cyan",
                    padding=(0, 1),
                    children=[
                        WireframeRenderer(width=field_width, height=field_height)
                    ],
                ),
                Spacer(height=1),
                Box(
                    title="⚡ FynX Reactive Pipeline",
                    border="yellow",
                    padding=(1, 2),
                    children=[
                        ShapeInfo(),
                        Spacer(height=1),
                        RotationStats(),
                        Spacer(height=1),
                        PerformanceStats(),
                    ],
                ),
                Line(width=field_width),
                Text(
                    text="Press Ctrl+C to exit • Shapes cycle every 300 frames",
                    color="dim",
                    italic=True,
                ),
            ],
        ).render()


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    print("🌌 FynX 3D Wireframe Engine")
    print("Real-time 3D graphics with O(affected) reactivity\n")

    import os

    fps = 24 if os.environ.get("LIMITED_FPS") else 0

    app = Scene3D()
    renderer = render(app, fps=fps)

    with renderer.start():
        try:
            while True:
                animate_scene()
                time.sleep(1 / 60)  # 60 Hz physics updates

        except KeyboardInterrupt:
            print(f"\n✨ Rendered {Graphics3D.frame.value:,} frames of 3D graphics!")
            print("Every transformation was O(affected) reactive! 🚀")
