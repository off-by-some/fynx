#!/usr/bin/env python3
"""
FynX Is You: a tiny reactive terminal puzzle.

Move text to change the rules:

    FYNX IS YOU
    ROCK IS PUSH
    WALL IS FLAG
    FLAG IS WIN

The game is deliberately small so the FynX graph is visible.  ``Game`` owns a
few source observables; everything else is derived with ``.then()``, ``>>``,
``+``, ``~``, ``|``, and ``&``.  Two ``@reactive`` functions handle side
effects, so the loop only reads keys and changes source state.

Controls: arrows/WASD move, r reset, n next, c color, q/Esc quit.
"""

from __future__ import annotations

import os
import re
import select
import sys
import termios
import textwrap
import time
import tty
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping

from fynx import Store, observable, reactive

RESET = "\x1b[0m"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FRAME_SECONDS = 1 / 60
CELL = 5
DEFAULT_VIEWPORT = (100, 30)


@dataclass(frozen=True, slots=True)
class Kind:
    glyph: str
    color: int


KINDS: Mapping[str, Kind] = {
    "fynx": Kind("@", 220),
    "rock": Kind("O", 246),
    "wall": Kind("#", 250),
    "flag": Kind("F", 118),
    "water": Kind("~", 45),
}
NOUNS = {name.upper(): name for name in KINDS}
PROPS = frozenset({"YOU", "PUSH", "STOP", "WIN", "SINK"})
WORDS = frozenset(NOUNS) | {"IS"} | PROPS
GLYPHS = {kind.glyph: name for name, kind in KINDS.items()}
COLORS = {name.upper(): kind.color for name, kind in KINDS.items()} | {
    "IS": 231,
    "YOU": 213,
    "PUSH": 81,
    "STOP": 196,
    "WIN": 118,
    "SINK": 45,
}
DIRECTIONS = {
    "UP": (0, -1),
    "W": (0, -1),
    "DOWN": (0, 1),
    "S": (0, 1),
    "LEFT": (-1, 0),
    "A": (-1, 0),
    "RIGHT": (1, 0),
    "D": (1, 0),
}
ORTHOGONAL_STEPS = frozenset(DIRECTIONS.values())


@dataclass(frozen=True, slots=True, order=True)
class Point:
    x: int
    y: int

    def step(self, dx: int, dy: int) -> "Point":
        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True, slots=True)
class Piece:
    id: int
    kind: str
    cell: Point
    text: bool = False

    def step(self, dx: int, dy: int) -> "Piece":
        return Piece(self.id, self.kind, self.cell.step(dx, dy), self.text)


@dataclass(frozen=True, slots=True)
class Level:
    title: str
    hint: str
    width: int
    height: int
    pieces: tuple[Piece, ...]


@dataclass(frozen=True, slots=True)
class RuleBook:
    """Rules derived from adjacent text.

    ``props`` answers "what is this kind?" and ``aliases`` answers "what does
    this kind become?"  Invalid level text fails during parsing instead of being
    ignored.
    """

    active: tuple[str, ...]
    props: Mapping[str, frozenset[str]]
    aliases: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class Layout:
    """Terminal-dependent board layout matching the original demo."""

    cell_width: int
    cell_height: int
    split: bool
    board_width: int
    panel_width: int


@dataclass(frozen=True, slots=True)
class Consequences:
    """Human-readable semantic summary derived from the active rule graph."""

    controlled: tuple[str, ...]
    pushable: tuple[str, ...]
    solid: tuple[str, ...]
    stopped: tuple[str, ...]
    winners: tuple[str, ...]
    sinkers: tuple[str, ...]
    transformed: tuple[str, ...]


class MoveKind(Enum):
    """Domain events that can change the player's status line."""

    READY = "ready"
    RESET = "reset"
    BLOCKED = "blocked"
    MOVED = "moved"
    TEXT_MOVED = "text_moved"
    RULES_CHANGED = "rules_changed"


@dataclass(frozen=True, slots=True)
class MoveReport:
    """Result of the latest command.

    Rule deltas are present only for ``RULES_CHANGED`` reports.  This keeps
    status text derivable and avoids reactions that synchronize message state.
    """

    kind: MoveKind
    gained_rules: tuple[str, ...] = ()
    lost_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is not MoveKind.RULES_CHANGED and (
            self.gained_rules or self.lost_rules
        ):
            raise ValueError("Only RULES_CHANGED reports may include rule deltas.")

    @classmethod
    def ready(cls) -> "MoveReport":
        return cls(MoveKind.READY)

    @classmethod
    def reset(cls) -> "MoveReport":
        return cls(MoveKind.RESET)

    @classmethod
    def blocked(cls) -> "MoveReport":
        return cls(MoveKind.BLOCKED)

    @classmethod
    def moved(cls, *, text_moved: bool = False) -> "MoveReport":
        return cls(MoveKind.TEXT_MOVED if text_moved else MoveKind.MOVED)

    @classmethod
    def rules_changed(
        cls,
        gained_rules: Iterable[str],
        lost_rules: Iterable[str],
    ) -> "MoveReport":
        return cls(
            MoveKind.RULES_CHANGED,
            tuple(sorted(gained_rules)),
            tuple(sorted(lost_rules)),
        )


@dataclass(frozen=True, slots=True)
class GameState:
    """Single atomic source value for player-visible state.

    A command replaces the whole state at once.  That prevents the screen
    reaction from seeing half-updated combinations of pieces and status.
    """

    level_index: int
    pieces: tuple[Piece, ...]
    last_move: MoveReport
    colored: bool
    viewport: tuple[int, int]


def level(title: str, hint: str, drawing: str) -> Level:
    """Parse a whitespace-separated ASCII level.

    Raises:
        ValueError: unknown tokens or an empty drawing.
    """

    rows = [line.split() for line in textwrap.dedent(drawing).strip().splitlines()]
    if not rows or any(not row for row in rows):
        raise ValueError(f"{title!r} must contain non-empty rows.")

    width = max(map(len, rows))
    pieces: list[Piece] = []
    next_id = 1
    for y, row in enumerate(rows):
        for x, token in enumerate(row + ["."] * (width - len(row))):
            if token == ".":
                continue
            if token in GLYPHS:
                pieces.append(Piece(next_id, GLYPHS[token], Point(x, y)))
            elif token in WORDS:
                pieces.append(Piece(next_id, token, Point(x, y), text=True))
            else:
                raise ValueError(f"Unknown token {token!r} in {title!r} at {x}, {y}.")
            next_id += 1
    return Level(title, hint, width, len(rows), tuple(pieces))


LEVELS = (
    level(
        "Tutorial: two rules, one corridor",
        "Make ROCK IS PUSH, then make FLAG IS WIN. The corridor needs both.",
        """
        . . . . . . . . . . . . .
        . FYNX IS YOU . ROCK IS . PUSH . . .
        . . . . . . . . . . . . .
        . FLAG IS . WIN . . . . . . . .
        . . . # # # # . . . . . .
        . @ . O F . # . . . . . .
        . . . # # # # . . . . . .
        . . . . . . . . . . . . .
        """,
    ),
    level(
        "Remote control",
        "Move YOU out of FYNX IS YOU, then make the remote flag winnable.",
        """
        . . . . . . ~ . . . . . .
        . ROCK IS . . . ~ . FLAG IS . WIN .
        . . . . . . ~ . . . . . .
        . FYNX IS YOU . . ~ . . . . . .
        . . . . . . ~ . . . . . .
        . @ . . . . ~ . . O F . .
        . . . . . . ~ . . . . . .
        . . . . . . ~ . . . . . .
        . . . . . . ~ . . . . . .
        """,
    ),
    level(
        "Turn the wall into the goal",
        "Make FLAG IS WIN, then rewrite WALL IS STOP into WALL IS FLAG.",
        """
        . . . . . . . . . . . . .
        . FYNX IS YOU . WALL IS STOP . . . . .
        . . . . . . . . # . . . .
        . . . . . . . FLAG # . . . .
        . . . . . . . . # . . . .
        . @ . . . . . . # . . . .
        . . . . . . . . # . . . .
        . FLAG IS . WIN . . . . . . . .
        . . . . . . . . . . . . .
        """,
    ),
    level(
        "Make the river inherit WIN",
        "Make ROCK pushable, clear the word lane, then make WATER IS FLAG.",
        """
        . . . . . . . . . . . . . . .
        . FYNX IS YOU . ROCK IS . PUSH . FLAG IS . WIN
        . . . . . . . . . . . . . . .
        . . . . . . . O . . . . . . .
        . . . . . . . . . . . . . . .
        . . . . @ . FLAG . . WATER IS . . .
        . . . . . . . . . ~ ~ ~ ~ ~ .
        . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . .
        """,
    ),
)


def at(pieces: Iterable[Piece], cell: Point) -> tuple[Piece, ...]:
    return tuple(piece for piece in pieces if piece.cell == cell)


def inside(cell: Point, board: Level) -> bool:
    return 0 <= cell.x < board.width and 0 <= cell.y < board.height


def rules_from(pieces: tuple[Piece, ...]) -> RuleBook:
    """Read horizontal and vertical ``NOUN IS TARGET`` phrases."""

    words = {piece.cell: piece.kind for piece in pieces if piece.text}
    props: dict[str, set[str]] = {kind: set() for kind in KINDS}
    aliases: dict[str, set[str]] = {kind: set() for kind in KINDS}
    active: list[str] = []

    for cell, word in words.items():
        if word not in NOUNS:
            continue
        for dx, dy in ((1, 0), (0, 1)):
            is_word = words.get(cell.step(dx, dy))
            target = words.get(cell.step(dx * 2, dy * 2))
            if is_word != "IS":
                continue
            subject = NOUNS[word]
            if target in PROPS:
                props[subject].add(target)
                active.append(f"{word} IS {target}")
            elif target in NOUNS:
                aliases[subject].add(NOUNS[target])
                active.append(f"{word} IS {target}")

    return RuleBook(
        active=tuple(sorted(active)),
        props={kind: frozenset(values) for kind, values in props.items()},
        aliases={
            kind: tuple(sorted(values)) for kind, values in aliases.items() if values
        },
    )


def aliases_for(kind: str, rules: RuleBook) -> frozenset[str]:
    seen = {kind}
    pending = [kind]
    while pending:
        for alias in rules.aliases.get(pending.pop(), ()):
            if alias not in seen:
                seen.add(alias)
                pending.append(alias)
    return frozenset(seen)


def shown_kind(kind: str, rules: RuleBook) -> str:
    return rules.aliases.get(kind, (kind,))[0]


def piece_props(piece: Piece, rules: RuleBook) -> frozenset[str]:
    if piece.text:
        return frozenset({"PUSH"})

    found: set[str] = set()
    for kind in aliases_for(piece.kind, rules):
        found.update(rules.props.get(kind, frozenset()))
    return frozenset(found)


def pushable(piece: Piece, rules: RuleBook) -> bool:
    return "PUSH" in piece_props(piece, rules)


def passable(piece: Piece, rules: RuleBook) -> bool:
    return bool(piece_props(piece, rules) & {"YOU", "WIN", "SINK"})


def blocking(piece: Piece, rules: RuleBook) -> bool:
    props = piece_props(piece, rules)
    return "STOP" in props or (
        not pushable(piece, rules) and not passable(piece, rules)
    )


def push_chain(
    pieces: tuple[Piece, ...],
    rules: RuleBook,
    board: Level,
    cell: Point,
    dx: int,
    dy: int,
    seen: frozenset[int] = frozenset(),
) -> frozenset[int] | None:
    """Return pieces forced to move, or ``None`` when the path is blocked."""

    if not inside(cell, board):
        return None

    moving: set[int] = set()
    for piece in at(pieces, cell):
        if piece.id in seen:
            continue
        if blocking(piece, rules):
            return None
        if pushable(piece, rules):
            after = push_chain(
                pieces, rules, board, piece.cell.step(dx, dy), dx, dy, seen | {piece.id}
            )
            if after is None:
                return None
            moving.update(after)
            moving.add(piece.id)
    return frozenset(moving)


def sink(pieces: tuple[Piece, ...], rules: RuleBook) -> tuple[Piece, ...]:
    doomed: set[int] = set()
    for sinker in pieces:
        if "SINK" not in piece_props(sinker, rules):
            continue
        victims = tuple(
            piece
            for piece in at(pieces, sinker.cell)
            if piece.id != sinker.id and not piece.text
        )
        if victims:
            doomed.add(sinker.id)
            doomed.update(piece.id for piece in victims)
    return tuple(piece for piece in pieces if piece.id not in doomed)


def solved(pieces: tuple[Piece, ...], rules: RuleBook) -> bool:
    players = tuple(piece for piece in pieces if "YOU" in piece_props(piece, rules))
    winners = tuple(piece for piece in pieces if "WIN" in piece_props(piece, rules))
    return any(player.cell == winner.cell for player in players for winner in winners)


def step(
    pieces: tuple[Piece, ...], rules: RuleBook, board: Level, dx: int, dy: int
) -> tuple[Piece, ...]:
    moving: set[int] = set()
    for player in tuple(
        piece for piece in pieces if "YOU" in piece_props(piece, rules)
    ):
        chain = push_chain(pieces, rules, board, player.cell.step(dx, dy), dx, dy)
        if chain is not None:
            moving.add(player.id)
            moving.update(chain)

    moved = tuple(
        piece.step(dx, dy) if piece.id in moving else piece for piece in pieces
    )
    return sink(moved, rules)


def graph(
    pieces: tuple[Piece, ...],
    rules: RuleBook,
    has_actor_rule: bool,
    has_goal_rule: bool,
    needs_rule: bool,
    is_solved: bool,
) -> tuple[str, ...]:
    """Expose the live semantic graph in the side panel."""

    def kinds(prop: str) -> str:
        names = {
            shown_kind(piece.kind, rules).upper()
            for piece in pieces
            if not piece.text and prop in piece_props(piece, rules)
        }
        return ", ".join(sorted(names)) or "—"

    properties = tuple(
        f"{prop:<5}: {kinds(prop)}" for prop in ("YOU", "PUSH", "STOP", "WIN", "SINK")
    )
    booleans = (
        f"has_actor: {has_actor_rule}",
        f"has_goal : {has_goal_rule}",
        f"needs   : {needs_rule}",
        f"solved  : {is_solved}",
    )
    return (*properties, *booleans)


def current_level(index: int) -> Level:
    """Return a level or fail with context for an invalid index."""

    if index < 0 or index >= len(LEVELS):
        raise ValueError(f"Invalid level index: {index}")
    return LEVELS[index]


def state_level_index(state: GameState) -> int:
    return state.level_index


def state_pieces(state: GameState) -> tuple[Piece, ...]:
    return state.pieces


def state_last_move(state: GameState) -> MoveReport:
    return state.last_move


def state_colored(state: GameState) -> bool:
    return state.colored


def state_viewport(state: GameState) -> tuple[int, int]:
    return state.viewport


def has_rule_property(rules: RuleBook, prop: str) -> bool:
    return any(prop in values for values in rules.props.values())


def has_actor(rules: RuleBook) -> bool:
    return has_rule_property(rules, "YOU")


def has_goal(rules: RuleBook) -> bool:
    return has_rule_property(rules, "WIN")


def rule_notice(has_actor_rule: bool, has_goal_rule: bool, needs_rule: bool) -> str:
    if not needs_rule:
        return "Rules are live."
    missing = [
        name
        for name, active in (("YOU", has_actor_rule), ("WIN", has_goal_rule))
        if not active
    ]
    return "Missing rule: " + " and ".join(missing)


def text_changed(before: tuple[Piece, ...], after: tuple[Piece, ...]) -> bool:
    """Detect moved or removed text blocks by stable piece id."""

    after_by_id = {piece.id: piece for piece in after}
    return any(piece != after_by_id.get(piece.id) for piece in before if piece.text)


def report_move(
    before: tuple[Piece, ...],
    after: tuple[Piece, ...],
    old_rules: RuleBook,
    new_rules: RuleBook,
) -> MoveReport:
    """Summarize a command without mutating FynX state."""

    if before == after:
        return MoveReport.blocked()
    old = set(old_rules.active)
    new = set(new_rules.active)
    if old != new:
        return MoveReport.rules_changed(new - old, old - new)
    return MoveReport.moved(text_moved=text_changed(before, after))


def report_text(report: MoveReport) -> str:
    """Format the non-reactive command result for the status line."""

    match report.kind:
        case MoveKind.READY:
            return "Move text; the graph updates itself."
        case MoveKind.RESET:
            return "Level reset."
        case MoveKind.BLOCKED:
            return "Blocked."
        case MoveKind.MOVED:
            return "Moved."
        case MoveKind.TEXT_MOVED:
            return "Text moved; FynX recomputed the graph."
        case MoveKind.RULES_CHANGED:
            gained = ", ".join(report.gained_rules) or "—"
            lost = ", ".join(report.lost_rules) or "—"
            return f"Rules changed: +{gained} / -{lost}"
    raise ValueError(f"Unhandled move report kind: {report.kind!r}")


def status_line(report: MoveReport, notice: str, is_solved: bool) -> str:
    prefix = "WIN! press n" if is_solved else notice
    return f"{prefix}  {report_text(report)}"


def initial_state() -> GameState:
    board = current_level(0)
    return GameState(
        level_index=0,
        pieces=board.pieces,
        last_move=MoveReport.ready(),
        colored=True,
        viewport=DEFAULT_VIEWPORT,
    )


def paint(text: str, ansi: int, enabled: bool, *, bold: bool = False) -> str:
    """Apply an ANSI color when color mode is enabled."""

    if not enabled:
        return text
    weight = "1;" if bold else ""
    return f"\x1b[{weight}38;5;{ansi}m{text}{RESET}"


def visible_width(text: str) -> int:
    """Measure printable width without ANSI escape codes."""

    return len(ANSI_RE.sub("", text))


def center_visible(text: str, width: int) -> str:
    """Center text while ignoring ANSI escape codes."""

    padding = max(0, width - visible_width(text))
    left = padding // 2
    return " " * left + text + " " * (padding - left)


def pad_visible(text: str, width: int) -> str:
    """Right-pad text while ignoring ANSI escape codes."""

    return text + " " * max(0, width - visible_width(text))


def fit_text(text: str, width: int) -> str:
    """Truncate text to a visible width without hiding overflow."""

    if width <= 0:
        return ""
    plain = ANSI_RE.sub("", text)
    if len(plain) <= width:
        return text
    return plain[: max(1, width - 1)] + "~"


def odd_at_most(value: int, maximum: int) -> int:
    """Choose an odd cell width so labels stay visually centered."""

    capped = max(3, min(value, maximum))
    return capped if capped % 2 else capped - 1


def color_for_word(word: str) -> int:
    """Return the original palette color for an object or rule word."""

    return COLORS.get(word, 231)


def render_cell(
    pieces: tuple[Piece, ...],
    rules: RuleBook,
    color_enabled: bool,
    cell_width: int,
    cell_height: int,
) -> list[str]:
    """Render one board cell using the original boxed-cell style."""

    lines = [" " * cell_width for _ in range(cell_height)]
    center = cell_height // 2
    if not pieces:
        if cell_height >= 3:
            lines[center] = center_visible(paint(".", 238, color_enabled), cell_width)
        return lines

    text_pieces = tuple(piece for piece in pieces if piece.text)
    if text_pieces:
        word = text_pieces[-1].kind
        if cell_width >= 5:
            label = fit_text(word, cell_width - 2)
            rendered = center_visible(f"[{label}]", cell_width)
        else:
            rendered = center_visible(fit_text(word, cell_width), cell_width)
        lines[center] = paint(rendered, color_for_word(word), color_enabled, bold=True)
        return lines

    piece = pieces[-1]
    kind = shown_kind(piece.kind, rules)
    ansi = KINDS[kind].color
    lines[center] = paint(
        center_visible(KINDS[kind].glyph, cell_width),
        ansi,
        color_enabled,
        bold=True,
    )
    if cell_height >= 4:
        label = fit_text(kind.upper(), cell_width)
        lines[min(center + 1, cell_height - 1)] = paint(
            center_visible(label, cell_width),
            ansi,
            color_enabled,
        )
    return lines


def choose_layout(board: Level, viewport: tuple[int, int]) -> Layout:
    """Fit the board and side panel to the current terminal size."""

    columns, lines = viewport
    panel_width = min(44, max(28, columns // 4))
    split = columns >= 90
    available_columns = columns - panel_width - 5 if split else columns - 2
    available_lines = lines - 8
    cell_width = odd_at_most((available_columns - 2) // board.width, 11)
    cell_height = max(1, min(5, (available_lines - 2) // board.height))
    board_width = board.width * cell_width + 2
    return Layout(cell_width, cell_height, split, board_width, panel_width)


def render_board(
    board: Level,
    pieces: tuple[Piece, ...],
    rules: RuleBook,
    color_enabled: bool,
    layout: Layout,
) -> list[str]:
    """Render the original bordered ASCII board."""

    border = "+" + "-" * (board.width * layout.cell_width) + "+"
    rows = [border]
    for y in range(board.height):
        cells = [
            render_cell(
                at(pieces, Point(x, y)),
                rules,
                color_enabled,
                layout.cell_width,
                layout.cell_height,
            )
            for x in range(board.width)
        ]
        for line_index in range(layout.cell_height):
            rows.append("|" + "".join(cell[line_index] for cell in cells) + "|")
    rows.append(border)
    return rows


def render_box(
    title: str,
    lines: Iterable[str],
    width: int,
    color_enabled: bool,
) -> list[str]:
    """Render an original-style side-panel box."""

    inner_width = max(8, width - 4)
    heading = paint(
        f" {fit_text(title, inner_width - 2)} ",
        81,
        color_enabled,
        bold=True,
    )
    body = tuple(fit_text(line, inner_width) for line in lines)
    return [
        "+" + "-" * (width - 2) + "+",
        "|" + center_visible(heading, width - 2) + "|",
        "|" + "-" * (width - 2) + "|",
        *(f"| {pad_visible(line, inner_width)} |" for line in body),
        "+" + "-" * (width - 2) + "+",
    ]


def effects_from(pieces: tuple[Piece, ...], rules: RuleBook) -> Consequences:
    """Summarize rule consequences for the side panel."""

    def names_for(prop: str) -> tuple[str, ...]:
        names = {
            shown_kind(piece.kind, rules)
            for piece in pieces
            if not piece.text and prop in piece_props(piece, rules)
        }
        return tuple(sorted(names))

    solid = {
        shown_kind(piece.kind, rules)
        for piece in pieces
        if not piece.text and blocking(piece, rules)
    }
    transformed = tuple(
        f"{source}->{', '.join(targets)}"
        for source, targets in sorted(rules.aliases.items())
    )
    return Consequences(
        controlled=names_for("YOU"),
        pushable=names_for("PUSH"),
        solid=tuple(sorted(solid)),
        stopped=names_for("STOP"),
        winners=names_for("WIN"),
        sinkers=names_for("SINK"),
        transformed=transformed,
    )


def consequence_lines(effects: Consequences) -> tuple[str, ...]:
    """Format semantic consequences exactly where players can inspect them."""

    lines = (
        "YOU controls " + ", ".join(effects.controlled or ("nothing",)),
        "PUSH moves " + ", ".join(effects.pushable or ("nothing",)),
        "SOLID blocks " + ", ".join(effects.solid or ("nothing",)),
        "STOP blocks " + ", ".join(effects.stopped or ("nothing",)),
        "WIN is " + ", ".join(effects.winners or ("missing",)),
        "SINK removes " + ", ".join(effects.sinkers or ("nothing",)),
    )
    if not effects.transformed:
        return lines
    return (*lines, "identity: " + ", ".join(effects.transformed))


def reactive_graph_lines(live_graph: tuple[str, ...]) -> tuple[str, ...]:
    """Show both the FynX dataflow and the live derived values."""

    return (
        "state >> level / pieces / viewport",
        "level_index.then(current_level)",
        "pieces.then(rules_from)",
        "(pieces + rules) >> effects",
        "(~has_actor) | (~has_goal)",
        "level_index @ solved",
        "screen >> terminal.draw",
        *live_graph,
    )


def compact_graph_lines(
    rules: RuleBook,
    effects: Consequences,
    live_graph: tuple[str, ...],
) -> tuple[str, ...]:
    """Compress the side panel when the terminal is too narrow."""

    active = " | ".join(rules.active) if rules.active else "none"
    controlled = ", ".join(effects.controlled or ("nothing",))
    pushable = ", ".join(effects.pushable or ("nothing",))
    solid = ", ".join(effects.solid or ("nothing",))
    winners = ", ".join(effects.winners or ("missing",))
    return (
        "rules: " + active,
        f"effects: YOU={controlled}  PUSH={pushable}  SOLID={solid}  WIN={winners}",
        "graph: pieces -> rules -> effects -> screen",
        *live_graph,
    )


def join_columns(left: list[str], right: list[str], gap: int = 3) -> list[str]:
    """Join the board and side panels without counting ANSI bytes as width."""

    width = max(visible_width(line) for line in left)
    height = max(len(left), len(right))
    rows: list[str] = []
    for index in range(height):
        left_line = left[index] if index < len(left) else ""
        right_line = right[index] if index < len(right) else ""
        rows.append(f"{pad_visible(left_line, width)}{' ' * gap}{right_line}")
    return rows


def screen(
    level_index: int,
    board: Level,
    pieces: tuple[Piece, ...],
    rules: RuleBook,
    effects: Consequences,
    live_graph: tuple[str, ...],
    status: str,
    colored: bool,
    viewport: tuple[int, int],
) -> str:
    """Render the terminal screen with the original board aesthetics."""

    columns, lines = viewport
    layout = choose_layout(board, viewport)
    title = paint("FynX Is You", 220, colored, bold=True)
    header = f"{title}  Level {level_index + 1}/{len(LEVELS)}  {board.title}"
    board_lines = render_board(board, pieces, rules, colored, layout)
    panel_specs = (
        ("Active Rules", rules.active or ("No rule is currently formed",)),
        ("Consequences", consequence_lines(effects)),
        ("Reactive Graph", reactive_graph_lines(live_graph)),
    )

    panels: list[str] = []
    for panel_title, panel_lines in panel_specs:
        if panels:
            panels.append("")
        panels.extend(render_box(panel_title, panel_lines, layout.panel_width, colored))

    if layout.split:
        content = join_columns(board_lines, panels)
    else:
        compact_width = min(columns, max(layout.board_width, 62))
        compact_panel = render_box(
            "Live Rule Graph",
            compact_graph_lines(rules, effects, live_graph),
            compact_width,
            colored,
        )
        content = [*board_lines, "", *compact_panel]

    highlighted_status = paint(status, 118, colored, bold=True)
    footer = [
        "",
        highlighted_status,
        "arrows/WASD move  r reset  n next  c color  q quit",
    ]
    reserved_rows = 2 + len(footer)
    if len(content) + reserved_rows > lines:
        content = content[: max(0, lines - reserved_rows)]

    output = [header, "", *content, *footer]
    if visible_width(header) < columns:
        output[0] = center_visible(header, columns)
    if lines > len(output):
        output = [""] * max(0, (lines - len(output)) // 3) + output
    return "\n".join(line[:columns] if not colored else line for line in output[:lines])


class Game(Store):
    """Reactive game model.

    ``state`` is the only mutable source for gameplay.  Every class attribute
    below it is a FynX derivation, so the terminal screen is always a pure
    function of the current source value.
    """

    # Source: command handlers replace this atomically.
    state = observable(initial_state())

    # Slices: demonstrate ``>>`` as pure projection from a domain value.
    level_index = state >> state_level_index
    pieces = state >> state_pieces
    last_move = state >> state_last_move
    colored = state >> state_colored
    viewport = state >> state_viewport

    # Domain graph: parse rules, consequences, and boolean facts declaratively.
    board = level_index.then(current_level)
    rules = pieces.then(rules_from)
    effects = (pieces + rules) >> effects_from
    has_actor = rules >> has_actor
    has_goal = rules >> has_goal
    needs_rule = (~has_actor) | (~has_goal)
    notice = (has_actor + has_goal + needs_rule) >> rule_notice
    solved = (pieces + rules) >> solved
    solved_level = level_index @ solved
    graph = (pieces + rules + has_actor + has_goal + needs_rule + solved) >> graph
    status = (last_move + notice + solved) >> status_line

    # Render graph: one computed value feeds the terminal effect boundary.
    screen = (
        level_index
        + board
        + pieces
        + rules
        + effects
        + graph
        + status
        + colored
        + viewport
    ) >> screen

    @classmethod
    def load(cls, index: int) -> None:
        """Replace game state with a validated level."""

        board = current_level(index)
        current = cls.state.value
        cls.state.set(
            replace(
                current,
                level_index=index,
                pieces=board.pieces,
                last_move=MoveReport.reset(),
            )
        )

    @classmethod
    def next_level(cls) -> None:
        cls.load((cls.state.value.level_index + 1) % len(LEVELS))

    @classmethod
    def reset_level(cls) -> None:
        cls.load(cls.state.value.level_index)

    @classmethod
    def toggle_color(cls) -> None:
        current = cls.state.value
        cls.state.set(replace(current, colored=not current.colored))

    @classmethod
    def resize(cls, viewport: tuple[int, int]) -> None:
        current = cls.state.value
        if viewport != current.viewport:
            cls.state.set(replace(current, viewport=viewport))

    @classmethod
    def move(cls, dx: int, dy: int) -> None:
        """Apply one player command as a single immutable state replacement."""

        if (dx, dy) not in ORTHOGONAL_STEPS:
            raise ValueError(f"Invalid move vector: {(dx, dy)!r}")

        current = cls.state.value
        board = current_level(current.level_index)
        old_rules = rules_from(current.pieces)
        moved = step(current.pieces, old_rules, board, dx, dy)
        new_rules = rules_from(moved)
        cls.state.set(
            replace(
                current,
                pieces=moved,
                last_move=report_move(current.pieces, moved, old_rules, new_rules),
            )
        )


class Terminal:
    """POSIX terminal with raw input and guaranteed cleanup."""

    def __enter__(self) -> "Terminal":
        if os.name == "nt":
            raise SystemExit("Run this POSIX example inside WSL on Windows.")
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise SystemExit("Run this demo in an interactive terminal.")

        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        self.buffer = ""
        tty.setcbreak(self.fd)
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        sys.stdout.flush()
        return self

    def __exit__(self, *_: object) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()

    def keys(self) -> tuple[str, ...]:
        while select.select([sys.stdin], [], [], 0)[0]:
            self.buffer += os.read(self.fd, 32).decode("utf-8", errors="ignore")

        keys: list[str] = []
        while self.buffer:
            if self.buffer[:3] in ("\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"):
                keys.append(
                    {
                        "\x1b[A": "UP",
                        "\x1b[B": "DOWN",
                        "\x1b[C": "RIGHT",
                        "\x1b[D": "LEFT",
                    }[self.buffer[:3]]
                )
                self.buffer = self.buffer[3:]
            elif self.buffer.startswith("\x1b"):
                keys.append("ESC")
                self.buffer = self.buffer[1:]
            else:
                keys.append(self.buffer[0])
                self.buffer = self.buffer[1:]
        return tuple(keys)

    @staticmethod
    def size() -> tuple[int, int]:
        size = os.get_terminal_size()
        return size.columns, size.lines

    @staticmethod
    def draw(text: str) -> None:
        sys.stdout.write("\x1b[H\x1b[J" + text)
        sys.stdout.flush()


def run(terminal: Terminal) -> None:
    """Connect the reactive screen value to the terminal resource."""

    @reactive(Game.screen)
    def draw_screen(rendered: str) -> None:
        terminal.draw(rendered)

    try:
        while True:
            Game.resize(terminal.size())

            for key in terminal.keys():
                if key in {"q", "Q", "ESC"}:
                    return

                direction = DIRECTIONS.get(key.upper())
                if direction is not None:
                    Game.move(*direction)
                elif key in {"r", "R"}:
                    Game.reset_level()
                elif key in {"n", "N"}:
                    Game.next_level()
                elif key in {"c", "C"}:
                    Game.toggle_color()

            time.sleep(FRAME_SECONDS)
    finally:
        draw_screen.unsubscribe()


def main() -> None:
    with Terminal() as terminal:
        run(terminal)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
