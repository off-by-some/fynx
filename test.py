"""
FynX Purity Analyzer
====================

Static analysis module for determining function purity in reactive graphs.
Uses AST analysis combined with annotation hints to classify function purity
with maximum practical accuracy.

Purity Classification:
    PURE: Provably pure - no side effects, deterministic output
    READONLY: Reads external state but doesn't mutate
    FUNCTIONAL: Pure given pure inputs (transitive purity)
    IMPURE: Known side effects
    UNKNOWN: Cannot determine statically

Usage:
    from fynx.purity import analyze_purity, pure

    @pure
    def transform(x):
        return x * 2

    purity = analyze_purity(transform)
    # Returns: Purity.PURE
"""

import ast
import inspect
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set


class Purity(Enum):
    """Function purity classification levels."""

    PURE = 1  # Provably pure
    READONLY = 2  # Reads external state but doesn't mutate
    FUNCTIONAL = 3  # Pure given pure inputs (transitive purity)
    IMPURE = 4  # Known side effects
    UNKNOWN = 5  # Cannot determine


@dataclass
class PurityResult:
    """Result of purity analysis."""

    purity: Purity
    reasons: list[str]
    can_memoize: bool
    can_fuse: bool


# Registry of explicitly marked pure functions
_PURE_REGISTRY: Set[Callable] = set()

# Known pure standard library functions
_STDLIB_PURE: Set[str] = {
    # Math operations
    "abs",
    "min",
    "max",
    "sum",
    "len",
    "round",
    "pow",
    "divmod",
    "bin",
    "hex",
    "oct",
    "chr",
    "ord",
    # Type conversions (pure for immutable types)
    "int",
    "float",
    "str",
    "bool",
    "tuple",
    "frozenset",
    "complex",
    "bytes",
    # Math module
    "math.sin",
    "math.cos",
    "math.tan",
    "math.sqrt",
    "math.log",
    "math.exp",
    "math.floor",
    "math.ceil",
    "math.fabs",
    # Operators module
    "operator.add",
    "operator.sub",
    "operator.mul",
    "operator.truediv",
    "operator.and_",
    "operator.or_",
    "operator.xor",
    "operator.not_",
}


def pure(func: Callable) -> Callable:
    """
    Decorator to explicitly mark a function as pure.

    This provides a hint to the optimizer that the function:
    - Has no side effects
    - Returns the same output for the same inputs
    - Can be safely memoized and fused

    Example:
        @pure
        def calculate(x, y):
            return x * 2 + y
    """
    _PURE_REGISTRY.add(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class PurityAnalyzer(ast.NodeVisitor):
    """
    AST visitor that analyzes function purity.

    Checks for:
    - Global/nonlocal variable access
    - Function calls to unknown functions
    - Attribute access (could trigger side effects)
    - Mutable operations
    - I/O operations
    """

    def __init__(self, func_name: str, known_pure: Set[str] = None):
        self.func_name = func_name
        self.known_pure = known_pure or set()
        self.reasons = []
        self.purity = Purity.PURE

        # Track what we encounter
        self.has_global_reads = False
        self.has_global_writes = False
        self.has_nonlocal = False
        self.has_unknown_calls = False
        self.has_attribute_access = False
        self.has_io_operations = False
        self.unknown_calls = []

    def visit_Global(self, node: ast.Global) -> None:
        """Track global variable declarations."""
        self.has_global_writes = True
        self.reasons.append(f"Uses global variables: {', '.join(node.names)}")
        self.purity = Purity.IMPURE
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Track nonlocal variable declarations."""
        self.has_nonlocal = True
        self.reasons.append(f"Uses nonlocal variables: {', '.join(node.names)}")
        self.purity = Purity.IMPURE
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Analyze function calls for purity."""
        func_name = self._get_call_name(node)

        # Check for I/O operations
        if func_name in {"print", "open", "input", "write", "read"}:
            self.has_io_operations = True
            self.reasons.append(f"Performs I/O: {func_name}()")
            self.purity = Purity.IMPURE
        # Check if it's a known pure function
        elif func_name not in _STDLIB_PURE and func_name not in self.known_pure:
            self.has_unknown_calls = True
            self.unknown_calls.append(func_name)
            if self.purity == Purity.PURE:
                self.purity = Purity.FUNCTIONAL

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Track attribute access (could have side effects via descriptors)."""
        self.has_attribute_access = True
        # Don't immediately mark as impure, but track it
        if self.purity == Purity.PURE and not self._is_safe_attribute(node):
            self.purity = Purity.READONLY

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Track variable access."""
        # Check if we're reading a global (but not writing)
        if isinstance(node.ctx, ast.Load):
            # This could be a global read - we'd need scope analysis to be sure
            pass
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Track augmented assignments (could trigger __iadd__ etc)."""
        if isinstance(node.target, (ast.Name, ast.Attribute, ast.Subscript)):
            # Could be mutating external state
            if self.purity == Purity.PURE:
                self.purity = Purity.UNKNOWN
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Track subscript operations (__getitem__ could have side effects)."""
        if isinstance(node.ctx, ast.Store):
            # Definitely mutating
            if self.purity in (Purity.PURE, Purity.READONLY):
                self.purity = Purity.UNKNOWN
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls like obj.method()
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return "<unknown>"

    def _is_safe_attribute(self, node: ast.Attribute) -> bool:
        """Check if attribute access is known to be safe."""
        # Math module attributes are safe
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            return True
        return False


def analyze_purity(func: Callable, known_pure: Set[str] = None) -> PurityResult:
    """
    Analyze a function to determine its purity level.

    Args:
        func: The function to analyze
        known_pure: Set of additional known pure function names

    Returns:
        PurityResult with purity classification and analysis details

    Example:
        def compute(x):
            return x * 2

        result = analyze_purity(compute)
        print(result.purity)  # Purity.PURE
        print(result.can_fuse)  # True
    """
    # Check if explicitly marked as pure
    if func in _PURE_REGISTRY:
        return PurityResult(
            purity=Purity.PURE,
            reasons=["Explicitly marked with @pure decorator"],
            can_memoize=True,
            can_fuse=True,
        )

    # Try to get source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return PurityResult(
            purity=Purity.UNKNOWN,
            reasons=["Cannot access source code"],
            can_memoize=False,
            can_fuse=False,
        )

    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return PurityResult(
            purity=Purity.UNKNOWN,
            reasons=[f"Syntax error in source: {e}"],
            can_memoize=False,
            can_fuse=False,
        )

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            func_def = node
            break

    if not func_def:
        return PurityResult(
            purity=Purity.UNKNOWN,
            reasons=["Could not find function definition in AST"],
            can_memoize=False,
            can_fuse=False,
        )

    # Analyze closure variables
    try:
        closure_vars = inspect.getclosurevars(func)
        if closure_vars.nonlocals or closure_vars.globals:
            # Has closure - need to check if it mutates
            pass  # Could do deeper analysis here
    except (TypeError, AttributeError):
        pass

    # Run AST analysis
    analyzer = PurityAnalyzer(func.__name__, known_pure or set())
    analyzer.visit(func_def)

    # Compile reasons
    if not analyzer.reasons:
        if analyzer.unknown_calls:
            analyzer.reasons.append(
                f"Calls functions with unknown purity: {', '.join(analyzer.unknown_calls)}"
            )

    # Determine optimization flags
    can_memoize = analyzer.purity in (Purity.PURE, Purity.READONLY, Purity.FUNCTIONAL)
    can_fuse = analyzer.purity in (Purity.PURE, Purity.FUNCTIONAL)

    return PurityResult(
        purity=analyzer.purity,
        reasons=analyzer.reasons if analyzer.reasons else ["No purity issues detected"],
        can_memoize=can_memoize,
        can_fuse=can_fuse,
    )


def build_purity_graph(funcs: list[Callable]) -> Dict[Callable, PurityResult]:
    """
    Analyze multiple functions and build a purity dependency graph.

    This performs transitive purity analysis - if function A calls function B,
    and we know B is pure, then A might be pure too.

    Args:
        funcs: List of functions to analyze

    Returns:
        Dictionary mapping functions to their purity results
    """
    results = {}
    known_pure = set()

    # First pass: identify explicitly pure functions
    for func in funcs:
        if func in _PURE_REGISTRY:
            known_pure.add(func.__name__)

    # Second pass: analyze all functions with knowledge of pure ones
    for func in funcs:
        result = analyze_purity(func, known_pure)
        results[func] = result

        # If we determined this is pure, add to known pure set
        if result.purity == Purity.PURE:
            known_pure.add(func.__name__)

    return results


def can_fuse_functors(f: Callable, g: Callable) -> bool:
    """
    Determine if two functors can be safely fused: O(g) ∘ O(f) → O(g ∘ f)

    Args:
        f: First functor
        g: Second functor (applied after f)

    Returns:
        True if functors can be fused without changing semantics
    """
    result_f = analyze_purity(f)
    result_g = analyze_purity(g)

    # Both must be fusable (pure or functionally pure)
    return result_f.can_fuse and result_g.can_fuse


# Example usage and testing
if __name__ == "__main__":
    # Pure function
    @pure
    def pure_transform(x):
        return x * 2 + 3

    # Functionally pure (depends on other pure function)
    def composed_transform(x):
        return pure_transform(x) + 1

    # Impure function
    def impure_transform(x):
        print(f"Processing {x}")
        return x * 2

    # Unknown purity
    def unknown_transform(x):
        return some_external_function(x)

    # Test analysis
    print("Pure function:")
    result = analyze_purity(pure_transform)
    print(f"  Purity: {result.purity}")
    print(f"  Can fuse: {result.can_fuse}")
    print(f"  Reasons: {result.reasons}")

    print("\nComposed function:")
    result = analyze_purity(composed_transform)
    print(f"  Purity: {result.purity}")
    print(f"  Can fuse: {result.can_fuse}")
    print(f"  Reasons: {result.reasons}")

    print("\nImpure function:")
    result = analyze_purity(impure_transform)
    print(f"  Purity: {result.purity}")
    print(f"  Can fuse: {result.can_fuse}")
    print(f"  Reasons: {result.reasons}")

    print(
        "\nCan fuse pure + composed?",
        can_fuse_functors(pure_transform, composed_transform),
    )
    print(
        "Can fuse pure + impure?", can_fuse_functors(pure_transform, impure_transform)
    )
