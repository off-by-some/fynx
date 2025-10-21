"""
Morphism Classes for Reactive Graph Optimization
==============================================

This module contains the Morphism and MorphismParser classes used in
categorical optimization of FynX reactive observable networks.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Morphism:
    """
    Data structure representing a morphism in the reactive category.

    Morphisms can be:
    - Identity: represents no transformation
    - Single: represents a single computation step
    - Compose: represents composition of two morphisms
    """

    _type: str
    _name: Optional[str] = None
    _left: Optional["Morphism"] = None
    _right: Optional["Morphism"] = None

    @staticmethod
    def identity() -> "Morphism":
        """Create an identity morphism."""
        return Morphism(_type="identity")

    @staticmethod
    def single(name: str) -> "Morphism":
        """Create a single morphism with the given name."""
        return Morphism(_type="single", _name=name)

    @staticmethod
    def compose(left: "Morphism", right: "Morphism") -> "Morphism":
        """Create a composition of two morphisms."""
        return Morphism(_type="compose", _left=left, _right=right)

    def normalize(self) -> "Morphism":
        """
        Normalize this morphism using category theory identities.

        Identity laws: f ∘ id = f, id ∘ f = f
        Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
        """
        if self._type == "identity":
            return self
        elif self._type == "single":
            return self
        elif self._type == "compose":
            # Recursively normalize components
            assert self._left is not None and self._right is not None
            left_norm = self._left.normalize()
            right_norm = self._right.normalize()

            # Apply identity laws
            if left_norm._type == "identity":
                return right_norm
            if right_norm._type == "identity":
                return left_norm

            # Associativity: flatten nested compositions
            if left_norm._type == "compose":
                assert left_norm._left is not None and left_norm._right is not None
                return Morphism.compose(
                    left_norm._left, Morphism.compose(left_norm._right, right_norm)
                ).normalize()

            return Morphism.compose(left_norm, right_norm)
        else:
            # This should never happen with valid morphism types
            return self

    def canonical_form(self) -> Tuple[str, ...]:
        """
        Get a canonical tuple representation for equality comparison.
        """
        normalized = self.normalize()
        if normalized._type == "identity":
            return ("identity",)
        elif normalized._type == "single":
            return ("single", normalized._name or "")
        elif normalized._type == "compose":
            assert normalized._left is not None and normalized._right is not None
            left_form = normalized._left.canonical_form()
            right_form = normalized._right.canonical_form()
            return ("compose",) + left_form + right_form
        else:
            # This should never happen with valid morphism types
            return ("unknown",)

    def __eq__(self, other: object) -> bool:
        """Check structural equality after normalization."""
        if not isinstance(other, Morphism):
            return NotImplemented
        return self.canonical_form() == other.canonical_form()

    def __hash__(self) -> int:
        """Hash based on canonical form."""
        return hash(self.canonical_form())

    def __str__(self) -> str:
        """Convert back to string representation."""
        if self._type == "identity":
            return "id"
        elif self._type == "single":
            return self._name or "unknown"
        elif self._type == "compose":
            assert self._left is not None and self._right is not None
            return f"({self._left}) ∘ ({self._right})"
        else:
            return f"unknown({self._type})"

    def __repr__(self) -> str:
        return f"Morphism({self})"


class MorphismParser:
    """
    Parser for morphism signature strings into Morphism objects.
    """

    @staticmethod
    def parse(signature: str) -> Morphism:
        """Parse a morphism signature string into a Morphism object."""
        signature = signature.strip()

        # Strip outer parentheses
        while signature.startswith("(") and signature.endswith(")"):
            inner = signature[1:-1].strip()
            if MorphismParser._is_balanced(inner):
                signature = inner
            else:
                break

        # Handle identity
        if signature == "id" or signature == "":
            return Morphism.identity()

        # Handle single morphisms (no composition)
        if " ∘ " not in signature:
            return Morphism.single(signature)

        # Parse composition - split by top-level " ∘ " operators
        parts = MorphismParser._split_composition(signature)

        # Build composition tree from right to left (functional composition)
        result = MorphismParser.parse(parts[-1])
        for part in reversed(parts[:-1]):
            result = Morphism.compose(MorphismParser.parse(part), result)

        return result

    @staticmethod
    def _is_balanced(s: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        for char in s:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
                if count < 0:
                    return False
        return count == 0

    @staticmethod
    def _split_composition(sig: str) -> List[str]:
        """Split by ' ∘ ' at top level, respecting parentheses."""
        parts = []
        current = ""
        paren_depth = 0
        i = 0

        while i < len(sig):
            if sig[i : i + 3] == " ∘ " and paren_depth == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
                i += 3
                continue
            elif sig[i] == "(":
                paren_depth += 1
                current += sig[i]
            elif sig[i] == ")":
                paren_depth -= 1
                current += sig[i]
            else:
                current += sig[i]
            i += 1

        if current.strip():
            parts.append(current.strip())

        return parts
