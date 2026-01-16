"""
Morphism Classes for Reactive Graph Optimization
==============================================

This module contains the Morphism and MorphismParser classes used in
categorical optimization of FynX reactive observable networks.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union


class MorphismType(Enum):
    """Enumeration of valid morphism types."""

    IDENTITY = "identity"
    SINGLE = "single"
    COMPOSE = "compose"


@dataclass(frozen=True)
class Morphism:
    """
    Data structure representing a morphism in the reactive category.

    Morphisms can be:
    - Identity: represents no transformation
    - Single: represents a single computation step
    - Compose: represents composition of two morphisms
    """

    morphism_type: MorphismType
    name: Optional[str] = None
    left: Optional["Morphism"] = None
    right: Optional["Morphism"] = None

    @staticmethod
    def identity() -> "Morphism":
        """Create an identity morphism."""
        return Morphism(morphism_type=MorphismType.IDENTITY)

    @staticmethod
    def single(name: str) -> "Morphism":
        """Create a single morphism with the given name."""
        return Morphism(morphism_type=MorphismType.SINGLE, name=name)

    @staticmethod
    def compose(left: "Morphism", right: "Morphism") -> "Morphism":
        """Create a composition of two morphisms."""
        return Morphism(morphism_type=MorphismType.COMPOSE, left=left, right=right)

    def _validate_compose_components(self) -> None:
        """Validate that a compose morphism has both left and right components."""
        if self.left is None or self.right is None:
            raise ValueError(
                "Compose morphism must have both left and right components"
            )

    def normalize(self) -> "Morphism":
        """
        Normalize this morphism using category theory identities.

        Identity laws: f ∘ id = f, id ∘ f = f
        Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
        """
        if self.morphism_type == MorphismType.IDENTITY:
            return self
        elif self.morphism_type == MorphismType.SINGLE:
            return self
        elif self.morphism_type == MorphismType.COMPOSE:
            # Recursively normalize components
            self._validate_compose_components()
            left_norm = self.left.normalize()
            right_norm = self.right.normalize()

            # Apply identity laws: f ∘ id = f, id ∘ f = f
            if left_norm.morphism_type == MorphismType.IDENTITY:
                return right_norm
            if right_norm.morphism_type == MorphismType.IDENTITY:
                return left_norm

            # Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
            if left_norm.morphism_type == MorphismType.COMPOSE:
                left_norm._validate_compose_components()
                return Morphism.compose(
                    left_norm.left, Morphism.compose(left_norm.right, right_norm)
                ).normalize()

            return Morphism.compose(left_norm, right_norm)
        else:
            # This should never happen with valid morphism types
            raise ValueError(f"Unknown morphism type: {self.morphism_type}")

    def canonical_form(self) -> Tuple[str, ...]:
        """
        Get a canonical tuple representation for equality comparison.
        """
        normalized = self.normalize()
        if normalized.morphism_type == MorphismType.IDENTITY:
            return ("identity",)
        elif normalized.morphism_type == MorphismType.SINGLE:
            return ("single", normalized.name or "")
        elif normalized.morphism_type == MorphismType.COMPOSE:
            normalized._validate_compose_components()
            left_form = normalized.left.canonical_form()
            right_form = normalized.right.canonical_form()
            return ("compose",) + left_form + right_form
        else:
            # This should never happen with valid morphism types
            raise ValueError(f"Unknown morphism type: {normalized.morphism_type}")

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
        if self.morphism_type == MorphismType.IDENTITY:
            return "id"
        elif self.morphism_type == MorphismType.SINGLE:
            return self.name or "unknown"
        elif self.morphism_type == MorphismType.COMPOSE:
            self._validate_compose_components()
            return f"({self.left}) ∘ ({self.right})"
        else:
            raise ValueError(f"Unknown morphism type: {self.morphism_type}")

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
