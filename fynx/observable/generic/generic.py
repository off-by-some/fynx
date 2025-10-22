"""
FynX Generic Observable - Centralized Type Checking and Identification
======================================================================

This module provides a centralized place for all observable type checking and
identification logic using proper protocol-based isinstance checks.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types.protocols.computed_protocol import Computed
    from ..types.protocols.conditional_protocol import Conditional
    from ..types.protocols.merged_protocol import Mergeable
    from ..types.protocols.observable_protocol import Observable


class GenericObservable:
    """
    Centralized type identification for observables.

    This class provides static methods to check observable types using
    proper isinstance checks with protocols for type safety.
    """

    @staticmethod
    def is_observable(obj: Any) -> bool:
        """Check if an object is any type of observable using proper type checking."""
        from ..types.protocols.observable_protocol import Observable

        return isinstance(obj, Observable)

    @staticmethod
    def is_merged_observable(obj: Any) -> bool:
        """Check if an object is a MergedObservable."""
        from ..computed import MergedObservable

        return isinstance(obj, MergedObservable)

    @staticmethod
    def is_conditional_observable(obj: Any) -> bool:
        """Check if an object is a ConditionalObservable."""
        from ..computed import ConditionalObservable

        return isinstance(obj, ConditionalObservable)

    @staticmethod
    def is_computed_observable(obj: Any) -> bool:
        """Check if an object is a ComputedObservable."""
        from ..computed.computed import ComputedObservable

        return isinstance(obj, ComputedObservable)

    @staticmethod
    def get_source_observables(obj: Any) -> list:
        """
        Get source observables from a merged observable.

        Returns:
            List of source observables, or [obj] if not merged.
        """
        if GenericObservable.is_merged_observable(obj):
            return obj._source_observables
        return [obj]

    @staticmethod
    def get_ultimate_source(obj: Any) -> Any:
        """
        Find the ultimate source observable in a chain.

        Walks the _source_observable chain to find the root.
        """
        current = obj
        visited = set()

        while (
            hasattr(current, "_source_observable")
            and current._source_observable is not None
        ):
            if id(current) in visited:
                break  # Cycle detected
            visited.add(id(current))
            current = current._source_observable

        return current

    @staticmethod
    def create_merged_observable(*observables: "Observable") -> "Mergeable":
        """
        Create a MergedObservable from source observables.

        This is a factory method that avoids needing to import MergedObservable directly.
        """
        from ..computed import MergedObservable

        return MergedObservable(*observables)

    @staticmethod
    def create_conditional_observable(
        source: "Observable", *conditions
    ) -> "Conditional":
        """
        Create a ConditionalObservable.

        This is a factory method that avoids needing to import ConditionalObservable directly.
        """
        from ..computed import ConditionalObservable

        return ConditionalObservable(source, *conditions)

    @staticmethod
    def create_computed_observable(
        key: str, initial_value: Any, computation_func: Any, source_observable: Any
    ) -> "Computed":
        """
        Create a ComputedObservable.

        This is a factory method that avoids needing to import ComputedObservable directly.
        """
        from ..computed import ComputedObservable

        return ComputedObservable(
            key, initial_value, computation_func, source_observable
        )
