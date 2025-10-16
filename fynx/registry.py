"""
Fynx Registry - Global Reactive Context Management
==================================================

This module provides global registries that track reactive contexts and their
relationships. These registries enable efficient subscription management and
cleanup across the entire Fynx reactive system.

Global Registries:
- **_all_reactive_contexts**: Set of all active ReactiveContext instances
- **_func_to_contexts**: Mapping from user functions to their reactive contexts

These registries support:
- **Efficient Unsubscription**: O(1) lookup of contexts by function
- **Memory Management**: Tracking all active contexts for cleanup
- **Cross-Component Coordination**: Managing reactive relationships globally
- **Debugging Support**: Inspecting active reactive contexts

The registries are primarily used internally by Observable and Store classes
for managing subscriptions and ensuring proper cleanup when reactive contexts
are no longer needed.
"""

from typing import Callable, Dict, Set

# Global registry of all active reactive contexts for unsubscribe functionality
_all_reactive_contexts: Set = set()

# Mapping from functions to their reactive contexts for O(1) unsubscribe
_func_to_contexts: Dict[Callable, list] = {}
