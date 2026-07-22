"""
FynX Registry - global reactive context management
=====================================================

Two module-level registries track reactive contexts across the whole
system, used internally by Observable and Store to manage subscriptions and
clean them up when no longer needed:

- `_all_reactive_contexts`: every active ReactiveContext instance
- `_func_to_contexts`: maps a user function to its reactive contexts, for O(1) unsubscribe
"""

from typing import Callable, Dict, Set

# Global registry of all active reactive contexts for unsubscribe functionality
_all_reactive_contexts: Set = set()

# Mapping from functions to their reactive contexts for O(1) unsubscribe
_func_to_contexts: Dict[Callable, list] = {}
