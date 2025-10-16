"""
Fynx Observable Module - Core Reactive Classes
==============================================

This module contains the fundamental classes that implement Fynx's reactive
programming system. These classes provide the building blocks for creating
observable values and managing their dependencies.

Core Classes:
- **Observable**: The basic reactive value class that notifies subscribers of changes
- **ReactiveContext**: Manages execution context for reactive functions and dependency tracking
- **MergedObservable**: Combines multiple observables into a single reactive unit
- **ConditionalObservable**: Creates observables that react based on conditions
- **SubscriptableDescriptor**: Descriptor for creating observable attributes in classes

Operators:
- **rshift_operator**: Implements the `>>` operator for computed observables
- **and_operator**: Implements the `&` operator for conditional observables

This module forms the foundation of Fynx's reactivity system, providing transparent
dependency tracking and automatic change propagation.
"""

from .base import Observable, ReactiveContext
from .merged import MergedObservable
from .conditional import ConditionalObservable
from .descriptors import SubscriptableDescriptor
from .operators import rshift_operator, and_operator

__all__ = [
    "Observable",
    "MergedObservable",
    "ConditionalObservable",
    "ReactiveContext",
    "SubscriptableDescriptor",
    "rshift_operator",
    "and_operator",
]
