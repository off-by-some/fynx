"""
FynX ObservableValue Module - Auto-Lifting Value Container
=========================================================

This module provides ObservableValue, a transparent wrapper that automatically
unwraps nested ObservableValue instances and provides mutation-based magic methods
that trigger change notifications in the underlying Observable.

It also provides SubscriptableDescriptor for creating reactive class attributes.
"""

from .descriptors import SubscriptableDescriptor
from .value import ObservableValue

__all__ = ["ObservableValue", "SubscriptableDescriptor"]
