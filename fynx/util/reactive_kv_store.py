"""
Reactive Key-Value Store
=========================

Extension of KeyValueStore with push-based change notifications.
Provides efficient subscription system for tracking value changes.
"""

import fnmatch
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Import the base store
from fynx.util.kv_store import KeyValueStore


class ChangeType(Enum):
    """Types of changes that can occur in the store."""

    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"


@dataclass
class ChangeEvent:
    """
    Represents a change event in the store.

    Attributes:
        key: The key that changed
        change_type: Type of change (SET, DELETE, CLEAR)
        old_value: Previous value (None if new key or deleted)
        new_value: New value (None if deleted)
    """

    key: Any
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None

    def __repr__(self):
        if self.change_type == ChangeType.SET:
            return f"ChangeEvent(SET {self.key}: {self.old_value} -> {self.new_value})"
        elif self.change_type == ChangeType.DELETE:
            return f"ChangeEvent(DELETE {self.key}: {self.old_value})"
        else:
            return f"ChangeEvent(CLEAR)"


class Subscription:
    """
    Represents a subscription to store changes.

    Provides methods to pause, resume, and unsubscribe.
    """

    def __init__(
        self,
        subscriber_id: int,
        callback: Callable[[ChangeEvent], None],
        store: "ReactiveKeyValueStore",
        keys: Optional[Set[Any]] = None,
        pattern: Optional[str] = None,
    ):
        self.id = subscriber_id
        self.callback = callback
        self._store_ref = weakref.ref(store)
        self.keys = keys  # None means subscribe to all keys
        self.pattern = pattern  # Glob pattern for key matching
        self.active = True

    def pause(self):
        """Pause this subscription (stop receiving notifications)."""
        self.active = False

    def resume(self):
        """Resume this subscription (start receiving notifications again)."""
        self.active = True

    def unsubscribe(self):
        """Unsubscribe from all notifications."""
        store = self._store_ref()
        if store:
            store.unsubscribe(self.id)

    def matches_key(self, key: Any) -> bool:
        """Check if this subscription is interested in the given key."""
        if self.keys is None and self.pattern is None:
            return True  # Subscribe to all keys

        if self.keys is not None and key in self.keys:
            return True  # Explicit key match

        if self.pattern is not None:
            return fnmatch.fnmatch(str(key), self.pattern)

        return False

    def notify(self, event: ChangeEvent):
        """Notify this subscriber of an event."""
        if self.active and self.matches_key(event.key):
            try:
                self.callback(event)
            except Exception as e:
                # Log but don't crash on subscriber errors
                print(f"Error in subscription {self.id}: {e}")


class ReactiveKeyValueStore(KeyValueStore):
    """
    Key-value store with push-based change notifications.

    Features:
    - All features of KeyValueStore (deduplication, caching, thread-safety)
    - Subscribe to changes on specific keys or all keys
    - Efficient O(1) notification dispatch
    - Weak references prevent memory leaks
    - Pause/resume subscriptions
    - Batch updates to reduce notification overhead
    - Pattern-based subscriptions with glob patterns
    - Async notification dispatch

    Usage:
        store = ReactiveKeyValueStore()

        # Subscribe to all changes
        sub = store.subscribe(lambda event: print(f"Changed: {event}"))

        # Subscribe to specific keys
        sub = store.subscribe(
            lambda event: print(f"User changed: {event}"),
            keys=["user:1", "user:2"]
        )

        # Subscribe with patterns
        sub = store.subscribe(
            lambda event: print(f"Config changed: {event}"),
            pattern="config:*"
        )

        # Make changes (triggers notifications)
        store.set("user:1", {"name": "Alice"})

        # Batch updates
        with store.batch():
            store.set("key1", "value1")
            store.set("key2", "value2")
            # Notifications sent after batch completes

        # Unsubscribe
        sub.unsubscribe()
    """

    def __init__(
        self,
        enable_dedup: bool = True,
        cache_size: int = 10000,
        async_notifications: bool = False,
        max_workers: int = 1,
    ):
        """Initialize the reactive store."""
        super().__init__(
            enable_dedup=enable_dedup,
            cache_size=cache_size,
            async_operations=async_notifications,
            max_workers=max_workers,
        )

        # Subscription management
        self._subscriptions: Dict[int, Subscription] = {}
        self._next_sub_id = 0

        # Key-based index for O(1) notification dispatch
        # Maps key -> set of subscription IDs interested in that key
        self._key_subscribers: Dict[Any, Set[int]] = defaultdict(set)

        # Global subscribers (interested in all keys)
        self._global_subscribers: Set[int] = set()

        # Batch mode
        self._batch_mode = False
        self._batch_events: List[ChangeEvent] = []

        # Statistics
        self._notification_count = 0

    def subscribe(
        self,
        callback: Callable[[ChangeEvent], None],
        keys: Optional[List[Any]] = None,
        pattern: Optional[str] = None,
    ) -> Subscription:
        """
        Subscribe to store changes.

        Args:
            callback: Function to call on changes, receives ChangeEvent
            keys: Optional list of specific keys to watch. None = watch all keys.
            pattern: Optional glob pattern for key matching (e.g., "user:*", "data:*.json")

        Returns:
            Subscription object that can be used to unsubscribe
        """
        with self._lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1

            keys_set = set(keys) if keys else None
            subscription = Subscription(sub_id, callback, self, keys_set, pattern)

            self._subscriptions[sub_id] = subscription

            # Register in indexes
            if keys_set is None and pattern is None:
                self._global_subscribers.add(sub_id)
            elif pattern:
                # Pattern-based subscription - will be checked dynamically
                self._global_subscribers.add(
                    sub_id
                )  # Subscribe to all for pattern matching
            else:
                for key in keys_set:
                    self._key_subscribers[key].add(sub_id)

            return subscription

    def unsubscribe(self, subscription_id: int) -> bool:
        """
        Remove a subscription.

        Args:
            subscription_id: ID of subscription to remove

        Returns:
            True if subscription was removed, False if not found
        """
        with self._lock:
            if subscription_id not in self._subscriptions:
                return False

            subscription = self._subscriptions[subscription_id]

            # Remove from indexes
            if subscription.keys is None:
                self._global_subscribers.discard(subscription_id)
            else:
                for key in subscription.keys:
                    self._key_subscribers[key].discard(subscription_id)
                    # Clean up empty sets
                    if not self._key_subscribers[key]:
                        del self._key_subscribers[key]

            del self._subscriptions[subscription_id]
            return True

    def _notify(self, event: ChangeEvent):
        """
        Notify all interested subscribers of an event.
        O(1) average case due to key-based indexing.
        Supports both sync and async notification dispatch.
        """
        if self._batch_mode:
            self._batch_events.append(event)
            return

        self._notification_count += 1

        # Find all interested subscribers
        interested_ids = set()

        # Add global subscribers
        interested_ids.update(self._global_subscribers)

        # Add key-specific subscribers
        if event.key in self._key_subscribers:
            interested_ids.update(self._key_subscribers[event.key])

        # Notify all interested subscribers
        for sub_id in interested_ids:
            if sub_id in self._subscriptions:
                subscription = self._subscriptions[sub_id]

                if self.async_enabled:
                    # Async dispatch - non-blocking
                    self.execute_async(subscription.notify, event)
                else:
                    # Sync dispatch - blocking but immediate
                    subscription.notify(event)

    def set(self, key: Any, value: Any) -> None:
        """Set key to value and notify subscribers."""
        with self._lock:
            # Get old value if exists
            old_value = self.get(key, self._MISSING) if self.has(key) else None

            # Call parent set
            super().set(key, value)

            # Notify subscribers
            event = ChangeEvent(
                key=key,
                change_type=ChangeType.SET,
                old_value=old_value if old_value is not self._MISSING else None,
                new_value=value,
            )
            self._notify(event)

    def delete(self, key: Any) -> bool:
        """Delete key and notify subscribers."""
        with self._lock:
            # Get old value before deletion
            old_value = self.get(key, self._MISSING) if self.has(key) else self._MISSING

            # Call parent delete
            result = super().delete(key)

            # Notify if deletion was successful
            if result:
                event = ChangeEvent(
                    key=key,
                    change_type=ChangeType.DELETE,
                    old_value=old_value if old_value is not self._MISSING else None,
                )
                self._notify(event)

            return result

    def clear(self) -> None:
        """Clear all data and notify subscribers."""
        with self._lock:
            # Get all keys before clearing
            keys_to_clear = list(self.keys())

            # Call parent clear
            super().clear()

            # Notify about each key deletion
            for key in keys_to_clear:
                event = ChangeEvent(key=key, change_type=ChangeType.CLEAR)
                self._notify(event)

    def batch(self):
        """
        Context manager for batching multiple updates.
        Notifications are sent only after the batch completes.

        Usage:
            with store.batch():
                store.set("key1", "value1")
                store.set("key2", "value2")
                # Notifications sent here
        """
        return _BatchContext(self)

    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get statistics about subscriptions."""
        with self._lock:
            return {
                "total_subscriptions": len(self._subscriptions),
                "global_subscriptions": len(self._global_subscribers),
                "key_specific_subscriptions": sum(
                    len(subs) for subs in self._key_subscribers.values()
                ),
                "watched_keys": len(self._key_subscribers),
                "notifications_sent": self._notification_count,
            }


class _BatchContext:
    """Context manager for batch updates."""

    def __init__(self, store: ReactiveKeyValueStore):
        self.store = store

    def __enter__(self):
        self.store._batch_mode = True
        self.store._batch_events = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore normal mode
        self.store._batch_mode = False

        # Send all batched notifications with deduplication
        events = self.store._batch_events
        self.store._batch_events = []

        # Deduplicate events by key - keep only the latest event per key
        deduplicated_events = {}
        for event in events:
            deduplicated_events[event.key] = event

        # Send deduplicated notifications
        for event in deduplicated_events.values():
            self.store._notify(event)

        return False


# Convenience function for creating reactive stores
def create_reactive_store(
    cache_size: int = 10000,
    enable_dedup: bool = True,
    async_notifications: bool = False,
    max_workers: int = 1,
) -> ReactiveKeyValueStore:
    """
    Create a reactive key-value store with specified settings.

    Args:
        cache_size: Size of the LRU cache
        enable_dedup: Whether to enable value deduplication
        async_notifications: Whether to enable async notification dispatch
        max_workers: Maximum number of worker threads for async operations (default: 1)

    Returns:
        Configured ReactiveKeyValueStore instance
    """
    return ReactiveKeyValueStore(
        enable_dedup=enable_dedup,
        cache_size=cache_size,
        async_notifications=async_notifications,
        max_workers=max_workers,
    )
