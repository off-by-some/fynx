"""Focused coverage for runtime edge paths that protect public guarantees."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from fynx import Observable
from fynx.observable.base import Observable as BaseObservable
from fynx.observable.base import TransformPurityError
from fynx.observable.computed import ComputedObservable
from fynx.observable.merged import MergedObservable


@pytest.fixture
def source() -> Observable[int]:
    """A simple integer source observable."""
    return Observable("source", 1)


@pytest.fixture
def received() -> list[int]:
    """Collect values delivered to subscribers."""
    return []


@pytest.mark.unit
@pytest.mark.observable
def test_set_rejects_mutation_while_computation_guard_is_active(
    source: Observable[int],
) -> None:
    """The circular guard rejects source mutation during computed evaluation."""
    BaseObservable._computation_dependency_stack.append({source})
    try:
        with pytest.raises(RuntimeError, match="Circular dependency detected"):
            source.set(2)
    finally:
        BaseObservable._computation_dependency_stack.pop()


@pytest.mark.unit
@pytest.mark.observable
def test_subscribe_replaces_existing_callback_without_duplicate_notifications(
    source: Observable[int],
    received: list[int],
) -> None:
    """Subscribing the same callback again replaces the old wrapper."""
    source.subscribe(received.append)
    source.subscribe(received.append)

    source.set(2)

    assert received == [2]


@pytest.mark.unit
@pytest.mark.observable
def test_empty_direct_callback_snapshot_is_stable(source: Observable[int]) -> None:
    """An observable with no direct subscribers returns an empty callback snapshot."""
    assert source._direct_callbacks_for_notification() == ()


@pytest.mark.unit
@pytest.mark.observable
def test_single_direct_callback_helper_dispatches_value(
    source: Observable[int],
    received: list[int],
) -> None:
    """The direct callback dispatcher delivers the changed value."""
    source._direct_callbacks.add(received.append)
    source._refresh_single_direct_callback()

    BaseObservable._notify_direct_callbacks_then_drain(source, 2)

    assert received == [2]


@pytest.mark.unit
@pytest.mark.observable
def test_direct_callback_helper_drains_pending_notifications(
    source: Observable[int],
    received: list[int],
) -> None:
    """Direct callback dispatch drains notifications queued by the callback."""
    dependent = Observable("dependent", 0)
    dependent.add_observer(lambda: received.append(dependent.value))
    source._direct_callbacks.add(
        lambda value: BaseObservable._schedule_notification(dependent)
    )
    source._refresh_single_direct_callback()

    BaseObservable._notify_direct_callbacks_then_drain(source, 2)

    assert received == [0]


@pytest.mark.unit
@pytest.mark.observable
def test_fast_observer_dispatches_source_value(
    source: Observable[int],
    received: list[int],
) -> None:
    """Fast observers receive the updated source value."""
    source.add_fast_observer(received.append)

    source.set(2)

    assert received == [2]


@pytest.mark.unit
@pytest.mark.observable
def test_fast_observer_exception_resets_notification_state(
    source: Observable[int],
) -> None:
    """Fast observer failures leave the notification scheduler reusable."""
    source.add_fast_observer(lambda value: (_ for _ in ()).throw(ValueError("boom")))

    with pytest.raises(ValueError, match="boom"):
        source.set(2)

    assert BaseObservable._notification_scheduled is False


@pytest.mark.unit
@pytest.mark.observable
def test_subscriber_exception_does_not_poison_next_update(
    source: Observable[int],
    received: list[int],
) -> None:
    """A failing subscriber leaves the scheduler usable for later updates."""
    other = Observable("other", 0)
    source.subscribe(lambda value: (_ for _ in ()).throw(ValueError("boom")))
    other.subscribe(received.append)

    with pytest.raises(ValueError, match="boom"):
        source.set(2)
    other.set(1)

    assert received == [1]


@pytest.mark.unit
@pytest.mark.observable
def test_remove_contexts_keeps_function_mapping_when_filtered_context_remains() -> None:
    """Filtered context removal disposes only matching reactive contexts."""
    first = Observable("first", 1)
    second = Observable("second", 2)
    calls: list[int] = []

    def record(value: int) -> None:
        calls.append(value)

    first.subscribe(record)
    second.subscribe(record)

    BaseObservable._dispose_subscription_contexts(
        record, lambda context: context.subscribed_observable is first
    )
    second.set(3)

    assert calls == [3]


@pytest.mark.unit
@pytest.mark.observable
def test_topological_sort_falls_back_when_dependency_view_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete dependency views keep every pending observable in the result."""
    first = Observable("first", 1)
    second = Observable("second", 2)
    monkeypatch.setattr(
        BaseObservable,
        "_observable_dependencies",
        classmethod(lambda cls, obs: {first, second}),
    )

    ordered = BaseObservable._topological_sort_notifications({first, second})

    assert set(ordered) == {first, second}


@pytest.mark.unit
@pytest.mark.observable
def test_computed_without_source_behaves_as_read_only_constant() -> None:
    """A source-less computed observable can be read but not assigned."""
    computed: ComputedObservable[int] = ComputedObservable("constant", 7, None)

    with pytest.raises(ValueError, match="read-only"):
        computed.set(8)

    assert computed.value == 7


@pytest.mark.unit
@pytest.mark.observable
def test_computed_without_source_reports_no_fusion_functions() -> None:
    """A source-less computed constant has no transform chain to fuse."""
    computed: ComputedObservable[int] = ComputedObservable("constant", 7, None)

    assert computed._get_fusion_funcs() == []


@pytest.mark.unit
@pytest.mark.observable
def test_computed_source_helpers_handle_absent_computation_or_source(
    source: Observable[int],
) -> None:
    """Source helper methods handle identity and source-less computed nodes."""
    identity = ComputedObservable("identity", 1, None, source)
    constant: ComputedObservable[int] = ComputedObservable("constant", 7, None)

    assert identity._apply_computation_to_source_value(3) == 3
    assert constant._source_current_value() is None
    assert constant._recompute_value() == 7


@pytest.mark.unit
@pytest.mark.observable
def test_computed_dynamic_signature_tracks_dynamic_dependency_versions(
    source: Observable[int],
) -> None:
    """Dynamic dependency signatures include captured dependency versions."""
    computed: ComputedObservable[int] = ComputedObservable(
        "dynamic", 1, lambda value: value, source
    )
    extra = Observable("extra", 2)
    computed._dynamic_dependencies = {extra}

    signature = computed._current_source_signature()

    assert signature is not None
    assert (id(extra), extra._version) in signature


@pytest.mark.unit
@pytest.mark.observable
def test_computed_fast_dependency_without_observers_marks_dirty(
    source: Observable[int],
) -> None:
    """Unobserved fast dependencies invalidate lazily instead of notifying."""
    computed: ComputedObservable[int] = ComputedObservable(
        "double", 2, lambda value: value * 2, source
    )

    computed._source_only_dependency_changed_fast(3)

    assert computed._is_dirty is True


@pytest.mark.unit
@pytest.mark.observable
def test_computed_fast_dependency_ignores_nodes_without_computation() -> None:
    """The fast source observer returns cleanly without a computation function."""
    computed: ComputedObservable[int] = ComputedObservable("constant", 7, None)

    computed._source_only_dependency_changed_fast(3)

    assert computed.value == 7


@pytest.mark.unit
@pytest.mark.observable
def test_computed_dependency_changed_schedules_when_forced_eager(
    source: Observable[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced eager computed nodes schedule when dependencies change."""
    computed = source >> (lambda value: value * 2)
    scheduled: list[Observable[int]] = []
    monkeypatch.setattr(BaseObservable, "_schedule_notification", scheduled.append)
    computed._force_eager = True

    computed._dependency_changed()

    assert scheduled == [computed]


@pytest.mark.unit
@pytest.mark.observable
def test_computed_sync_removes_stale_dynamic_dependency(
    source: Observable[int],
) -> None:
    """Dependency sync unsubscribes observers that are no longer runtime inputs."""
    computed = source >> (lambda value: value * 2)
    stale = Observable("stale", 9)
    computed._dependencies_active = True
    computed._dependency_observers = {stale}
    stale.add_observer(computed._dependency_callback)

    computed._sync_dependency_observers()

    assert computed._dependency_callback not in stale._observers


@pytest.mark.unit
@pytest.mark.observable
def test_computed_sync_noops_when_dependencies_are_inactive(
    source: Observable[int],
) -> None:
    """Dependency sync leaves inactive computed nodes alone."""
    computed = source >> (lambda value: value * 2)

    computed._sync_dependency_observers()

    assert computed._dependency_observers == set()


@pytest.mark.unit
@pytest.mark.observable
def test_computed_deactivate_dependencies_removes_non_fast_observers(
    source: Observable[int],
) -> None:
    """Deactivation removes observer callbacks from active dependencies."""
    computed = source >> (lambda value: value * 2)
    computed.add_observer(lambda: None)

    computed._deactivate_dependencies()

    assert computed._dependencies_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_computed_deactivate_noops_when_already_inactive(
    source: Observable[int],
) -> None:
    """Repeated deactivation is harmless."""
    computed = source >> (lambda value: value * 2)

    computed._deactivate_dependencies()

    assert computed._dependencies_active is False


@pytest.mark.unit
@pytest.mark.observable
def test_computed_refreshes_dependency_callback_when_subscriber_shape_changes(
    source: Observable[int],
) -> None:
    """Adding a second subscriber refreshes the active dependency callback."""
    doubled = source >> (lambda value: value * 2)
    first: list[int] = []
    second: list[int] = []

    doubled.subscribe(first.append)
    callback_before: Callable[..., object] = doubled._dependency_callback
    doubled.subscribe(second.append)

    assert doubled._dependencies_active is True
    assert doubled._dependency_callback is not callback_before


@pytest.mark.unit
@pytest.mark.observable
def test_computed_refresh_dependency_callback_noops_when_inactive(
    source: Observable[int],
) -> None:
    """Inactive computed nodes do not alter their dependency callback."""
    doubled = source >> (lambda value: value * 2)
    callback_before = doubled._dependency_callback

    doubled._refresh_dependency_callback()

    assert doubled._dependency_callback is callback_before


@pytest.mark.unit
@pytest.mark.observable
def test_observed_computed_with_unchanged_value_suppresses_notifications(
    source: Observable[int],
    received: list[int],
) -> None:
    """Observed computed nodes do not notify when recomputation is value-stable."""
    parity = source >> (lambda value: value % 2)
    parity.subscribe(received.append)

    source.set(3)

    assert received == []


@pytest.mark.unit
@pytest.mark.observable
def test_computed_source_only_notify_suppresses_stable_values(
    source: Observable[int],
) -> None:
    """Source-only recomputation returns without notifying when value is unchanged."""
    doubled = source >> (lambda value: value * 2)
    version_before = doubled._version

    doubled._notify_observers_source_only()

    assert doubled._version == version_before


@pytest.mark.unit
@pytest.mark.observable
def test_computed_source_only_notify_calls_plain_observers(
    source: Observable[int],
    received: list[int],
) -> None:
    """Plain observers are called after a source-only recomputation."""
    doubled = source >> (lambda value: value * 2)
    doubled.add_observer(lambda: received.append(doubled.value))
    source._value = 2
    source._version += 1

    doubled._notify_observers_source_only()

    assert received == [4]


@pytest.mark.unit
@pytest.mark.observable
def test_computed_value_read_inside_transform_is_rejected(
    source: Observable[int],
) -> None:
    """Computed observables enforce the same purity contract as sources."""
    doubled = source >> (lambda value: value * 2)
    outer = Observable("outer", 1)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        outer >> (lambda value: value + doubled.value)


@pytest.mark.unit
@pytest.mark.observable
def test_descriptor_access_without_owner_uses_instance_type() -> None:
    """Observable descriptors can resolve the owner from an instance."""
    descriptor = Observable("field", "value")

    class Owner:
        pass

    value = descriptor.__get__(Owner(), None)

    assert value.value == "value"


@pytest.mark.unit
@pytest.mark.observable
def test_descriptor_access_without_owner_or_instance_raises() -> None:
    """Descriptor access requires either an owner class or an instance."""
    descriptor = Observable("field", "value")

    with pytest.raises(AttributeError, match="requires an owner class"):
        descriptor.__get__(None, None)


@pytest.mark.unit
@pytest.mark.observable
def test_merged_observable_requires_at_least_one_source() -> None:
    """Products must contain at least one source observable."""
    with pytest.raises(ValueError, match="At least one observable"):
        MergedObservable.from_sources()


@pytest.mark.unit
@pytest.mark.observable
def test_merged_value_read_inside_transform_is_rejected(
    source: Observable[int],
) -> None:
    """Product reads obey the transform purity contract."""
    other = Observable("other", 2)
    product = source + other
    outer = Observable("outer", 1)

    with pytest.raises(TransformPurityError, match="inside a transform"):
        outer >> (lambda value: value + product.value[0])


@pytest.mark.unit
@pytest.mark.observable
def test_merged_context_manager_runs_block_immediately_and_on_source_change(
    source: Observable[int],
) -> None:
    """The merged context helper invokes a block with current product values."""
    other = Observable("other", 2)
    received: list[tuple[int, int]] = []

    with source + other as context:
        context(lambda left, right: received.append((left, right)))
    source.set(3)

    assert received == [(1, 2), (3, 2)]


@pytest.mark.unit
@pytest.mark.observable
def test_merged_subscribe_replaces_existing_callback(source: Observable[int]) -> None:
    """Subscribing the same product callback twice replaces its wrapper."""
    product = source + Observable("other", 2)
    calls: list[tuple[int, int]] = []

    def callback(left: int, right: int) -> None:
        calls.append((left, right))

    product.subscribe(callback)
    product.subscribe(callback)

    assert len(product._direct_observers) == 1


@pytest.mark.unit
@pytest.mark.observable
def test_merged_unsubscribe_removes_effect_delivery(source: Observable[int]) -> None:
    """Unsubscribed product effects do not fire on later source changes."""
    product = source + Observable("other", 2)
    received: list[tuple[int, int]] = []

    def callback(left: int, right: int) -> None:
        received.append((left, right))

    product.subscribe(callback)
    product.unsubscribe(callback)
    source.set(2)

    assert received == []


@pytest.mark.unit
@pytest.mark.observable
def test_conditional_unsubscribe_removes_effect_delivery(
    source: Observable[int],
    received: list[int],
) -> None:
    """Unsubscribed conditional effects do not fire on later source changes."""
    gate = source & (lambda value: value > 0)
    callback = received.append

    gate.subscribe(callback)
    source.set(2)
    gate.unsubscribe(callback)
    source.set(3)

    assert received == [2]
