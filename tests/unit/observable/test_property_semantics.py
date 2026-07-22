"""Property-style tests for core observable algebra laws."""

from __future__ import annotations

from functools import reduce
from operator import add

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fynx import Observable


@pytest.mark.unit
@pytest.mark.observable
@given(st.integers(min_value=-1_000_000, max_value=1_000_000))
@settings(max_examples=30)
def test_transform_identity_preserves_value_semantics(value: int) -> None:
    """Mapping identity preserves the current observable value."""
    source = Observable("source", value)

    transformed = source >> (lambda current: current)

    assert transformed.value == source.value


@pytest.mark.unit
@pytest.mark.observable
@given(st.integers(min_value=-10_000, max_value=10_000))
@settings(max_examples=30)
def test_transform_composition_matches_single_composed_transform(value: int) -> None:
    """Sequential pure transforms match one explicitly composed transform."""
    source = Observable("source", value)

    chained = source >> (lambda current: current + 3) >> (lambda current: current * 5)
    composed = source >> (lambda current: (current + 3) * 5)

    assert chained.value == composed.value


@pytest.mark.unit
@pytest.mark.observable
@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=3, max_size=8))
@settings(max_examples=30)
def test_product_chains_flatten_into_ordered_tuples(values: list[int]) -> None:
    """Chained products are flat ordered tuples, not nested pairs."""
    sources = [
        Observable(f"value_{index}", value) for index, value in enumerate(values)
    ]

    product = reduce(add, sources)

    assert product.value == tuple(values)
