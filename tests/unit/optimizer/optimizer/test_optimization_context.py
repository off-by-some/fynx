"""Unit tests for OptimizationContext functionality."""

import pytest

from fynx import observable
from fynx.optimizer import OptimizationContext, ReactiveGraph


@pytest.mark.unit
@pytest.mark.optimizer
def test_optimization_context_creation():
    """OptimizationContext creates with proper initial state"""
    context = OptimizationContext()

    assert context.optimizer is not None
    assert isinstance(context.optimizer, ReactiveGraph)
    assert context._thread_local is not None


@pytest.mark.unit
@pytest.mark.optimizer
def test_optimization_context_as_context_manager():
    """OptimizationContext works as context manager"""
    with OptimizationContext() as context:
        assert context is not None
        assert isinstance(context, OptimizationContext)


@pytest.mark.unit
@pytest.mark.optimizer
def test_current_context_retrieval():
    """OptimizationContext.current() retrieves active context"""
    # Initially no context
    assert OptimizationContext.current() is None

    # Create context
    with OptimizationContext() as context:
        assert OptimizationContext.current() is context


@pytest.mark.unit
@pytest.mark.optimizer
def test_get_optimizer_creates_if_needed():
    """OptimizationContext.get_optimizer() creates optimizer when no context exists"""
    # Without context, should create a new optimizer
    optimizer = OptimizationContext.get_optimizer()
    assert isinstance(optimizer, ReactiveGraph)


@pytest.mark.unit
@pytest.mark.optimizer
def test_register_observable():
    """OptimizationContext.register_observable() registers observables in optimizer"""
    context = OptimizationContext()
    obs = observable(1)

    context.register_observable(obs)

    # Should be registered in the optimizer
    assert obs in context.optimizer
