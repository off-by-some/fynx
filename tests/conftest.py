"""
Shared pytest fixtures and configuration for FynX tests.
"""

import pytest

from fynx import Store
from fynx.observable import _reset_global_store


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset the global store before each test to prevent state leakage."""
    _reset_global_store()


@pytest.fixture
def store():
    """Provide a fresh Store instance for tests that need it."""
    return Store()
