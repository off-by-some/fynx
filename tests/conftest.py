"""
Shared pytest fixtures and configuration for FynX tests.
"""

import pytest

from fynx import Store


@pytest.fixture
def store():
    """Provide a fresh Store instance for tests that need it."""
    return Store()
