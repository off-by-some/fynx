#!/usr/bin/env python3
"""
FynX Streamlit TODO Application Store
=====================================

This module contains the core data models and reactive store for the FynX TODO application.
It defines the TodoItem data structure and TodoStore class that manage application state
with automatic Streamlit session state synchronization.

The module provides:
- Immutable TodoItem dataclass with unique ID generation
- Reactive TodoStore with computed properties for filtering and statistics
- Comprehensive test suite for all functionality

To run the tests:
```bash
$ pip install fynx && python examples/streamlit/todo_store.py
```

"""

import logging
from typing import List

from store import StreamlitStore
from todo_item_model import UUID_STRING_LENGTH, TodoItem

from fynx import observable, reactive

# ==============================================================================================
# Configuration and Constants
# ==============================================================================================

# Logging configuration
LOG_LEVEL = logging.INFO


# Filter mode constants
FILTER_MODE_ALL = "all"
FILTER_MODE_ACTIVE = "active"
FILTER_MODE_COMPLETED = "completed"

# ==============================================================================================
# Logging Setup
# ==============================================================================================

logging.basicConfig(level=LOG_LEVEL)


# ==============================================================================================
# TodoStore - Reactive State Management for TODO Application
# ==============================================================================================


class TodoStore(StreamlitStore):
    """
    A reactive store managing TODO application state with Streamlit session state integration.

    This store extends StreamlitStore to provide comprehensive state management for a TODO
    application. It automatically synchronizes with Streamlit's session state while providing
    reactive computed properties for filtering, statistics, and UI state management.

    The store maintains immutability by working with TodoItem instances and provides
    methods for all common todo operations like adding, toggling, deleting, and bulk operations.

    Attributes:
        todos: Observable list of TodoItem objects representing all todos.
        filter_mode: Observable string controlling which todos are visible
                    ("all", "active", or "completed").

    Computed Properties:
        active_todos: List of incomplete todos (computed from todos).
        completed_todos: List of completed todos (computed from todos).
        filtered_todos: Todos matching the current filter_mode.
        total_count: Total number of todos.
        active_count: Number of active (incomplete) todos.
        completed_count: Number of completed todos.
        has_no_todos: Boolean indicating if there are no todos.
        all_todos_completed: Boolean indicating if all todos are completed.
        has_any_completed: Boolean indicating if any todos are completed.
        stats_text: Human-readable summary of current todo state.

    Reactive Methods:
        _update_stats_text: Automatically updates stats_text when dependencies change.

    Example:
        ```python
        # Add a new todo
        TodoStore.add_todo("Buy groceries")

        # Toggle completion
        TodoStore.toggle_todo(todo_id)

        # Change filter
        TodoStore.filter_mode = "completed"
        ```
    """

    # ==============================================================================================
    # Core Observable State
    # ==============================================================================================

    todos: List[TodoItem] = observable([])
    filter_mode: str = observable(
        FILTER_MODE_ALL
    )  # Filter mode: "all", "active", "completed"

    # ==============================================================================================
    # Computed Properties - Derived State
    # ==============================================================================================

    # Basic filtering computed properties
    active_todos = todos >> (
        lambda todos_list: [todo for todo in todos_list if not todo.completed]
    )

    completed_todos = todos >> (
        lambda todos_list: [todo for todo in todos_list if todo.completed]
    )

    # Count-based computed properties
    total_count = todos >> (lambda todos_list: len(todos_list))
    active_count = active_todos >> (lambda active_list: len(active_list))
    completed_count = completed_todos >> (lambda completed_list: len(completed_list))

    # Dynamic filtering based on current filter mode
    filtered_todos = (todos | filter_mode) >> (
        lambda todos_list, current_filter: {
            FILTER_MODE_ACTIVE: [todo for todo in todos_list if not todo.completed],
            FILTER_MODE_COMPLETED: [todo for todo in todos_list if todo.completed],
            FILTER_MODE_ALL: todos_list,
        }.get(current_filter, todos_list)
    )

    # Boolean state indicators for conditional logic
    has_no_todos = total_count >> (lambda total: total == 0)
    all_todos_completed = active_count >> (lambda active: active == 0)
    has_any_completed = completed_count >> (lambda completed: completed > 0)

    # Human-readable statistics text (updated reactively)
    stats_text = observable("")

    # ==============================================================================================
    # Reactive Methods - Automatic State Updates
    # ==============================================================================================

    @staticmethod
    @reactive(
        has_no_todos,
        all_todos_completed,
        has_any_completed,
        active_count,
        completed_count,
        total_count,
    )
    def _update_stats_text(
        is_empty: bool,
        all_completed: bool,
        has_any_completed_items: bool,
        active_items_count: int,
        completed_items_count: int,
        total_items_count: int,
    ) -> None:
        """
        Automatically update the statistics text when todo state changes.

        This reactive method is triggered whenever any of the dependency observables
        change. It computes and updates the human-readable statistics text that
        summarizes the current state of the todo list.

        The method handles different scenarios:
        - Empty todo list
        - All todos completed
        - Only active todos
        - Mixed active and completed todos

        Args:
            is_empty: True if there are no todos in the list.
            all_completed: True if all todos are completed (no active todos).
            has_any_completed_items: True if at least one todo is completed.
            active_items_count: Number of active (incomplete) todos.
            completed_items_count: Number of completed todos.
            total_items_count: Total number of todos.

        Note:
            This method uses proper singular/plural forms and celebratory messaging
            to provide a friendly user experience.
        """
        # Handle empty todo list
        if is_empty:
            TodoStore.stats_text.set("No todos yet. Add one above!")
            return

        # Handle all todos completed
        if all_completed:
            TodoStore.stats_text.set(f"🎉 All {total_items_count} todos completed!")
            return

        # Handle only active todos (no completed ones)
        if not has_any_completed_items:
            todo_word = "todo" if active_items_count == 1 else "todos"
            TodoStore.stats_text.set(
                f"{active_items_count} active {todo_word} remaining"
            )
            return

        # Handle mixed state (both active and completed todos)
        TodoStore.stats_text.set(
            f"{active_items_count} active, {completed_items_count} completed"
        )

    # ==============================================================================================
    # Public API Methods - Todo Management Operations
    # ==============================================================================================

    @classmethod
    def add_todo(cls, text: str) -> None:
        """
        Add a new todo item to the store with the specified text.

        Creates a new TodoItem with a unique ID and adds it to the todos list.
        The text is trimmed of leading/trailing whitespace. Empty strings after
        trimming are ignored (no todo is created).

        Args:
            text: The text content for the new todo item. Leading and trailing
                 whitespace will be stripped.

        Note:
            This method triggers reactive updates to all computed properties
            and synchronizes the change with Streamlit session state.
        """
        sanitized_text = text.strip()
        if sanitized_text:
            new_todo_item = TodoItem.create(sanitized_text)
            cls.todos = cls.todos.value + [new_todo_item]

    @classmethod
    def toggle_todo(cls, todo_id: str) -> None:
        """
        Toggle the completion status of the todo item with the specified ID.

        Finds the todo item by ID and creates a new version with the opposite
        completion status. All other todos remain unchanged.

        Args:
            todo_id: The unique identifier of the todo item to toggle.

        Note:
            Due to TodoItem immutability, this creates new TodoItem instances
            for any todos that need to be updated.
        """
        cls.todos = [
            todo.toggle_completion() if todo.id == todo_id else todo
            for todo in cls.todos.value
        ]

    @classmethod
    def delete_todo(cls, todo_id: str) -> None:
        """
        Remove the todo item with the specified ID from the store.

        Args:
            todo_id: The unique identifier of the todo item to remove.

        Note:
            If the specified ID doesn't exist, this method does nothing.
            No error is raised for non-existent IDs.
        """
        cls.todos = [todo for todo in cls.todos.value if todo.id != todo_id]

    @classmethod
    def clear_completed(cls) -> None:
        """
        Remove all completed todo items from the store.

        This method filters out all todos where completed=True, keeping only
        the active (incomplete) todos. If there are no completed todos, the
        store remains unchanged.

        Note:
            This is a bulk operation that affects multiple todos at once.
        """
        cls.todos = [todo for todo in cls.todos.value if not todo.completed]

    @classmethod
    def toggle_all_active(cls) -> None:
        """
        Mark all currently active (incomplete) todos as completed.

        Finds all todos that are not completed and toggles them to completed.
        Todos that are already completed remain unchanged.

        Note:
            If there are no active todos, this method has no effect.
            This is useful for bulk completion operations.
        """
        active_todos_list = [todo for todo in cls.todos.value if not todo.completed]
        if active_todos_list:
            cls.todos = [
                todo.toggle_completion() if not todo.completed else todo
                for todo in cls.todos.value
            ]

    @classmethod
    def toggle_all_completed(cls) -> None:
        """
        Mark all currently completed todos as active (incomplete).

        Finds all todos that are completed and toggles them to active.
        Todos that are already active remain unchanged.

        Note:
            If there are no completed todos, this method has no effect.
            This is useful for bulk reactivation operations.
        """
        completed_todos_list = [todo for todo in cls.todos.value if todo.completed]
        if completed_todos_list:
            cls.todos = [
                todo.toggle_completion() if todo.completed else todo
                for todo in cls.todos.value
            ]


# ==============================================================================================
# Test Suite - Comprehensive Validation of TodoStore Functionality
# ==============================================================================================


def reset_store_to_clean_state():
    """
    Reset the TodoStore to a clean state for testing.

    This helper function ensures tests start with a consistent, empty state
    and prevents test interference.
    """
    TodoStore.todos = []
    TodoStore.filter_mode = FILTER_MODE_ALL


def verify_store_is_in_initial_empty_state():
    """
    Verify that the TodoStore is in its initial empty state.

    Checks that all counters are zero and the todo list is empty,
    ensuring tests start from a known baseline.
    """
    assert len(TodoStore.todos.value) == 0
    assert TodoStore.total_count.value == 0
    assert TodoStore.active_count.value == 0
    assert TodoStore.completed_count.value == 0


def test_empty_input_validation():
    """
    Test that empty or whitespace-only inputs are properly rejected.

    Ensures that the add_todo method ignores inputs that are empty or
    contain only whitespace characters, preventing creation of meaningless
    todo items.
    """
    reset_store_to_clean_state()

    # Test various forms of empty/whitespace input
    empty_inputs = ["", "   ", "\t\n", "  \t  "]

    for empty_input in empty_inputs:
        TodoStore.add_todo(empty_input)

    # Verify no todos were created from empty inputs
    assert len(TodoStore.todos.value) == 0, "Empty inputs should not create todos"
    print("✅ Empty input validation works correctly")


def test_successful_todo_creation():
    """
    Test successful creation and addition of valid todo items.

    Verifies that valid text inputs result in properly created TodoItem
    instances with correct properties and that the store state is updated
    appropriately.
    """
    reset_store_to_clean_state()

    # Add valid todo items
    first_task_text = "First Task"
    second_task_text = "Second Task"

    TodoStore.add_todo(first_task_text)
    TodoStore.add_todo(second_task_text)

    # Verify todos were created and added correctly
    assert len(TodoStore.todos.value) == 2
    assert TodoStore.total_count.value == 2
    assert TodoStore.active_count.value == 2  # All should be active initially
    assert TodoStore.completed_count.value == 0  # None should be completed

    current_todos = TodoStore.todos.value
    todo_texts = [todo.text for todo in current_todos]
    assert todo_texts == [first_task_text, second_task_text]

    # Verify all todos are initially active and have valid IDs
    for todo in current_todos:
        assert not todo.completed
        assert isinstance(todo.id, str)
        assert len(todo.id) == UUID_STRING_LENGTH

    print("✅ Todo creation and addition works correctly")


def test_todo_completion_toggling():
    """
    Test toggling the completion status of todo items.

    Verifies that the toggle_todo method correctly changes a todo's completion
    status and that the computed counters update appropriately.
    """
    reset_store_to_clean_state()

    # Create a test todo and capture its ID
    test_task_text = "Test Task"
    TodoStore.add_todo(test_task_text)
    target_todo_id = TodoStore.todos.value[0].id

    # Initially should be active
    assert not TodoStore.todos.value[0].completed
    assert TodoStore.active_count.value == 1
    assert TodoStore.completed_count.value == 0

    # Toggle to completed state
    TodoStore.toggle_todo(target_todo_id)
    assert TodoStore.todos.value[0].completed
    assert TodoStore.active_count.value == 0
    assert TodoStore.completed_count.value == 1

    # Toggle back to active state
    TodoStore.toggle_todo(target_todo_id)
    assert not TodoStore.todos.value[0].completed
    assert TodoStore.active_count.value == 1
    assert TodoStore.completed_count.value == 0

    print("✅ Todo completion toggling works correctly")


def test_todo_deletion():
    """
    Test deletion of todo items from the store.

    Verifies that delete_todo correctly removes items by ID and that
    the store state updates appropriately after deletion.
    """
    reset_store_to_clean_state()

    # Create two test todos
    first_todo_text = "First"
    second_todo_text = "Second"

    TodoStore.add_todo(first_todo_text)
    TodoStore.add_todo(second_todo_text)

    first_todo_id = TodoStore.todos.value[0].id
    assert len(TodoStore.todos.value) == 2

    # Delete the first todo
    TodoStore.delete_todo(first_todo_id)

    # Verify deletion was successful
    remaining_todos = TodoStore.todos.value
    assert len(remaining_todos) == 1
    assert remaining_todos[0].text == second_todo_text
    assert TodoStore.total_count.value == 1

    print("✅ Todo deletion works correctly")


def test_basic_crud_operations():
    """
    Run comprehensive tests for basic Create, Read, Update, Delete operations.

    This test suite covers the fundamental todo management operations:
    - Creating todos (with input validation)
    - Reading/verifying todo state
    - Updating todo completion status
    - Deleting todos
    """
    test_empty_input_validation()
    test_successful_todo_creation()
    test_todo_completion_toggling()
    test_todo_deletion()

    print("✅ Basic CRUD operations work correctly")


def test_filtering():
    """Test todo filtering by completion status."""
    # Setup: one active, one completed
    TodoStore.todos = [
        TodoItem.create("Active"),
        TodoItem("completed-id", "Completed", True),
    ]

    # Verify initial state
    assert len(TodoStore.todos.value) == 2
    assert TodoStore.total_count.value == 2
    assert TodoStore.active_count.value == 1
    assert TodoStore.completed_count.value == 1

    # Test active filter
    TodoStore.filter_mode = "active"
    assert len(TodoStore.filtered_todos.value) == 1
    assert TodoStore.filtered_todos.value[0].text == "Active"
    assert not TodoStore.filtered_todos.value[0].completed
    assert isinstance(TodoStore.filtered_todos.value[0].id, str)
    assert len(TodoStore.filtered_todos.value[0].id) == 36

    # Test completed filter
    TodoStore.filter_mode = "completed"
    assert len(TodoStore.filtered_todos.value) == 1
    assert TodoStore.filtered_todos.value[0].text == "Completed"
    assert TodoStore.filtered_todos.value[0].completed
    assert TodoStore.filtered_todos.value[0].id == "completed-id"

    # Test all filter
    TodoStore.filter_mode = "all"
    assert len(TodoStore.filtered_todos.value) == 2
    texts = [t.text for t in TodoStore.filtered_todos.value]
    assert "Active" in texts and "Completed" in texts

    # Test filter mode validation (default behavior)
    TodoStore.filter_mode = "invalid"
    assert len(TodoStore.filtered_todos.value) == 2  # Should fall back to "all"

    print("✅ Filtering works")


def test_bulk_operations():
    """Test clear completed and toggle all operations."""
    # Setup: mixed completed and active todos
    TodoStore.todos = [
        TodoItem.create("Active 1"),
        TodoItem.create("Active 2"),
        TodoItem("comp1", "Completed 1", True),
        TodoItem("comp2", "Completed 2", True),
    ]

    # Verify initial mixed state
    assert len(TodoStore.todos.value) == 4
    assert TodoStore.total_count.value == 4
    assert TodoStore.active_count.value == 2
    assert TodoStore.completed_count.value == 2

    active_items = [t for t in TodoStore.todos.value if not t.completed]
    completed_items = [t for t in TodoStore.todos.value if t.completed]
    assert len(active_items) == 2
    assert len(completed_items) == 2
    assert all(isinstance(t.id, str) for t in TodoStore.todos.value)

    # Clear completed
    TodoStore.clear_completed()
    assert len(TodoStore.todos.value) == 2
    assert TodoStore.total_count.value == 2
    assert TodoStore.active_count.value == 2
    assert TodoStore.completed_count.value == 0
    assert all(not t.completed for t in TodoStore.todos.value)
    assert all(t.id != "comp1" and t.id != "comp2" for t in TodoStore.todos.value)

    # Add completed todos back and test toggle all active
    TodoStore.todos = TodoStore.todos.value + [
        TodoItem("comp1", "Completed 1", True),
        TodoItem("comp2", "Completed 2", True),
    ]
    assert len(TodoStore.todos.value) == 4
    assert TodoStore.active_count.value == 2
    assert TodoStore.completed_count.value == 2

    TodoStore.toggle_all_active()
    assert len(TodoStore.todos.value) == 4
    assert TodoStore.total_count.value == 4
    assert TodoStore.active_count.value == 0
    assert TodoStore.completed_count.value == 4
    assert all(t.completed for t in TodoStore.todos.value)

    # Test toggle all completed (toggles all completed to active)
    TodoStore.toggle_all_completed()
    assert len(TodoStore.todos.value) == 4
    assert TodoStore.active_count.value == 4
    assert TodoStore.completed_count.value == 0
    assert all(not t.completed for t in TodoStore.todos.value)

    # Reset to mixed state and test toggle all completed
    TodoStore.todos = [
        TodoItem("act1", "Active", False),
        TodoItem("comp1", "Completed", True),
    ]
    TodoStore.toggle_all_completed()
    assert len(TodoStore.todos.value) == 2
    assert TodoStore.active_count.value == 2
    assert TodoStore.completed_count.value == 0
    assert all(not t.completed for t in TodoStore.todos.value)

    print("✅ Bulk operations work")


def test_stats_empty_state():
    """Test stats computation when there are no todos."""
    TodoStore.todos = []

    assert TodoStore.stats_text.value == "No todos yet. Add one above!"
    assert TodoStore.total_count.value == 0
    assert TodoStore.active_count.value == 0
    assert TodoStore.completed_count.value == 0
    assert TodoStore.has_no_todos.value is True
    assert TodoStore.all_todos_completed.value is True  # vacuously true for empty set
    assert TodoStore.has_any_completed.value is False

    print("✅ Empty state stats work")


def test_stats_all_active():
    """Test stats computation when all todos are active."""
    TodoStore.todos = [TodoItem.create("Task 1"), TodoItem.create("Task 2")]

    assert "2 active todos remaining" in TodoStore.stats_text.value
    assert TodoStore.total_count.value == 2
    assert TodoStore.active_count.value == 2
    assert TodoStore.completed_count.value == 0
    assert TodoStore.has_no_todos.value is False
    assert TodoStore.all_todos_completed.value is False
    assert TodoStore.has_any_completed.value is False

    print("✅ All active stats work")


def test_stats_mixed_state():
    """Test stats computation with both active and completed todos."""
    TodoStore.todos = [TodoItem.create("Active"), TodoItem("comp1", "Completed", True)]

    assert "1 active, 1 completed" in TodoStore.stats_text.value
    assert TodoStore.total_count.value == 2
    assert TodoStore.active_count.value == 1
    assert TodoStore.completed_count.value == 1
    assert TodoStore.has_no_todos.value is False
    assert TodoStore.all_todos_completed.value is False
    assert TodoStore.has_any_completed.value is True

    print("✅ Mixed state stats work")


def test_stats_all_completed():
    """Test stats computation when all todos are completed."""
    TodoStore.todos = [
        TodoItem("comp1", "Completed 1", True),
        TodoItem("comp2", "Completed 2", True),
    ]

    assert "All 2 todos completed!" in TodoStore.stats_text.value
    assert TodoStore.total_count.value == 2
    assert TodoStore.active_count.value == 0
    assert TodoStore.completed_count.value == 2
    assert TodoStore.has_no_todos.value is False
    assert TodoStore.all_todos_completed.value is True
    assert TodoStore.has_any_completed.value is True

    print("✅ All completed stats work")


def test_stats_single_active():
    """Test stats computation with a single active todo."""
    TodoStore.todos = [TodoItem.create("Single Task")]

    assert (
        "1 active todo remaining" in TodoStore.stats_text.value
    )  # Note: no 's' for singular
    assert TodoStore.total_count.value == 1
    assert TodoStore.active_count.value == 1
    assert TodoStore.completed_count.value == 0

    print("✅ Single active stats work")


def test_stats_single_completed():
    """Test stats computation with a single completed todo."""
    TodoStore.todos = [TodoItem("done1", "Single Completed", True)]

    assert "All 1 todos completed!" in TodoStore.stats_text.value
    assert TodoStore.total_count.value == 1
    assert TodoStore.active_count.value == 0
    assert TodoStore.completed_count.value == 1

    print("✅ Single completed stats work")


def test_stats_computation():
    """Run all stats computation tests."""
    test_stats_empty_state()
    test_stats_all_active()
    test_stats_mixed_state()
    test_stats_all_completed()
    test_stats_single_active()
    test_stats_single_completed()

    print("✅ Stats computation works")


if __name__ == "__main__":
    print("🧪 Running comprehensive TodoStore functionality tests...")

    test_basic_crud_operations()
    test_filtering()
    test_bulk_operations()
    test_stats_computation()

    print("🎉 All tests passed! TodoStore functionality is working correctly.")
