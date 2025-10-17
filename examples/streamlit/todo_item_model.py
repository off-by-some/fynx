#!/usr/bin/env python3
"""
Fynx Streamlit TODO Item Model
==============================

This module defines the TodoItem data structure used throughout the Fynx TODO application.
The TodoItem represents an immutable todo item with a unique identifier, text content,
and completion status.

The module provides:
- Immutable TodoItem dataclass with unique ID generation
- Factory method for creating new todo items
- Method for toggling completion status immutably

This separation allows the data model to be imported independently of the store logic,
promoting better modularity and testability.

Example:
    ```python
    from todo_item_model import TodoItem

    # Create a new todo
    todo = TodoItem.create("Buy groceries")

    # Toggle completion (returns new instance)
    completed_todo = todo.toggle_completion()
    ```
"""

import uuid
from dataclasses import dataclass

# ==============================================================================================
# Configuration and Constants
# ==============================================================================================

# UUID configuration
UUID_STRING_LENGTH = 36

# ==============================================================================================
# TodoItem - Immutable Todo Data Structure
# ==============================================================================================


@dataclass(frozen=True)
class TodoItem:
    """
    An immutable data structure representing a single todo item.

    Each todo item has a unique identifier, text content, and completion status.
    The frozen dataclass ensures immutability, requiring new instances to be
    created for any modifications (such as toggling completion status).

    This design promotes functional programming principles and makes the
    todo items safe to use in reactive systems without unexpected mutations.

    Attributes:
        id: Unique identifier for the todo item (UUID string).
        text: The text content/description of the todo item.
        completed: Boolean indicating whether the todo is completed.

    Example:
        ```python
        # Create a new todo
        todo = TodoItem.create("Buy groceries")

        # Toggle completion (returns new instance)
        completed_todo = todo.toggle_completion()

        # Check properties
        print(f"ID: {todo.id[:8]}...")
        print(f"Text: {todo.text}")
        print(f"Completed: {todo.completed}")
        ```
    """

    id: str
    text: str
    completed: bool = False

    @classmethod
    def create(cls, text: str, completed: bool = False) -> "TodoItem":
        """
        Create a new TodoItem instance with a unique UUID.

        This factory method generates a unique identifier for each todo item,
        ensuring that no two todo items will ever have the same ID, even if
        they have identical text content.

        Args:
            text: The text content for the todo item.
            completed: Initial completion status (default: False).

        Returns:
            A new TodoItem instance with a unique identifier.

        Note:
            The ID is generated using uuid.uuid4() to ensure uniqueness
            across all todo items in the application. The UUID is converted
            to a string for easier storage and serialization.
        """
        return cls(id=str(uuid.uuid4()), text=text, completed=completed)

    def toggle_completion(self) -> "TodoItem":
        """
        Create a new TodoItem with the opposite completion status.

        Due to the immutable nature of TodoItem instances, this method
        returns a new instance rather than modifying the existing one.
        This ensures thread safety and predictable behavior in reactive
        systems.

        Returns:
            A new TodoItem instance with the same id and text, but with
            the completed status toggled.

        Example:
            ```python
            active_todo = TodoItem.create("Task")
            completed_todo = active_todo.toggle_completion()

            assert active_todo.completed == False
            assert completed_todo.completed == True
            assert active_todo.id == completed_todo.id
            ```
        """
        return self.__class__(id=self.id, text=self.text, completed=not self.completed)


# ==============================================================================================
# Test Suite - TodoItem Functionality Validation
# ==============================================================================================


def test_todo_item_creation_and_immutability():
    """
    Test TodoItem creation, properties, and immutability guarantees.

    Verifies that TodoItem instances are created correctly with unique IDs,
    maintain their properties, and that toggle operations return new instances
    without modifying the original.
    """
    test_description = "Buy groceries"
    todo_item = TodoItem.create(test_description)

    # Verify basic properties
    assert todo_item.text == test_description
    assert not todo_item.completed
    assert isinstance(todo_item.id, str)
    assert len(todo_item.id) == UUID_STRING_LENGTH  # Standard UUID4 length
    assert len(todo_item.text.strip()) > 0

    # Test immutability: toggle should return new instance
    completed_version = todo_item.toggle_completion()
    assert completed_version.completed
    assert completed_version.id == todo_item.id  # Same ID
    assert completed_version.text == todo_item.text  # Same text
    assert completed_version is not todo_item  # Different object
    assert not todo_item.completed  # Original unchanged

    # Test multiple toggles maintain immutability
    active_version = completed_version.toggle_completion()
    assert not active_version.completed
    assert active_version.id == todo_item.id  # Same ID throughout
    assert active_version.text == todo_item.text  # Same text throughout
    assert active_version is not completed_version  # Different objects

    print("✅ TodoItem creation and immutability works correctly")


if __name__ == "__main__":
    print("🧪 Testing TodoItem model functionality...")
    test_todo_item_creation_and_immutability()
    print("🎉 TodoItem model tests passed!")
