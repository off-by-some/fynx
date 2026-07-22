#!/usr/bin/env python3
"""
Fynx Streamlit TODO Item Model
==============================

Defines the TodoItem data structure used throughout the Fynx TODO
application: an immutable todo item with a unique identifier, text content,
and completion status.

Kept separate from the store logic so the data model can be imported and
tested independently.

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

    The frozen dataclass means any modification, like toggling completion,
    creates a new instance instead of mutating in place - which keeps todo
    items safe to use in a reactive system without unexpected mutations.

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

        Args:
            text: The text content for the todo item.
            completed: Initial completion status (default: False).

        Returns:
            A new TodoItem instance with a unique identifier.

        Note:
            The ID comes from uuid.uuid4(), converted to a string for
            storage and serialization.
        """
        return cls(id=str(uuid.uuid4()), text=text, completed=completed)

    def toggle_completion(self) -> "TodoItem":
        """
        Create a new TodoItem with the opposite completion status.

        Returns a new instance rather than modifying this one, since
        TodoItem is immutable.

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
