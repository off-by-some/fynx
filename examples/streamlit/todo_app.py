#!/usr/bin/env python3
"""
FynX Streamlit TODO App Example
===============================

This example demonstrates how to build a reactive TODO application using FynX
with Streamlit. It showcases:

- Reactive state management with StreamlitStore (automatic session state sync)
- Computed properties for derived state
- Reactive subscriptions for UI updates
- Integration with Streamlit's component model
- Real-time updates without manual state synchronization
- Automatic persistence across app reruns

To run this example:
```bash
$ pip install fynx streamlit && python run examples/streamlit/todo_app.py
```

"""

import logging
from typing import Callable, Dict, List, Optional

import streamlit as st
from todo_item_model import TodoItem
from todo_store import TodoStore

from fynx import reactive

# ==============================================================================================
# Reactive UI Rerender Handler
# ==============================================================================================

# TODO: Implement reactive UI rerendering without interfering with Streamlit's render cycle
# The current reactive approach causes issues because it triggers during initial render
# Need to find a way to only trigger reruns after user interactions, not during initial render

# ==============================================================================================
# Configuration and Constants
# ==============================================================================================

# Logging configuration
LOG_LEVEL = logging.DEBUG

# UI Constants
MAX_VISIBLE_TODO_TEXT_LENGTH = 50
UI_COLUMN_RATIO_INPUT_BUTTON = [4, 1]
UI_COLUMN_RATIO_CHECKBOX_TEXT_DELETE = [0.1, 0.8, 0.1]

# Filter options
FILTER_OPTION_ALL = "all"
FILTER_OPTION_ACTIVE = "active"
FILTER_OPTION_COMPLETED = "completed"
FILTER_OPTIONS = [FILTER_OPTION_ALL, FILTER_OPTION_ACTIVE, FILTER_OPTION_COMPLETED]

# User Messages
EMPTY_TODO_WARNING_MESSAGE = "Please enter some text for the todo."
TODO_ADDED_SUCCESS_TEMPLATE = "Added: {todo_text}"
TODO_DELETED_SUCCESS_MESSAGE = "Todo deleted!"
BULK_OPERATION_PROCESSING_TEMPLATE = "Processing {count} todos..."
CLEAR_COMPLETED_PROCESSING_MESSAGE = "Clearing completed todos..."
BULK_OPERATION_SUCCESS_TEMPLATE = "Successfully {operation_description}!"
ALL_COMPLETED_INFO_MESSAGE = "All todos are already completed!"
ALL_ACTIVE_INFO_MESSAGE = "All todos are already active!"

# Bulk Operations Configuration
BULK_OPERATION_CONFIGS = [
    {
        "label": "Mark All Complete",
        "action": lambda: TodoStore.toggle_all_active(),
        "target_getter": lambda: TodoStore.active_todos.value,
        "empty_message": ALL_COMPLETED_INFO_MESSAGE,
        "success_template": "Successfully marked {count} todos as complete!",
    },
    {
        "label": "Mark All Active",
        "action": lambda: TodoStore.toggle_all_completed(),
        "target_getter": lambda: TodoStore.completed_todos.value,
        "empty_message": ALL_ACTIVE_INFO_MESSAGE,
        "success_template": "Successfully marked {count} todos as active!",
    },
    {
        "label": "Clear Completed",
        "action": lambda: TodoStore.clear_completed(),
        "target_getter": lambda: TodoStore.completed_todos.value,
        "empty_message": None,  # Always show button for clear
        "success_template": "Completed todos cleared!",
    },
]

# ==============================================================================================
# Logging Setup
# ==============================================================================================

# Configure logging for the application
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


# ==============================================================================================
# UI Utility Functions
# ==============================================================================================


def trigger_ui_rerun() -> None:
    """
    Trigger a Streamlit app rerun to refresh the user interface.

    This function forces Streamlit to rerun the entire application, which is
    necessary when state changes need to be immediately reflected in the UI.
    Use sparingly to avoid performance issues.
    """
    st.rerun()


def display_success_message(message: str) -> None:
    """
    Display a success message using Streamlit's success styling.

    Args:
        message: The success message text to display to the user.
    """
    st.success(message)


def display_warning_message(message: str) -> None:
    """
    Display a warning message using Streamlit's warning styling.

    Args:
        message: The warning message text to display to the user.
    """
    st.warning(message)


def display_info_message(message: str) -> None:
    """
    Display an informational message using Streamlit's info styling.

    Args:
        message: The informational message text to display to the user.
    """
    st.info(message)


# ==============================================================================================
# UI Component Functions
# ==============================================================================================


def render_todo_input_form() -> None:
    """
    Render the input form component for adding new todo items.

    This component provides a text input field and an "Add Todo" button arranged
    in a responsive column layout. It handles input validation, state updates,
    and user feedback for the todo creation process.

    The form validates that input is not empty or whitespace-only before creating
    a new todo item. Upon successful creation, it logs the state change and
    triggers a UI refresh to show the new todo immediately.
    """
    st.header("Add New Todo")

    # Create responsive column layout for input and button
    input_column, button_column = st.columns(UI_COLUMN_RATIO_INPUT_BUTTON)

    with input_column:
        user_input_text = st.text_input(
            "What needs to be done?",
            placeholder="Enter your todo...",
            label_visibility="collapsed",
        )

    with button_column:
        if st.button("Add Todo", type="primary", use_container_width=True):
            # Validate and sanitize input
            sanitized_text = user_input_text.strip()
            if not sanitized_text:
                display_warning_message(EMPTY_TODO_WARNING_MESSAGE)
                return

            # Create new todo and provide feedback
            TodoStore.add_todo(sanitized_text)
            TodoStore.log_current_state()
            display_success_message(
                TODO_ADDED_SUCCESS_TEMPLATE.format(todo_text=sanitized_text)
            )
            trigger_ui_rerun()


def render_todo_filter_controls() -> None:
    """
    Render filter controls component for todo visibility management.

    This component displays a set of radio buttons that allow users to filter
    the visible todos by their completion status. Each filter option shows a
    count of todos in that category, and selecting a filter updates the store
    state to change which todos are displayed.

    The component automatically updates when the underlying todo data changes,
    ensuring that the counts remain accurate.
    """
    st.header("Filter Todos")

    # Define filter count calculation functions
    def get_todo_count_for_filter(filter_name: str) -> int:
        """
        Calculate the number of todos for a given filter type.

        Args:
            filter_name: The name of the filter ("all", "active", or "completed").

        Returns:
            The count of todos matching the specified filter.
        """
        filter_count_functions = {
            FILTER_OPTION_ALL: lambda: len(TodoStore.todos),
            FILTER_OPTION_ACTIVE: lambda: len(TodoStore.active_todos),
            FILTER_OPTION_COMPLETED: lambda: len(TodoStore.completed_todos),
        }
        count_function = filter_count_functions.get(filter_name, lambda: 0)
        return count_function()

    def format_filter_option_with_count(option: str) -> str:
        """
        Format a filter option name with its current count for display.

        Args:
            option: The filter option name.

        Returns:
            A formatted string like "All (5)" showing the option and count.
        """
        current_count = get_todo_count_for_filter(option)
        return f"{option.title()} ({current_count})"

    # Render filter selection radio buttons
    selected_filter_option = st.radio(
        "Show:",
        FILTER_OPTIONS,
        format_func=format_filter_option_with_count,
        horizontal=True,
        label_visibility="collapsed",
    )

    # Update store state if filter selection changed
    if selected_filter_option != TodoStore.filter_mode.value:
        TodoStore.filter_mode = selected_filter_option


def render_todo_statistics() -> None:
    """
    Render the todo completion statistics component.

    This component displays a summary of the current todo state using the
    store's computed statistics text. It shows information about active vs
    completed todos, providing users with an overview of their progress.

    The statistics automatically update when the underlying todo data changes,
    thanks to FynX's reactive computed properties.
    """
    st.info(TodoStore.stats_text.value)


def render_todo_item(todo_item: TodoItem) -> None:
    """
    Render a complete todo item component with all interactive controls.

    This component displays a single todo item in a container with three sections:
    - A completion checkbox on the left
    - The todo text in the center (with strike-through if completed)
    - A delete button on the right

    Args:
        todo_item: The TodoItem instance to render and provide controls for.
    """
    with st.container():
        # Create responsive column layout for todo item controls
        checkbox_column, text_column, delete_column = st.columns(
            UI_COLUMN_RATIO_CHECKBOX_TEXT_DELETE
        )

        _render_completion_checkbox_for_item(checkbox_column, todo_item)
        _render_todo_text_with_formatting(text_column, todo_item)
        _render_delete_button_for_item(delete_column, todo_item)

        st.divider()


def _render_completion_checkbox_for_item(column, todo_item: TodoItem) -> None:
    """
    Render the completion checkbox control for a specific todo item.

    Args:
        column: The Streamlit column container to render the checkbox in.
        todo_item: The TodoItem to create a completion control for.
    """
    with column:
        # Create unique key using truncated ID and completion state for the checkbox
        # This ensures the checkbox gets recreated when the state changes externally
        checkbox_key = f"complete_{todo_item.id[:8]}_{todo_item.completed}"
        checkbox_value = st.checkbox(
            f"Mark '{todo_item.text[:MAX_VISIBLE_TODO_TEXT_LENGTH]}' as completed",
            value=todo_item.completed,
            label_visibility="collapsed",
            key=checkbox_key,
        )

        # Only toggle if the checkbox value differs from the current todo state
        # This handles user clicks while avoiding false triggers during bulk operations
        if checkbox_value != todo_item.completed:
            TodoStore.toggle_todo(todo_item.id)
            trigger_ui_rerun()


def _render_todo_text_with_formatting(column, todo_item: TodoItem) -> None:
    """
    Render the todo text with conditional formatting based on completion status.

    Args:
        column: The Streamlit column container to render the text in.
        todo_item: The TodoItem whose text should be displayed.
    """
    with column:
        if todo_item.completed:
            # Show completed todos with strike-through formatting
            formatted_text = f"~~{todo_item.text}~~"
        else:
            # Show active todos with normal formatting
            formatted_text = todo_item.text

        st.markdown(formatted_text)


def _render_delete_button_for_item(column, todo_item: TodoItem) -> None:
    """
    Render the delete button control for a specific todo item.

    Args:
        column: The Streamlit column container to render the button in.
        todo_item: The TodoItem to create a delete control for.
    """
    with column:
        # Create unique key for the delete button
        delete_button_key = f"delete_{todo_item.id}"
        if st.button("🗑️", key=delete_button_key):
            TodoStore.delete_todo(todo_item.id)
            display_success_message(TODO_DELETED_SUCCESS_MESSAGE)
            trigger_ui_rerun()


def render_todo_list() -> None:
    """
    Render the complete todo list component with filtering and empty states.

    This component displays all todos that match the current filter criteria,
    showing the count in the header. If no todos match the filter, it displays
    an appropriate empty state message based on the current filter mode.

    The list automatically updates when the filter changes or when todos are
    added, completed, or deleted.
    """
    visible_todos = TodoStore.filtered_todos.value
    st.header(f"Todos ({len(visible_todos)})")

    if not visible_todos:
        _render_contextual_empty_state_message()
        return

    # Render each visible todo item
    for current_todo in visible_todos:
        render_todo_item(current_todo)


def _render_contextual_empty_state_message() -> None:
    """
    Render an appropriate empty state message based on the current filter context.

    Different filter modes show different messages to help guide user understanding:
    - "all": Encourages creating the first todo
    - "active": Celebrates completion of all todos
    - "completed": Explains that no todos have been completed yet
    """
    EMPTY_STATE_MESSAGES_BY_FILTER = {
        FILTER_OPTION_COMPLETED: "No completed todos yet. Complete some todos to see them here!",
        FILTER_OPTION_ACTIVE: "No active todos! All caught up! 🎉",
        FILTER_OPTION_ALL: "No todos yet. Add some above to get started!",
    }

    current_filter_mode = TodoStore.filter_mode.value
    contextual_message = EMPTY_STATE_MESSAGES_BY_FILTER.get(
        current_filter_mode, EMPTY_STATE_MESSAGES_BY_FILTER[FILTER_OPTION_ALL]
    )
    st.write(contextual_message)


def render_bulk_actions() -> None:
    """
    Render the bulk operations component for managing multiple todos at once.

    This component provides buttons for common bulk operations like marking all
    todos as complete/active or clearing completed todos. The component only
    renders when there are todos to operate on.

    Each operation shows appropriate feedback messages and handles edge cases
    like attempting operations when no suitable todos exist.
    """
    if not TodoStore.todos.value:
        return

    st.header("Bulk Actions")

    # Create columns for each bulk operation
    operation_columns = st.columns(len(BULK_OPERATION_CONFIGS))
    for column, operation_config in zip(operation_columns, BULK_OPERATION_CONFIGS):
        _render_single_bulk_operation_button(column, operation_config)


def _render_single_bulk_operation_button(column, operation_config: Dict) -> None:
    """
    Render a single bulk operation button with all its associated logic.

    Args:
        column: The Streamlit column container to render the button in.
        operation_config: Configuration dictionary defining the operation behavior.
    """
    with column:
        # Get the todos that would be affected by this operation
        affected_todos = operation_config["target_getter"]()

        # Determine if the button should be shown
        should_display_button = operation_config["empty_message"] is None or bool(
            affected_todos
        )

        # Render button and handle click if conditions are met
        if should_display_button and st.button(
            operation_config["label"], use_container_width=True
        ):
            # Show info message if operation would affect no todos
            if operation_config["empty_message"] and not affected_todos:
                display_info_message(operation_config["empty_message"])
                return

            # Execute the bulk operation
            _perform_bulk_operation(operation_config, affected_todos)


def _perform_bulk_operation(operation_config: Dict, affected_todos: List) -> None:
    """
    Execute a bulk operation and provide user feedback.

    Args:
        operation_config: Configuration dictionary defining the operation.
        affected_todos: List of todos that will be affected by the operation.
    """
    # Display processing message
    if affected_todos:
        processing_feedback = BULK_OPERATION_PROCESSING_TEMPLATE.format(
            count=len(affected_todos)
        )
    else:
        processing_feedback = CLEAR_COMPLETED_PROCESSING_MESSAGE

    display_info_message(processing_feedback)

    # Execute the actual operation
    operation_config["action"]()

    # Display success message
    if affected_todos:
        success_feedback = operation_config["success_template"].format(
            count=len(affected_todos)
        )
    else:
        success_feedback = operation_config["success_template"]

    display_success_message(success_feedback)
    trigger_ui_rerun()


def render_debug_information() -> None:
    """
    Render the debug information component for development and troubleshooting.

    This component provides detailed insight into the current state of the todo store,
    including counts, filter status, and individual todo details. It's collapsible
    to avoid cluttering the main interface during normal use.
    """
    with st.expander("🔍 Debug Info (FynX Reactive State)"):
        _render_comprehensive_store_summary()
        _render_detailed_todo_breakdown()


def _render_comprehensive_store_summary() -> None:
    """
    Render a comprehensive summary of the todo store's current state.

    Displays key metrics and statistics about the todos, including counts
    by status and the current filter setting.
    """
    st.write("**Store State Summary:**")
    current_state_summary = {
        "total_todos": len(TodoStore.todos.value),
        "active_todos": len(TodoStore.active_todos.value),
        "completed_todos": len(TodoStore.completed_todos.value),
        "current_filter_mode": TodoStore.filter_mode.value,
        "visible_todos_count": len(TodoStore.filtered_todos.value),
    }
    st.json(current_state_summary)


def _render_detailed_todo_breakdown() -> None:
    """
    Render detailed information about each individual todo item.

    Shows each todo with its ID, text, and completion status for debugging
    and development purposes.
    """
    st.write("**Individual Todo Details:**")
    for todo_item in TodoStore.todos.value:
        status_indicator = "✅" if todo_item.completed else "⏳"
        status_text = "completed" if todo_item.completed else "active"
        st.write(
            f"- ID {todo_item.id[:8]}: {todo_item.text} ({status_indicator} {status_text})"
        )


def render_todo_app() -> None:
    """
    Render the complete FynX TODO application with all its components.

    This is the main application entry point that orchestrates the rendering
    of all UI components in the correct order. It sets up the application
    structure, initializes the reactive state logging, and composes the
    application from modular, focused components.

    The application follows a logical flow:
    1. Input form for adding todos
    2. Filter controls for visibility management
    3. Statistics summary
    4. Todo list display
    5. Bulk operations
    6. Debug information (collapsible)
    """
    # Set up the main application header
    st.title("🚀 FynX Reactive TODO App")
    st.markdown("*Built with Streamlit + FynX reactive state management*")

    # Log initial state for debugging
    TodoStore.log_current_state()

    # Compose the application from modular components
    render_todo_input_form()
    render_todo_filter_controls()
    render_todo_statistics()
    render_todo_list()
    render_bulk_actions()
    render_debug_information()


if __name__ == "__main__":
    render_todo_app()
