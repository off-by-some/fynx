"""
Streamlit Store Example - Reactive State Management with Streamlit Session State

This example demonstrates how to create a custom Store subclass that automatically
synchronizes observable values with Streamlit's session state, enabling seamless
integration between FynX's reactive system and Streamlit's state management.

The StreamlitStore provides:
- Automatic synchronization between observables and session state
- Transparent reactive behavior within Streamlit apps
- Easy persistence of state across reruns
- Bidirectional sync that works both ways (observable → session state and vice versa)

To run this example:
    $ pip install fynx streamlit && python run examples/streamlit/store.py
"""

import logging
from typing import Any, Dict, Set

from fynx import Store
from fynx.observable.computed import ComputedObservable
from fynx.store import SessionValue, StoreMeta

# ==============================================================================================
# Constants
# ==============================================================================================

# Session state synchronization
SESSION_KEY_PREFIX = "fynx_session_sync"
SYNC_LOG_PREFIX = "💾 Synced"
BULK_SYNC_LOG_PREFIX = "🔄 Synced"
INIT_LOG_PREFIX = "📝 Initialized"
STATE_LOG_PREFIX = "📊"
ERROR_LOG_PREFIX = "❌ Error"

# Logging configuration
LOG_LEVEL = logging.INFO
MAX_LOG_STRING_LENGTH = 50
MAX_LOG_LIST_ITEMS = 3

# UUID configuration
UUID_STRING_LENGTH = 36

# We can't add streamlit as a dependency to fynx, because it constrains versions that
# fynx actually supports, so we need to check if it is installed. It must be installed outside of poetry.
try:
    import streamlit as st  # type: ignore

    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None  # type: ignore
    STREAMLIT_AVAILABLE = False

# ==============================================================================================
# Logging Configuration
# ==============================================================================================

# Set up logging for the store module
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Configure console handler for clean output
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(message)s")  # Simplified format for cleaner output
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Suppress noisy Streamlit warnings when running script directly
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)


# ==============================================================================================
# StreamlitStoreMeta - Metaclass for Session State Synchronization
# ==============================================================================================


class StreamlitStoreMeta(StoreMeta):
    """
    Metaclass for StreamlitStore that automatically synchronizes observable changes to session state.

    This metaclass intercepts attribute assignments on Store classes and ensures that primitive
    observable values are automatically synchronized with Streamlit's session state when available.
    Computed observables are excluded from synchronization as they are derived from primitive values.

    The metaclass provides seamless bidirectional synchronization between FynX observables and
    Streamlit's session state, enabling reactive state management that persists across app reruns.

    Attributes:
        None: This is a metaclass and doesn't define instance attributes.

    Example:
        ```python
        class MyStore(StreamlitStore):
            counter = observable(0)
            name = observable("")

        # These assignments automatically sync to st.session_state
        MyStore.counter = 42
        MyStore.name = "Alice"
        ```
    """

    def __setattr__(cls, attribute_name: str, value: Any) -> None:
        """
        Intercept attribute assignment and synchronize primitive observables to session state.

        This method is called whenever an attribute is assigned on a class that uses this metaclass.
        It first performs the normal attribute assignment, then checks if the attribute is a primitive
        observable that should be synchronized to Streamlit session state.

        Args:
            attribute_name: The name of the attribute being assigned.
            value: The new value being assigned to the attribute.

        Raises:
            AttributeError: If session state synchronization fails due to missing Streamlit context.
                          This is caught and logged internally.
        """
        # Perform the normal attribute assignment first
        super().__setattr__(attribute_name, value)

        # Skip synchronization if conditions aren't met
        if not cls._should_sync_to_session_state(attribute_name):
            return

        # Get the observable and prepare synchronization
        current_observable = getattr(cls, "_observables")[attribute_name]
        session_key = f"{cls.__name__}_{attribute_name}"
        observable_current_value = current_observable.value

        try:
            cls._sync_observable_to_session_state(session_key, observable_current_value)
        except AttributeError as session_error:
            # Streamlit session_state not available (e.g., outside Streamlit context)
            logger.debug(f"Session state not available for sync: {session_error}")
            pass

    @classmethod
    def _should_sync_to_session_state(cls, attribute_name: str) -> bool:
        """
        Determine whether an attribute should be synchronized to Streamlit session state.

        This method checks several conditions to ensure safe synchronization:
        1. Streamlit must be available in the environment
        2. The class must have an _observables registry
        3. The attribute name must exist in the observables registry
        4. The observable must be primitive (not computed)

        Args:
            attribute_name: The name of the attribute to evaluate for synchronization.

        Returns:
            True if the attribute is a primitive observable that should be synchronized
            to session state, False otherwise.

        Note:
            Computed observables are excluded from synchronization because they are
            derived from primitive values and would create unnecessary sync operations.
        """
        # Check if Streamlit is available in the current environment
        if not STREAMLIT_AVAILABLE:
            return False

        # Ensure the class has an observables registry
        if not hasattr(cls, "_observables"):
            return False

        # Get the observables registry and check if attribute exists
        class_observables = getattr(cls, "_observables", {})
        if attribute_name not in class_observables:
            return False

        # Only sync primitive observables, not computed ones
        target_observable = class_observables[attribute_name]
        return not isinstance(target_observable, ComputedObservable)

    @classmethod
    def _sync_observable_to_session_state(
        cls, session_key: str, observable_current_value: Any
    ) -> None:
        """
        Synchronize an observable's current value to Streamlit session state.

        This method updates the session state only if the value has actually changed,
        preventing unnecessary updates and potential performance issues.

        Args:
            session_key: The unique key used to store this observable in session state.
                        Format: "{ClassName}_{attribute_name}"
            observable_current_value: The current value of the observable to synchronize.

        Note:
            This method performs a change detection check to avoid redundant session
            state updates, which is important for performance in reactive applications.
        """
        # Only update session state if the value has actually changed
        session_state = st.session_state  # type: ignore
        if (
            session_key not in session_state
            or session_state[session_key] != observable_current_value
        ):
            session_state[session_key] = observable_current_value


# ==============================================================================================
# StreamlitStore - Reactive Store with Session State Integration
# ==============================================================================================


class StreamlitStore(Store, metaclass=StreamlitStoreMeta):
    """
    A reactive store implementation with automatic Streamlit session state synchronization.

    This store extends FynX's base Store class with seamless integration to Streamlit's
    session state, enabling reactive state management that automatically persists across
    app reruns. All primitive observable attributes are automatically synchronized
    bidirectionally between the store and Streamlit's session state.

    Key Features:
        - Automatic bidirectional synchronization with Streamlit session state
        - Transparent reactive behavior within Streamlit applications
        - State persistence across application reruns without manual intervention
        - Full compatibility with all existing FynX reactive features
        - Graceful fallback to standard FynX behavior when Streamlit is unavailable

    Attributes:
        Inherited from Store: All observable attributes defined in subclasses are
        automatically synchronized with session state.

    Example:
        ```python
        class TodoStore(StreamlitStore):
            task_input = observable("")
            tasks = observable([])
            filter_status = observable("all")

        # State automatically syncs with session state
        TodoStore.task_input = "Buy groceries"
        # This value is now available in st.session_state['TodoStore_task_input']
        ```

    Note:
        Streamlit integration is optional. If Streamlit is not available in the environment,
        the store falls back to normal FynX behavior without session state synchronization.
        This allows the same store classes to work in both Streamlit and non-Streamlit contexts.
    """

    @classmethod
    def _setup_session_state_sync(cls) -> None:
        """
        Initialize automatic session state synchronization for observable changes.

        This method sets up a subscription to observable changes that automatically
        synchronizes all primitive observable values to Streamlit session state
        whenever they change. The setup is performed only once per class.

        The method creates a change handler that:
        1. Checks if Streamlit is available
        2. Iterates through all primitive observables
        3. Updates session state only when values have actually changed

        Note:
            This method uses a guard attribute (_session_sync_setup) to ensure
            the synchronization is set up only once, preventing duplicate subscriptions.
        """
        # Prevent duplicate setup
        if hasattr(cls, "_session_sync_setup"):
            return

        def handle_observable_changes(changed_snapshot: Dict[str, Any]) -> None:
            """
            Handle observable changes by synchronizing primitive values to session state.

            This inner function is called whenever any observable in the store changes.
            It ensures that only primitive (non-computed) observables are synchronized
            and only when their values have actually changed.

            Args:
                changed_snapshot: Dictionary containing information about what changed.
                                 This parameter is required by the subscription interface
                                 but not directly used in this implementation.
            """
            # Skip if Streamlit is not available
            if not STREAMLIT_AVAILABLE:
                return

            # Get all primitive observable names and sync each one
            primitive_attribute_names = cls._get_primitive_observable_names()
            for attribute_name in primitive_attribute_names:
                current_observable = cls._observables[attribute_name]
                session_state_key = f"{cls.__name__}_{attribute_name}"
                observable_current_value = current_observable.value

                # Only update if the value has actually changed
                if cls._needs_session_state_update(
                    session_state_key, observable_current_value
                ):
                    cls._update_session_state(
                        session_state_key, attribute_name, observable_current_value
                    )

        # Subscribe to changes and mark setup as complete
        cls.subscribe(handle_observable_changes)
        cls._session_sync_setup = True

    @classmethod
    def _get_primitive_observable_names(cls) -> Set[str]:
        """
        Retrieve the names of all primitive (non-computed) observable attributes.

        This method filters the class's observable registry to return only the names
        of primitive observables, excluding computed observables which are derived
        from other values and don't need direct synchronization.

        Returns:
            A set of attribute names corresponding to primitive observables in this store.

        Note:
            Primitive observables are the core state values that drive the application's
            reactive behavior. Computed observables are automatically updated when their
            dependencies change, so they don't need explicit session state synchronization.
        """
        class_observables = getattr(cls, "_observables", {})
        return {
            attribute_name
            for attribute_name in class_observables
            if not isinstance(class_observables[attribute_name], ComputedObservable)
        }

    @classmethod
    def _needs_session_state_update(
        cls, session_state_key: str, observable_current_value: Any
    ) -> bool:
        """
        Determine if session state requires an update for the given observable value.

        This method performs change detection to avoid unnecessary session state updates,
        which is important for performance in reactive applications.

        Args:
            session_state_key: The key used to identify this observable in session state.
            observable_current_value: The current value of the observable to check.

        Returns:
            True if session state should be updated because the value has changed
            or doesn't exist in session state, False otherwise.

        Note:
            This check prevents redundant updates and potential performance issues
            that could occur from frequent but unnecessary session state writes.
        """
        session_state = st.session_state  # type: ignore
        return (
            session_state_key not in session_state
            or session_state[session_state_key] != observable_current_value
        )

    @classmethod
    def _update_session_state(
        cls, session_state_key: str, observable_attribute_name: str, new_value: Any
    ) -> None:
        """
        Update Streamlit session state with a new observable value and log the change.

        This method performs the actual session state update and provides logging
        for debugging and monitoring synchronization activity.

        Args:
            session_state_key: The unique key used to store this observable in session state.
            observable_attribute_name: The name of the observable attribute being updated.
            new_value: The new value to store in session state.

        Note:
            Logging is performed at the INFO level to track synchronization activity
            without being overly verbose in production applications.
        """
        logger.info(
            f"{SYNC_LOG_PREFIX} {cls.__name__}.{observable_attribute_name}: {new_value!r}"
        )
        st.session_state[session_state_key] = new_value  # type: ignore

    @classmethod
    def sync_from_session_state(cls) -> None:
        """
        Restore observable state from Streamlit session state during app initialization.

        This method should be called once when the Streamlit app starts to restore
        the store's state from persisted session state. It performs two key operations:

        1. Updates observables with values restored from session state
        2. Initializes session state with default values for new observables

        The method uses a guard attribute to ensure synchronization happens only once
        per application session, preventing redundant operations.

        Note:
            This method is essential for maintaining state across Streamlit app reruns.
            Without it, the store would reset to default values on each rerun instead
            of maintaining user state.

        Raises:
            None: Exceptions during synchronization are logged but not raised to prevent
                  app crashes. The store continues with default values if sync fails.
        """
        # Prevent duplicate synchronization and check Streamlit availability
        if hasattr(cls, "_synced_from_session") or not STREAMLIT_AVAILABLE:
            return

        # Ensure the synchronization system is properly initialized
        cls._setup_session_state_sync()

        try:
            # Get all primitive observables that need synchronization
            primitive_attribute_names = cls._get_primitive_observable_names()
            restored_attribute_names = []
            initialized_attribute_names = []

            # Process each primitive observable
            for attribute_name in primitive_attribute_names:
                current_observable = cls._observables[attribute_name]
                session_state_key = f"{cls.__name__}_{attribute_name}"

                if session_state_key in st.session_state:  # type: ignore
                    # Restore value from session state if it exists and differs
                    persisted_value = st.session_state[session_state_key]  # type: ignore
                    current_value = current_observable.value

                    if persisted_value != current_value:
                        current_observable.set(persisted_value)
                        restored_attribute_names.append(attribute_name)
                else:
                    # Initialize session state with the observable's default value
                    st.session_state[session_state_key] = current_observable.value  # type: ignore
                    initialized_attribute_names.append(attribute_name)

            # Log synchronization results and mark as complete
            cls._log_session_state_sync_results(
                restored_attribute_names, initialized_attribute_names
            )
            cls._synced_from_session = True

        except Exception as synchronization_error:
            logger.error(
                f"{ERROR_LOG_PREFIX} Failed to sync from session state: {synchronization_error}"
            )
            # Continue with default values rather than crashing the app

    @classmethod
    def _log_session_state_sync_results(
        cls, restored_attribute_names: list[str], initialized_attribute_names: list[str]
    ) -> None:
        """
        Log the results of session state synchronization operations.

        This method provides informative logging about what happened during
        the synchronization process, helping with debugging and monitoring.

        Args:
            restored_attribute_names: Names of attributes that were restored from session state.
            initialized_attribute_names: Names of attributes that were newly initialized
                                       in session state with default values.
        """
        if restored_attribute_names:
            logger.info(
                f"{BULK_SYNC_LOG_PREFIX} Restored {len(restored_attribute_names)} attributes "
                f"from session state: {', '.join(restored_attribute_names)}"
            )
        if initialized_attribute_names:
            logger.info(
                f"{INIT_LOG_PREFIX} Initialized {len(initialized_attribute_names)} attributes "
                f"in session state: {', '.join(initialized_attribute_names)}"
            )

    @classmethod
    def log_current_state(cls) -> None:
        """
        Log the current state of all primitive observables in the store.

        This method creates a formatted summary of all primitive observable values
        and logs it for debugging and monitoring purposes. Long values are truncated
        for readability while preserving essential information.

        The log output includes the class name and a dictionary-like representation
        of all primitive observable states, making it easy to understand the
        current application state at a glance.
        """
        primitive_attribute_names = cls._get_primitive_observable_names()
        state_entries = []

        for attribute_name in primitive_attribute_names:
            current_observable = cls._observables[attribute_name]
            current_value = current_observable.value
            formatted_value = cls._format_observable_value_for_logging(current_value)
            state_entries.append(f"{attribute_name}={formatted_value}")

        state_summary = ", ".join(state_entries)
        logger.info(f"{STATE_LOG_PREFIX} {cls.__name__} state: {{{state_summary}}}")

    @classmethod
    def _format_observable_value_for_logging(cls, value: Any) -> str:
        """
        Format an observable value for logging with appropriate truncation.

        This method provides human-readable representations of values for logging,
        truncating long collections and strings to prevent log spam while still
        conveying useful information about the data.

        Args:
            value: The observable value to format for logging.

        Returns:
            A string representation suitable for logging, with long values truncated
            for readability.

        Note:
            - Lists/tuples longer than 3 items show item count instead of contents
            - Strings longer than 50 characters are truncated with ellipsis
            - Other values use their standard repr() representation
        """
        if isinstance(value, (list, tuple)) and len(value) > MAX_LOG_LIST_ITEMS:
            return f"[{len(value)} items]"
        elif isinstance(value, str) and len(value) > MAX_LOG_STRING_LENGTH:
            return f'"{value[:MAX_LOG_STRING_LENGTH-3]}..."'
        else:
            return repr(value)

    @classmethod
    def load_state(cls, state_dict: Dict[str, SessionValue]) -> None:
        """
        Load state from a dictionary into observables with Streamlit session state synchronization.

        This method extends the parent implementation by also updating Streamlit's session state
        to reflect the loaded values. This ensures that manually loaded state is also persisted
        in the session state for future app reruns.

        The method first loads state into the observables using the parent Store implementation,
        then synchronizes those values to session state to maintain consistency.

        Args:
            state_dict: Dictionary mapping observable names to their desired values.
                       The keys should match observable attribute names, and values should
                       be compatible with the observables' expected types.

        Note:
            This method does NOT restore from session state, as that would overwrite the
            explicitly loaded values. It's intended for programmatic state loading scenarios.
        """
        # Load state into observables using parent implementation
        super().load_state(state_dict)

        # Synchronize loaded values to session state for persistence
        # (Skip session state restoration to preserve the loaded values)
        if STREAMLIT_AVAILABLE:
            primitive_attribute_names = cls._get_primitive_observable_names()
            for attribute_name in primitive_attribute_names:
                current_observable = cls._observables[attribute_name]
                session_state_key = f"{cls.__name__}_{attribute_name}"
                st.session_state[session_state_key] = current_observable.value  # type: ignore
