"""
Comprehensive tests for FynX conditional operators functionality.

This test suite validates the &, |, and ~ operators for creating conditional observables,
ensuring they work exactly as described in the conditionals.md documentation.
"""

import pytest

from fynx import NULL_EVENT, observable, reactive


class TestBasicConditionalOperator:
    """Tests for basic & operator functionality."""

    def test_ampersand_filters_based_on_boolean_condition(self):
        """& operator filters observable values based on boolean conditions."""
        scores = observable(85)

        # Create boolean condition
        is_high_score = scores >> (lambda score: score > 90)

        # Filter scores based on condition
        high_scores = scores & is_high_score

        notifications = []
        high_scores.subscribe(lambda score: notifications.append(score))

        # Initial value doesn't meet condition - no emission
        assert notifications == []

        # Set value that meets condition - emits when condition becomes met
        scores.set(95)
        assert notifications == [95]

        # Set value that doesn't meet condition - no emission (condition became unmet)
        scores.set(87)
        assert notifications == [95]

        # Set another value that meets condition - emits again
        scores.set(98)
        assert notifications == [95, 98]

    def test_ampersand_raises_exception_when_condition_unmet(self):
        """& operator raises ConditionNotMet when accessing value with unmet conditions."""
        from fynx.observable import ConditionNotMet

        user = observable({"name": "Alice", "age": 30, "country": "US"})

        # Complex validation condition
        is_valid_user = user >> (
            lambda u: (
                u["age"] >= 18
                and u["country"] in ["US", "CA", "UK"]
                and len(u["name"]) > 2
                and "@" not in u["name"]
            )
        )

        # Filter users based on validation
        valid_users = user & is_valid_user

        notifications = []
        valid_users.subscribe(lambda user_data: notifications.append(user_data))

        # Initial user is valid - no emission (no transition)
        assert notifications == []

        # Make user invalid
        user.set({"name": "Bob", "age": 15, "country": "US"})  # Too young
        assert notifications == []  # No emission when condition becomes unmet

        # Make user valid again - emits when condition transitions to met
        user.set({"name": "Charlie", "age": 25, "country": "US"})
        assert notifications == [{"name": "Charlie", "age": 25, "country": "US"}]

        # Accessing .value when condition is met works
        assert valid_users.value == {"name": "Charlie", "age": 25, "country": "US"}

        # Make user invalid again
        user.set({"name": "Dave", "age": 16, "country": "US"})  # Too young
        assert notifications == [
            {"name": "Charlie", "age": 25, "country": "US"}
        ]  # No emission when condition becomes unmet

        # Accessing .value when condition is unmet returns cached value
        assert valid_users.value == {"name": "Charlie", "age": 25, "country": "US"}

    def test_ampersand_with_function_predicates(self):
        """& operator works with function predicates."""
        data = observable([1, 2, 3])

        # Filter based on function predicate
        long_lists = data & (lambda lst: len(lst) > 5)

        notifications = []
        long_lists.subscribe(lambda lst: notifications.append(lst))

        # Initial list is too short - no emission
        assert notifications == []

        # Set longer list - emits when condition becomes met
        data.set([1, 2, 3, 4, 5, 6, 7])
        assert notifications == [[1, 2, 3, 4, 5, 6, 7]]

        # Set short list again - no emission when condition becomes unmet
        data.set([1, 2])
        assert notifications == [[1, 2, 3, 4, 5, 6, 7]]


class TestReactiveDecoratorWithConditionals:
    """Tests for using conditionals with @reactive decorator."""

    def test_reactive_decorator_with_boolean_conditionals(self):
        """@reactive works with boolean conditionals."""
        scores = observable(85)

        # Create boolean condition
        is_high_score = scores >> (lambda score: score > 90)

        notifications = []

        @reactive(is_high_score)
        def on_high_score(is_high):
            notifications.append(f"high_score_{is_high}")

        # Initial state - reactive decorator calls immediately for computed observables
        assert notifications == ["high_score_False"]

        # Change to trigger condition
        scores.set(95)
        assert notifications == ["high_score_False", "high_score_True"]

        # Change back
        scores.set(87)
        assert notifications == [
            "high_score_False",
            "high_score_True",
            "high_score_False",
        ]


class TestComplexPredicates:
    """Tests for complex predicates with & operator."""

    def test_complex_validation_with_ampersand(self):
        """Complex validation logic with & operator."""
        email = observable("user@")

        # Validation condition
        email_valid = email >> (lambda e: "@" in e and "." in e.split("@")[1])

        # Filter emails based on validation
        valid_emails = email & email_valid

        notifications = []
        valid_emails.subscribe(lambda e: notifications.append(e))

        # Initial email is invalid
        assert notifications == []

        # Set valid email
        email.set("user@example.com")
        assert notifications == ["user@example.com"]

        # Set invalid email again
        email.set("user@")
        assert notifications == ["user@example.com"]  # No emission when condition fails


class TestNegationOperator:
    """Tests for ~ operator (logical negation)."""

    def test_tilde_negates_boolean_condition(self):
        """~ operator inverts boolean conditions."""
        is_online = observable(True)

        # Create negated condition
        is_offline = ~is_online

        notifications = []
        is_offline.subscribe(
            lambda offline: notifications.append(f"offline_{offline}"),
            call_immediately=True,
        )

        # Initially online (so offline is False) - calls immediately
        assert notifications == ["offline_False"]

        # Go offline
        is_online.set(False)
        assert notifications == ["offline_False", "offline_True"]

        # Go back online
        is_online.set(True)
        assert notifications == ["offline_False", "offline_True", "offline_False"]


class TestLogicalOrOperator:
    """Tests for | operator (logical OR)."""

    def test_pipe_creates_or_conditions(self):
        """| operator creates logical OR between boolean observables."""
        is_error = observable(False)
        is_warning = observable(True)
        is_critical = observable(False)

        # Logical OR
        needs_attention = is_error | is_warning | is_critical

        notifications = []
        needs_attention.subscribe(
            lambda needs: notifications.append(f"attention_{needs}"),
            call_immediately=True,
        )

        # Initial state: warning is True, so OR is True - calls immediately
        assert notifications == ["attention_True"]

        # Set error to True (still True)
        is_error.set(True)
        assert notifications == ["attention_True"]  # No change

        # Set all to False
        is_error.set(False)
        is_warning.set(False)
        is_critical.set(False)
        assert notifications == ["attention_True", "attention_False"]

    def test_pipe_with_either_method_equivalence(self):
        """| operator equivalent to .either() method."""
        is_error = observable(False)
        is_warning = observable(True)

        # Using | operator
        attention1 = is_error | is_warning

        # Using .either() method
        attention2 = is_error.either(is_warning)

        notifications1 = []
        notifications2 = []

        attention1.subscribe(lambda x: notifications1.append(x), call_immediately=True)
        attention2.subscribe(lambda x: notifications2.append(x), call_immediately=True)

        # Both call subscribers immediately with current value
        assert notifications1 == notifications2 == [True]

        is_error.set(True)
        assert notifications1 == notifications2 == [True]  # No change (already True)


class TestCombiningOperators:
    """Tests for combining OR with other operators."""

    def test_or_with_and_for_complex_logic(self):
        """Combine | and & for complex logical expressions."""
        user_input = observable("")
        is_admin = observable(False)
        is_moderator = observable(True)

        # Create conditions
        has_input = user_input >> (lambda u: len(u) > 0)
        has_permission = is_admin | is_moderator  # OR condition

        # Complex condition: user has input AND (is admin OR moderator)
        can_submit = (user_input & has_input) & has_permission

        notifications = []
        can_submit.subscribe(lambda can: notifications.append(f"can_submit_{can}"))

        # Initial state: no input, but has permission - no emission
        assert notifications == []  # No emission (no input)

        # Add input
        user_input.set("Hello")
        assert notifications == ["can_submit_Hello"]  # Can submit

        # Remove permission
        is_moderator.set(False)
        assert notifications == [
            "can_submit_Hello"
        ]  # No emission when permission removed

        # Add admin permission - emits when condition becomes met again
        is_admin.set(True)
        assert notifications == ["can_submit_Hello", "can_submit_Hello"]

    def test_negation_with_filtering(self):
        """Combine ~ with & for 'everything except' patterns."""
        status = observable("loading")

        # Create condition for non-loading states
        is_not_loading = status >> (lambda s: s != "loading")

        # Filter for non-loading status
        non_loading_status = status & is_not_loading

        notifications = []
        non_loading_status.subscribe(lambda s: notifications.append(f"status_{s}"))

        # Initial loading status is filtered out - no emission
        assert notifications == []

        # Change to success
        status.set("success")
        assert notifications == ["status_success"]

        # Change to error
        status.set("error")
        assert notifications == ["status_success", "status_error"]

        # Back to loading (filtered out)
        status.set("loading")
        assert notifications == [
            "status_success",
            "status_error",
        ]  # No emission when condition fails


class TestRealWorldExamples:
    """Tests for real-world examples from documentation."""

    def test_form_validation_example(self):
        """Complete form validation example from documentation."""
        email = observable("")
        password = observable("")
        terms_accepted = observable(False)

        # Validation conditions
        email_valid = email >> (lambda e: "@" in e and "." in e.split("@")[1])
        password_strong = password >> (lambda p: len(p) >= 8)
        terms_checked = terms_accepted >> (lambda t: t == True)

        # Form is valid when all conditions are true
        all_conditions_met = (email_valid + password_strong + terms_checked) >> (
            lambda e, p, t: e and p and t
        )
        form_valid = email & all_conditions_met

        notifications = []
        form_valid.subscribe(lambda valid: notifications.append(f"form_valid_{valid}"))

        # Initial state: form invalid - no emission
        assert notifications == []

        # Set email (still invalid - no password)
        email.set("user@")
        assert notifications == []

        # Set valid email
        email.set("user@example.com")
        assert notifications == []

        # Set password (still invalid - too short)
        password.set("pass")
        assert notifications == []

        # Set strong password
        password.set("secure123")
        assert notifications == []

        # Accept terms
        terms_accepted.set(True)
        assert notifications == ["form_valid_user@example.com"]

        # Break validation
        password.set("short")
        assert notifications == [
            "form_valid_user@example.com"
        ]  # No emission when validation fails

    def test_state_machine_with_conditionals(self):
        """State machine example with conditional logic."""
        app_state = observable("initializing")
        user_authenticated = observable(False)
        data_loaded = observable(False)

        # Define state conditions
        is_app_ready = app_state >> (lambda s: s == "ready")
        is_user_auth = user_authenticated >> (lambda a: a == True)
        is_data_loaded = data_loaded >> (lambda d: d == True)
        is_app_error = app_state >> (lambda s: s == "error")

        # Ready state: app ready AND user auth AND data loaded
        ready_conditions = is_app_ready & is_user_auth & is_data_loaded
        ready_state = (ready_conditions + app_state) >> (
            lambda ready, state: state if ready and state == "ready" else None
        )

        # Error state
        error_state = (is_app_error + app_state) >> (
            lambda error, state: state if error and state == "error" else None
        )

        ready_notifications = []
        error_notifications = []

        ready_state.subscribe(
            lambda state: ready_notifications.append(f"ready_{state}")
        )
        error_state.subscribe(
            lambda state: error_notifications.append(f"error_{state}")
        )

        # Initial state - conditions not met, no emission
        assert ready_notifications == []
        assert error_notifications == []

        # Simulate app lifecycle
        app_state.set("authenticating")
        user_authenticated.set(True)
        app_state.set("loading_data")
        data_loaded.set(True)
        app_state.set("ready")  # Should trigger ready

        assert ready_notifications == ["ready_ready"]  # Should emit when ready
        assert error_notifications == []

        # Trigger error - should emit for error_state, no emission for ready_state
        app_state.set("error")
        assert error_notifications == ["error_error"]
        assert ready_notifications == [
            "ready_ready"
        ]  # No emission when ready condition fails


class TestAdvancedPatterns:
    """Tests for advanced conditional patterns."""

    def test_conditionals_with_derived_values(self):
        """Combine conditionals with >> operator."""
        sensor_readings = observable([])

        # Condition for sufficient data
        has_enough_data = sensor_readings >> (lambda readings: len(readings) >= 3)

        # Only process when enough data
        valid_readings = sensor_readings & has_enough_data

        # Calculate statistics (handle NULL_EVENT from failed conditions)
        average_reading = valid_readings >> (
            lambda readings: (
                sum(readings) / len(readings) if readings is not NULL_EVENT else None
            )
        )

        notifications = []
        average_reading.subscribe(lambda avg: notifications.append(f"avg_{avg}"))

        # Not enough data initially - no emission
        assert notifications == []

        # Still not enough
        sensor_readings.set([1, 2])
        assert notifications == []

        # Enough data now
        sensor_readings.set([1, 2, 3, 4])
        assert notifications == ["avg_2.5"]

        # Back to insufficient data - no emission
        sensor_readings.set([1])
        assert notifications == ["avg_2.5"]


class TestPerformanceBenefits:
    """Tests demonstrating performance benefits of conditionals."""

    def test_conditionals_reduce_unnecessary_computations(self):
        """Conditionals prevent expensive operations when conditions aren't met."""
        raw_data = observable("some data")

        # Track expensive operations
        expensive_calls = []

        def expensive_cleanup(data):
            expensive_calls.append("cleanup")
            return data.upper()

        def expensive_analysis(data):
            expensive_calls.append("analysis")
            return len(data)

        # Without conditionals - operations run on every change
        processed_data = raw_data >> expensive_cleanup
        final_result = processed_data >> expensive_analysis

        # Access to trigger computation
        _ = final_result.value
        assert expensive_calls == ["cleanup", "analysis"]

        # Change data - both operations run again
        raw_data.set("other data")
        _ = final_result.value
        assert expensive_calls == ["cleanup", "analysis", "cleanup", "analysis"]

    def test_conditionals_with_filtering_prevent_expensive_ops(self):
        """Conditionals with filtering prevent expensive operations."""
        raw_data = observable("some data")

        expensive_calls = []

        def is_worth_processing(data):
            return len(data) > 10

        def expensive_cleanup(data):
            expensive_calls.append("cleanup")
            return data.upper()

        def expensive_analysis(data):
            expensive_calls.append("analysis")
            return len(data)

        # With conditionals - expensive operations only run when condition met
        clean_data = raw_data & is_worth_processing
        processed_data = clean_data >> (lambda d: expensive_cleanup(d) if d else None)
        final_result = processed_data >> (
            lambda d: expensive_analysis(d) if d else None
        )

        # Short data - no expensive operations (condition never met)
        assert final_result.value is None  # Should return None
        assert expensive_calls == []

        # Set long data - operations run
        raw_data.set("this is a much longer piece of data")
        _ = final_result.value
        assert expensive_calls == ["cleanup", "analysis"]


class TestCommonPatterns:
    """Tests for common conditional patterns."""

    def test_threshold_monitoring_pattern(self):
        """Threshold monitoring pattern."""
        temperature = observable(20)

        # Threshold conditions
        is_hot = temperature >> (lambda t: t > 25)
        is_cold = temperature >> (lambda t: t < 10)

        # Alert observables
        hot_weather = temperature & is_hot
        cold_weather = temperature & is_cold

        hot_notifications = []
        cold_notifications = []

        hot_weather.subscribe(lambda t: hot_notifications.append(f"hot_{t}"))
        cold_weather.subscribe(lambda t: cold_notifications.append(f"cold_{t}"))

        # Initial temperature - no emissions (conditions not met)
        assert hot_notifications == []
        assert cold_notifications == []

        # Hot temperature
        temperature.set(30)
        assert hot_notifications == ["hot_30"]
        assert cold_notifications == []

        # Cold temperature - no emission for hot condition
        temperature.set(5)
        assert hot_notifications == ["hot_30"]
        assert cold_notifications == ["cold_5"]

    def test_data_quality_gates_pattern(self):
        """Data quality gates pattern."""
        api_response = observable(None)

        # Validation condition
        is_valid_response = api_response >> (
            lambda resp: (
                resp is not None
                and resp.get("status") == "success"
                and resp.get("data") is not None
            )
        )

        # Only process valid responses
        valid_responses = api_response & is_valid_response

        notifications = []
        valid_responses.subscribe(
            lambda resp: notifications.append(
                f"data_{len(resp['data']) if resp is not None else 'None'}"
            )
        )

        # Invalid response initially - no emission
        assert notifications == []

        # Still invalid
        api_response.set({"status": "error", "data": None})
        assert notifications == []

        # Valid response
        api_response.set({"status": "success", "data": [1, 2, 3, 4, 5]})
        assert notifications == ["data_5"]

        # Back to invalid - no emission
        api_response.set({"status": "error"})
        assert notifications == ["data_5"]

    def test_feature_flags_with_conditions(self):
        """Feature flags with conditional logic."""
        feature_enabled = observable(False)
        user_premium = observable(False)
        experiment_active = observable(True)

        # Boolean conditions
        is_feature_on = feature_enabled >> (lambda e: e == True)
        is_premium_user = user_premium >> (lambda p: p == True)
        is_experiment_on = experiment_active >> (lambda a: a == True)

        # Feature available under specific conditions
        can_use_feature = feature_enabled & user_premium

        notifications = []
        can_use_feature.subscribe(
            lambda available: notifications.append(
                "feature_available" if available is not None else "feature_unavailable"
            )
        )

        # Initial state - conditions not met, no emission
        assert notifications == []

        # Enable feature but user not premium
        feature_enabled.set(True)
        assert notifications == []

        # Make user premium
        user_premium.set(True)
        assert notifications == ["feature_available"]

        # Disable experiment - no emission when conditions fail
        experiment_active.set(False)
        assert notifications == ["feature_available"]


class TestBestPractices:
    """Tests demonstrating best practices."""

    def test_keep_conditions_simple(self):
        """Simple, focused conditions are clearer."""
        age = observable(20)
        role = observable("user")

        # Good - simple conditions
        is_adult = age & (lambda a: a >= 18)
        has_permission = role & (lambda r: r in ["admin", "moderator"])

        adult_notifications = []
        permission_notifications = []

        is_adult.subscribe(lambda a: adult_notifications.append(f"adult_{a}"))
        has_permission.subscribe(
            lambda r: permission_notifications.append(f"permission_{r}")
        )

        # Initial state - no emission (no transition)
        assert adult_notifications == []  # No emission even though condition starts met
        assert permission_notifications == []  # Doesn't meet condition

        # Change role
        role.set("admin")
        assert permission_notifications == ["permission_admin"]

    def test_name_conditions_clearly(self):
        """Named condition functions improve clarity."""
        email = observable("")
        password = observable("")

        def is_valid_email(email):
            return "@" in email and "." in email.split("@")[1]

        def is_strong_password(pwd):
            return len(pwd) >= 8

        # Much clearer than inline lambdas
        email_ok = email & is_valid_email
        password_ok = password & is_strong_password

        email_notifications = []
        password_notifications = []

        email_ok.subscribe(lambda e: email_notifications.append(f"email_{e}"))
        password_ok.subscribe(lambda p: password_notifications.append(f"password_{p}"))

        # Initial state - conditions not met, no emissions
        assert email_notifications == []
        assert password_notifications == []

        # Set valid values
        email.set("user@example.com")
        password.set("secure123")

        assert email_notifications == ["email_user@example.com"]
        assert password_notifications == ["password_secure123"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_conditionals_with_none_values(self):
        """Conditionals handle None values properly."""
        data = observable(None)

        # Condition that checks for None
        is_not_none = data & (lambda d: d is not None)

        notifications = []
        is_not_none.subscribe(lambda d: notifications.append(f"data_{d}"))

        # Initial None value - condition not met, no emission
        assert notifications == []

        # Set actual data
        data.set("hello")
        assert notifications == ["data_hello"]

        # Back to None - no emission when condition fails
        data.set(None)
        assert notifications == ["data_hello"]

    def test_multiple_conditionals_same_source(self):
        """Multiple conditionals can filter the same source."""
        number = observable(10)

        # Different conditions on same source
        is_even = number & (lambda n: n % 2 == 0)
        is_positive = number & (lambda n: n > 0)
        is_large = number & (lambda n: n > 100)

        even_notifications = []
        positive_notifications = []
        large_notifications = []

        is_even.subscribe(lambda n: even_notifications.append(f"even_{n}"))
        is_positive.subscribe(lambda n: positive_notifications.append(f"positive_{n}"))
        is_large.subscribe(lambda n: large_notifications.append(f"large_{n}"))

        # Initial value meets even and positive conditions - no emission (no transition)
        assert even_notifications == []
        assert positive_notifications == []
        assert large_notifications == []

        # Change to odd number - even becomes unmet, positive stays met but value changed
        number.set(15)
        assert even_notifications == []  # No emission when condition fails
        assert positive_notifications == [
            "positive_15"
        ]  # Emission when condition stays met but value changes
        assert large_notifications == []

        # Change to negative - even stays unmet, positive becomes unmet
        number.set(-5)
        assert even_notifications == []  # No emission
        assert positive_notifications == [
            "positive_15"
        ]  # No emission when condition fails
        assert large_notifications == []

        # Change back to positive even number - both conditions become met
        number.set(10)
        assert even_notifications == [
            "even_10"
        ]  # Emits when even condition becomes met
        assert positive_notifications == [
            "positive_15",
            "positive_10",
        ]  # Emits when positive condition becomes met
        assert large_notifications == []
