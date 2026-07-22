"""Static typing regression tests for the public observable operators."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.unit
@pytest.mark.observable
def test_operator_type_inference_is_flat_and_source_preserving():
    """Mypy sees FynX's operator algebra the same way users do."""
    pytest.importorskip("mypy")

    source = textwrap.dedent(
        """
        from typing import Literal, Optional, TypedDict, cast

        from fynx import Observable
        from fynx import Store, StoreSnapshot, StoreState, observable, reactive

        height: Observable[float] = Observable("height", 1.8)
        weight: Observable[float] = Observable("weight", 75.0)

        def calculate_bmi(height: float, weight: float) -> float:
            return weight / (height * height)

        bmi_data = height + weight
        reveal_type(bmi_data)

        bmi = bmi_data >> calculate_bmi
        reveal_type(bmi)

        a: Observable[int] = Observable("a", 1)
        b: Observable[str] = Observable("b", "x")
        c: Observable[bool] = Observable("c", True)

        raw_then = a.then(lambda value: reveal_type(value) or value + 1)
        reveal_type(raw_then)

        triple = a + b + c
        reveal_type(triple)

        label = triple >> (lambda i, s, flag: s if flag else str(i))
        reveal_type(label)
        reveal_type(label.subscribe)

        def on_label(value: str) -> None:
            pass

        label_subscription = label.subscribe(on_label)
        reveal_type(label_subscription)

        all_condition = a & c
        reveal_type(all_condition)

        filtered = a @ c
        reveal_type(filtered)

        choice = c | (a >> (lambda i: i > 0))
        reveal_type(choice)

        negated = ~c
        reveal_type(negated)

        class Cart(Store):
            total = observable(0)
            logged_in = observable(False)
            items = observable([1])
            optional_age = observable(cast(Optional[int], None))
            tags = observable(cast(list[str], []))

        class Preferences(TypedDict):
            theme: Literal["light", "dark"]
            notifications: bool

        class Profile(Store):
            preferences = observable(
                cast(Preferences, {"theme": "light", "notifications": True})
            )

        reveal_type(Cart.total)
        reveal_type(Cart.logged_in)
        reveal_type(Cart.items + [2])
        reveal_type(Cart.total.subscribe)

        is_adult = Cart.optional_age >> (
            lambda age: age is not None and age >= 18
        )
        reveal_type(Cart.optional_age)
        reveal_type(is_adult)

        eligible_age = Cart.optional_age @ is_adult
        reveal_type(eligible_age)

        tag_label = Cart.tags.then(lambda tags: ", ".join(tags))
        reveal_type(tag_label)

        selected_theme = Profile.preferences.then(lambda pref: pref["theme"])
        reveal_type(selected_theme)

        store_product = Cart.total + Cart.logged_in
        reveal_type(store_product)

        store_then = Cart.total.then(lambda value: reveal_type(value) or value + 1)
        reveal_type(store_then)

        store_ready = Cart.logged_in & (Cart.total >> (lambda total: total > 0))
        reveal_type(store_ready)

        store_filtered = Cart.total @ Cart.logged_in
        reveal_type(store_filtered)

        raw_store_filtered = a @ Cart.logged_in
        reveal_type(raw_store_filtered)

        store_choice = Cart.logged_in | (Cart.total >> (lambda total: total > 0))
        reveal_type(store_choice)

        raw_store_choice = (a.then(lambda value: value > 0)) | Cart.logged_in
        reveal_type(raw_store_choice)

        raw_store_choice_method = (a.then(lambda value: value > 0)).either(
            Cart.logged_in
        )
        reveal_type(raw_store_choice_method)

        age_category = Cart.total.then(
            lambda age: (
                "unknown"
                if age is None
                else ("minor" if age < 18 else "adult" if age < 65 else "senior")
            )
        )
        reveal_type(age_category)

        saved_state = Cart.to_dict()
        reveal_type(saved_state)
        typed_state: StoreState = saved_state
        Cart.load_state(typed_state)

        class UserProfile(Store):
            is_active = observable(True)
            is_verified = observable(False)
            subscription_tier = observable("premium")
            first_name = observable("")
            last_name = observable("")
            email = observable("")
            phone = observable("")

        @reactive(UserProfile)
        def on_profile_change(profile_snapshot: StoreSnapshot) -> None:
            reveal_type(profile_snapshot)

        reveal_type(on_profile_change)

        account_status = (
            UserProfile.is_active
            + UserProfile.is_verified
            + UserProfile.subscription_tier
        ).then(lambda active, verified, tier: reveal_type(active) or "")

        profile_completeness = (
            UserProfile.first_name
            + UserProfile.last_name
            + UserProfile.email
            + UserProfile.phone
        ).then(lambda fn, ln, em, ph: reveal_type(fn) or 0.0)

        d: Observable[float] = Observable("d", 1.0)
        e: Observable[bytes] = Observable("e", b"x")
        f: Observable[complex] = Observable("f", 1j)
        g: Observable[list[int]] = Observable("g", [1])
        h: Observable[dict[str, int]] = Observable("h", {"x": 1})
        i: Observable[set[str]] = Observable("i", {"x"})
        j: Observable[tuple[int, str]] = Observable("j", (1, "x"))

        high_arity_product = a + b + c + d + e + f + g + h + i + j
        reveal_type(high_arity_product.subscribe)
        high_arity_product.subscribe(
            lambda av, bv, cv, dv, ev, fv, gv, hv, iv, jv: reveal_type(jv)
        )

        @reactive(a, b, c, d, e, f, g, h, i, j)
        def on_many_change(
            av: int,
            bv: str,
            cv: bool,
            dv: float,
            ev: bytes,
            fv: complex,
            gv: list[int],
            hv: dict[str, int],
            iv: set[str],
            jv: tuple[int, str],
        ) -> None:
            pass

        reveal_type(on_many_change)
        """
    )

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    def assert_revealed(*renderings: str) -> None:
        assert any(
            f'Revealed type is "{rendering}"' in result.stdout
            for rendering in renderings
        ), result.stdout

    assert (
        'Revealed type is "fynx.observable.merged.MergedObservable'
        '[builtins.float, builtins.float]"'
    ) in result.stdout
    assert (
        'Revealed type is "fynx.observable.base.Observable[builtins.float]"'
    ) in result.stdout
    assert (
        result.stdout.count(
            'Revealed type is "fynx.observable.base.Observable[builtins.int]"'
        )
        >= 1
    )
    assert result.stdout.count('Revealed type is "builtins.list[builtins.int]"') >= 1
    assert_revealed(
        "fynx.observable.descriptors.ObservableValue[builtins.int | None]",
        "fynx.observable.descriptors.ObservableValue[Union[builtins.int, None]]",
    )
    assert (
        result.stdout.count(
            'Revealed type is "fynx.observable.base.Observable[builtins.bool]"'
        )
        >= 6
    )
    assert_revealed(
        "fynx.observable.conditional.ConditionalObservable[builtins.int | None]",
        "fynx.observable.conditional.ConditionalObservable[Union[builtins.int, None]]",
    )
    assert (
        'Revealed type is "fynx.observable.base.Observable[builtins.str]"'
    ) in result.stdout
    assert (
        'Revealed type is "def (func: def (builtins.str) -> builtins.object) -> '
        'fynx.observable.base.Observable[builtins.str]"'
    ) in result.stdout
    assert_revealed(
        "fynx.observable.base.Observable[Literal['light'] | Literal['dark']]",
        "fynx.observable.base.Observable" "[Union[Literal['light'], Literal['dark']]]",
    )
    assert (
        'Revealed type is "fynx.observable.merged.MergedObservable'
        '[builtins.int, builtins.str, builtins.bool]"'
    ) in result.stdout
    assert (
        'Revealed type is "fynx.observable.base.Observable[builtins.str]"'
    ) in result.stdout
    assert (
        "Revealed type is "
        '"fynx.observable.conditional.ConditionalObservable[builtins.int]"'
    ) in result.stdout
    assert (
        result.stdout.count(
            "Revealed type is "
            '"fynx.observable.conditional.ConditionalObservable[builtins.int]"'
        )
        >= 2
    )
    assert result.stdout.count('Revealed type is "builtins.int"') >= 2
    assert (
        'Revealed type is "fynx.observable.base.Observable[builtins.str]"'
    ) in result.stdout
    assert 'Revealed type is "fynx.store.StoreSnapshot"' in result.stdout
    assert (
        "Revealed type is "
        '"fynx.reactive.ReactiveWrapper[[fynx.store.StoreSnapshot], None]"'
    ) in result.stdout
    assert 'Revealed type is "builtins.str"' in result.stdout
    assert 'Revealed type is "builtins.bool"' in result.stdout
    assert 'Revealed type is "tuple[builtins.int, builtins.str]"' in result.stdout
    assert (
        'Revealed type is "def (func: def (builtins.int, builtins.str, '
        "builtins.bool, builtins.float, builtins.bytes, builtins.complex, "
        "builtins.list[builtins.int], builtins.dict[builtins.str, builtins.int], "
        "builtins.set[builtins.str], tuple[builtins.int, builtins.str]) -> "
        "builtins.object) -> fynx.observable.merged.MergedObservable"
        "[builtins.int, builtins.str, builtins.bool, builtins.float, "
        "builtins.bytes, builtins.complex, builtins.list[builtins.int], "
        "builtins.dict[builtins.str, builtins.int], builtins.set[builtins.str], "
        'tuple[builtins.int, builtins.str]]"'
    ) in result.stdout
    assert (
        "Revealed type is "
        '"fynx.reactive.ReactiveWrapper[[builtins.int, builtins.str, '
        "builtins.bool, builtins.float, builtins.bytes, builtins.complex, "
        "builtins.list[builtins.int], builtins.dict[builtins.str, builtins.int], "
        'builtins.set[builtins.str], tuple[builtins.int, builtins.str]], None]"'
    ) in result.stdout
    assert (
        'Revealed type is "fynx.observable.descriptors.ObservableValue[builtins.int]"'
    ) in result.stdout
    assert (
        'Revealed type is "def (func: def (builtins.int) -> builtins.object) -> '
        'fynx.observable.base.Observable[builtins.int]"'
    ) in result.stdout
    assert (
        'Revealed type is "fynx.observable.descriptors.ObservableValue[builtins.bool]"'
    ) in result.stdout
    assert (
        'Revealed type is "fynx.observable.merged.MergedObservable'
        '[builtins.int, builtins.bool]"'
    ) in result.stdout
    assert (
        "Revealed type is "
        '"fynx.observable.conditional.ConditionalObservable[builtins.int]"'
    ) in result.stdout
