"""Unit tests for morphism classes and parsing."""

import pytest

from fynx.optimizer.morphism import Morphism, MorphismParser


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_repr():
    """Test Morphism.__repr__() method (line 123)"""
    morphism = Morphism.identity()
    assert repr(morphism) == "Morphism(id)"


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_handles_edge_cases_with_empty_parts():
    """MorphismParser handles edge cases that result in special morphisms"""
    # Test cases that should result in special morphisms
    assert MorphismParser.parse(" ∘ ") == Morphism.single("∘")
    assert MorphismParser.parse(" ∘  ∘ ") == Morphism.single("∘  ∘")
    assert MorphismParser.parse("") == Morphism.identity()
    assert MorphismParser.parse("   ") == Morphism.identity()


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_parses_valid_compositions():
    """MorphismParser parses valid composition strings correctly"""
    # Test parsing valid compositions
    result = MorphismParser.parse("f ∘ g")
    expected = Morphism.compose(Morphism.single("f"), Morphism.single("g"))
    assert result == expected

    # Test with empty string should return identity
    result = MorphismParser.parse("")
    assert result == Morphism.identity()

    # Test with whitespace only should return identity
    result = MorphismParser.parse("   ")
    assert result == Morphism.identity()


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_split_composition_with_whitespace():
    """Test MorphismParser._split_composition() with various whitespace scenarios"""
    # Test with leading/trailing whitespace
    result = MorphismParser._split_composition(" f ∘ g ")
    assert result == ["f", "g"]

    # Test with multiple spaces
    result = MorphismParser._split_composition("f   ∘   g")
    assert result == ["f", "g"]


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_split_composition_nested_parentheses():
    """Test MorphismParser._split_composition() with nested parentheses"""
    result = MorphismParser._split_composition("(f ∘ g) ∘ h")
    assert result == ["(f ∘ g)", "h"]

    result = MorphismParser._split_composition("f ∘ (g ∘ h)")
    assert result == ["f", "(g ∘ h)"]


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_split_composition_complex():
    """Test MorphismParser._split_composition() with complex compositions"""
    result = MorphismParser._split_composition("f ∘ g ∘ h")
    assert result == ["f", "g", "h"]

    result = MorphismParser._split_composition("(f ∘ g) ∘ (h ∘ i)")
    assert result == ["(f ∘ g)", "(h ∘ i)"]


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_is_balanced():
    """Test MorphismParser._is_balanced() method"""
    assert MorphismParser._is_balanced("()") is True
    assert MorphismParser._is_balanced("(())") is True
    assert MorphismParser._is_balanced("((()))") is True
    assert MorphismParser._is_balanced("") is True
    assert MorphismParser._is_balanced("(") is False
    assert MorphismParser._is_balanced(")") is False
    assert MorphismParser._is_balanced("())") is False
    assert MorphismParser._is_balanced("(()") is False


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_parse_identity():
    """Test MorphismParser.parse() with identity"""
    result = MorphismParser.parse("id")
    assert result == Morphism.identity()


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_parse_single():
    """Test MorphismParser.parse() with single morphism"""
    result = MorphismParser.parse("f")
    assert result == Morphism.single("f")


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_parse_composition():
    """Test MorphismParser.parse() with composition"""
    result = MorphismParser.parse("f ∘ g")
    expected = Morphism.compose(Morphism.single("f"), Morphism.single("g"))
    assert result == expected


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_parse_nested_parentheses():
    """Test MorphismParser.parse() with nested parentheses"""
    result = MorphismParser.parse("((f ∘ g) ∘ h)")
    expected = Morphism.compose(
        Morphism.compose(Morphism.single("f"), Morphism.single("g")),
        Morphism.single("h"),
    )
    assert result == expected


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_parser_parse_complex():
    """Test MorphismParser.parse() with complex compositions"""
    result = MorphismParser.parse("f ∘ g ∘ h")
    expected = Morphism.compose(
        Morphism.single("f"),
        Morphism.compose(Morphism.single("g"), Morphism.single("h")),
    )
    assert result == expected


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_normalize():
    """Test Morphism.normalize() method"""
    # Test identity normalization
    id_morph = Morphism.identity()
    assert id_morph.normalize() == id_morph

    # Test single morphism normalization
    single_morph = Morphism.single("f")
    assert single_morph.normalize() == single_morph

    # Test composition normalization
    composed = Morphism.compose(Morphism.identity(), Morphism.single("f"))
    assert composed.normalize() == Morphism.single("f")

    composed = Morphism.compose(Morphism.single("f"), Morphism.identity())
    assert composed.normalize() == Morphism.single("f")


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_canonical_form():
    """Test Morphism.canonical_form() method"""
    id_morph = Morphism.identity()
    assert id_morph.canonical_form() == ("identity",)

    single_morph = Morphism.single("f")
    assert single_morph.canonical_form() == ("single", "f")

    composed = Morphism.compose(Morphism.single("f"), Morphism.single("g"))
    expected = ("compose", "single", "f", "single", "g")
    assert composed.canonical_form() == expected


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_equality():
    """Test Morphism equality"""
    id1 = Morphism.identity()
    id2 = Morphism.identity()
    assert id1 == id2

    f1 = Morphism.single("f")
    f2 = Morphism.single("f")
    assert f1 == f2

    # Test composition equality
    comp1 = Morphism.compose(Morphism.single("f"), Morphism.single("g"))
    comp2 = Morphism.compose(Morphism.single("f"), Morphism.single("g"))
    assert comp1 == comp2


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_hash():
    """Test Morphism hashing"""
    id1 = Morphism.identity()
    id2 = Morphism.identity()
    assert hash(id1) == hash(id2)

    f1 = Morphism.single("f")
    f2 = Morphism.single("f")
    assert hash(f1) == hash(f2)


@pytest.mark.unit
@pytest.mark.optimizer
def test_morphism_str():
    """Test Morphism string representation"""
    id_morph = Morphism.identity()
    assert str(id_morph) == "id"

    single_morph = Morphism.single("f")
    assert str(single_morph) == "f"

    composed = Morphism.compose(Morphism.single("f"), Morphism.single("g"))
    assert str(composed) == "(f) ∘ (g)"
