"""Tests for policy lookup tool."""

from app.tools.policy_lookup import NO_MATCH_MESSAGE, lookup_policy


def test_hotel_query() -> None:
    result = lookup_policy("hotel accommodation nightly rate")

    assert result["section_titles"]
    assert any("Hotel" in title for title in result["section_titles"])
    assert "Hotel" in result["policy_text"]


def test_meals_query() -> None:
    result = lookup_policy("meals daily allowance lunch dinner")

    assert result["section_titles"]
    assert any("Meals" in title for title in result["section_titles"])
    assert "Meals" in result["policy_text"]


def test_transport_query() -> None:
    result = lookup_policy("taxi metro local transport")

    assert result["section_titles"]
    assert any("Transport" in title for title in result["section_titles"])
    assert "taxi" in result["policy_text"].lower()


def test_mixed_query() -> None:
    result = lookup_policy("business trip hotel meals taxi receipts")

    assert len(result["section_titles"]) <= 3
    assert len(result["section_titles"]) >= 1
    assert result["policy_text"]
    assert "---" in result["policy_text"] or len(result["section_titles"]) == 1


def test_unknown_query() -> None:
    result = lookup_policy("xyzunknownterm123")

    assert result["policy_text"] == NO_MATCH_MESSAGE
    assert result["section_titles"] == []
