"""Tests for duplicate detection tool."""

from app.models import ExpenseCategory
from app.tools.duplicate_checker import check_duplicates
from tests.conftest import make_claim, make_expense


def test_duplicate_line_items() -> None:
    expense = make_expense(ExpenseCategory.MEALS, "400", description="Lunch")
    claim = make_claim("DUP-LINE-1", [expense, expense])

    result = check_duplicates(claim)

    assert result["has_duplicates"] is True
    assert result["is_valid"] is False
    assert any(
        item["type"] == "within_claim_line_item" for item in result["duplicates"]
    )


def test_no_duplicates() -> None:
    claim = make_claim(
        "UNIQUE-1",
        [
            make_expense(ExpenseCategory.MEALS, "400", description="Lunch"),
            make_expense(ExpenseCategory.TRANSPORT, "300", description="Taxi"),
        ],
    )

    result = check_duplicates(claim)

    assert result["has_duplicates"] is False
    assert result["is_valid"] is True
    assert result["duplicates"] == []


def test_duplicate_claim_id() -> None:
    claim = make_claim(
        "DUP-ID-1",
        [make_expense(ExpenseCategory.MEALS, "400", description="Lunch")],
    )

    first = check_duplicates(claim)
    second = check_duplicates(claim)

    assert first["has_duplicates"] is False
    assert second["has_duplicates"] is True
    assert any(item["type"] == "duplicate_claim_id" for item in second["duplicates"])


def test_duplicate_submission_fingerprint() -> None:
    expenses = [make_expense(ExpenseCategory.MEALS, "400", description="Lunch")]
    first_claim = make_claim("FINGER-1", expenses)
    second_claim = make_claim("FINGER-2", expenses)

    first = check_duplicates(first_claim)
    second = check_duplicates(second_claim)

    assert first["has_duplicates"] is False
    assert second["has_duplicates"] is True
    assert any(
        item["type"] == "duplicate_submission" for item in second["duplicates"]
    )


def test_duplicate_plus_normal_expense() -> None:
    duplicate = make_expense(ExpenseCategory.MEALS, "400", description="Lunch")
    claim = make_claim(
        "DUP-MIX-1",
        [
            duplicate,
            duplicate,
            make_expense(ExpenseCategory.TRANSPORT, "250", description="Metro"),
        ],
    )

    result = check_duplicates(claim)

    assert result["has_duplicates"] is True
    assert len(result["duplicates"]) == 1
    assert result["duplicates"][0]["type"] == "within_claim_line_item"
