"""Tests for expense validation tool."""

from decimal import Decimal

from app.models import ExpenseCategory
from app.tools.expense_validator import validate_expenses
from tests.conftest import make_expense


def test_hotel_within_limit() -> None:
    result = validate_expenses(
        [make_expense(ExpenseCategory.HOTEL, "4500", description="Hotel stay")]
    )

    assert result["is_valid"] is True
    assert result["manual_review"] is False
    assert result["approved_amount"] == Decimal("4500")
    assert result["rejected_amount"] == Decimal("0")
    assert result["violations"] == []


def test_hotel_exceeds_limit() -> None:
    result = validate_expenses(
        [make_expense(ExpenseCategory.HOTEL, "6000", description="Luxury hotel")]
    )

    assert result["is_valid"] is False
    assert result["manual_review"] is False
    assert result["approved_amount"] == Decimal("0")
    assert result["rejected_amount"] == Decimal("6000")
    assert len(result["violations"]) == 1
    assert "Hotel expense exceeds" in result["violations"][0]


def test_meals_within_limit() -> None:
    result = validate_expenses(
        [make_expense(ExpenseCategory.MEALS, "1200", description="Team lunch")]
    )

    assert result["is_valid"] is True
    assert result["approved_amount"] == Decimal("1200")
    assert result["rejected_amount"] == Decimal("0")


def test_meals_exceeds_limit() -> None:
    result = validate_expenses(
        [make_expense(ExpenseCategory.MEALS, "2000", description="Dinner")]
    )

    assert result["is_valid"] is False
    assert result["rejected_amount"] == Decimal("2000")
    assert "Meals expense exceeds" in result["violations"][0]


def test_personal_fuel_rejected() -> None:
    result = validate_expenses(
        [
            make_expense(
                ExpenseCategory.TRANSPORT,
                "1500",
                description="Petrol for personal car",
            )
        ]
    )

    assert result["is_valid"] is False
    assert result["rejected_amount"] == Decimal("1500")
    assert "Personal vehicle fuel" in result["violations"][0]


def test_taxi_approved() -> None:
    result = validate_expenses(
        [make_expense(ExpenseCategory.TRANSPORT, "350", description="Uber taxi")]
    )

    assert result["is_valid"] is True
    assert result["manual_review"] is False
    assert result["approved_amount"] == Decimal("350")


def test_unsupported_transport_manual_review() -> None:
    result = validate_expenses(
        [
            make_expense(
                ExpenseCategory.TRANSPORT,
                "800",
                description="Helicopter charter",
            )
        ]
    )

    assert result["is_valid"] is False
    assert result["manual_review"] is True
    assert result["approved_amount"] == Decimal("0")
    assert "Unsupported transport mode" in result["violations"][0]


def test_foreign_currency_manual_review() -> None:
    result = validate_expenses(
        [
            make_expense(
                ExpenseCategory.HOTEL,
                "200",
                currency="USD",
                description="US hotel",
            )
        ]
    )

    assert result["is_valid"] is False
    assert result["manual_review"] is True
    assert "Foreign currency" in result["violations"][0]


def test_other_category_manual_review() -> None:
    result = validate_expenses(
        [make_expense(ExpenseCategory.OTHER, "500", description="Gift")]
    )

    assert result["is_valid"] is False
    assert result["manual_review"] is True
    assert "Unsupported expense category" in result["violations"][0]
