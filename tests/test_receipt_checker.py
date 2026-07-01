"""Tests for receipt checking tool."""

from app.models import ExpenseCategory
from app.tools.receipt_checker import check_receipts
from tests.conftest import make_expense


def test_receipt_present() -> None:
    result = check_receipts(
        [
            make_expense(
                ExpenseCategory.MEALS,
                "800",
                description="Dinner",
                receipt_attached=True,
            )
        ]
    )

    assert result["is_valid"] is True
    assert result["missing_receipts"] == []


def test_receipt_missing_below_threshold() -> None:
    result = check_receipts(
        [
            make_expense(
                ExpenseCategory.MEALS,
                "400",
                description="Snack",
                receipt_attached=False,
            )
        ]
    )

    assert result["is_valid"] is True
    assert result["missing_receipts"] == []


def test_receipt_missing_above_threshold() -> None:
    result = check_receipts(
        [
            make_expense(
                ExpenseCategory.MEALS,
                "600",
                description="Dinner",
                receipt_attached=False,
            )
        ]
    )

    assert result["is_valid"] is False
    assert len(result["missing_receipts"]) == 1
    assert result["missing_receipts"][0]["description"] == "Dinner"


def test_multiple_missing_receipts() -> None:
    result = check_receipts(
        [
            make_expense(
                ExpenseCategory.HOTEL,
                "4500",
                description="Hotel",
                receipt_attached=False,
            ),
            make_expense(
                ExpenseCategory.MEALS,
                "700",
                description="Lunch",
                receipt_attached=False,
            ),
        ]
    )

    assert result["is_valid"] is False
    assert len(result["missing_receipts"]) == 2


def test_all_receipts_attached() -> None:
    result = check_receipts(
        [
            make_expense(
                ExpenseCategory.HOTEL,
                "4500",
                description="Hotel",
                receipt_attached=True,
            ),
            make_expense(
                ExpenseCategory.MEALS,
                "700",
                description="Lunch",
                receipt_attached=True,
            ),
        ]
    )

    assert result["is_valid"] is True
    assert result["missing_receipts"] == []
