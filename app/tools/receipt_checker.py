from decimal import Decimal
from typing import Any

from app.models import ExpenseItem

RECEIPT_THRESHOLD_INR = Decimal("500")


def check_receipts(expenses: list[ExpenseItem]) -> dict[str, Any]:
    """Flag INR expenses above the receipt threshold that lack documentation."""
    missing_receipts: list[dict[str, Any]] = []

    for expense in expenses:
        if (
            expense.currency == "INR"
            and expense.amount > RECEIPT_THRESHOLD_INR
            and not expense.receipt_attached
        ):
            missing_receipts.append(
                {
                    "category": expense.category.value,
                    "amount": str(expense.amount),
                    "currency": expense.currency,
                    "description": expense.description,
                }
            )

    return {
        "is_valid": len(missing_receipts) == 0,
        "missing_receipts": missing_receipts,
    }
