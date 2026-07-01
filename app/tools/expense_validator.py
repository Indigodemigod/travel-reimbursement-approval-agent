from decimal import Decimal
from typing import Any

from app.models import ExpenseCategory, ExpenseItem

HOTEL_MAX_INR = Decimal("5000")
MEALS_MAX_INR = Decimal("1500")

PERSONAL_FUEL_KEYWORDS = (
    "fuel",
    "petrol",
    "diesel",
    "gasoline",
    "personal vehicle",
    "own car",
    "mileage",
)

ALLOWED_TRANSPORT_KEYWORDS = (
    "taxi",
    "cab",
    "metro",
    "bus",
    "train",
    "uber",
    "ola",
    "subway",
    "rail",
)


def _description_lower(expense: ExpenseItem) -> str:
    return expense.description.lower()


def _is_personal_fuel(expense: ExpenseItem) -> bool:
    description = _description_lower(expense)
    return any(keyword in description for keyword in PERSONAL_FUEL_KEYWORDS)


def _is_allowed_transport(expense: ExpenseItem) -> bool:
    description = _description_lower(expense)
    return any(keyword in description for keyword in ALLOWED_TRANSPORT_KEYWORDS)


def validate_expenses(expenses: list[ExpenseItem]) -> dict[str, Any]:
    """Validate expenses against company travel policy limits and categories."""
    approved_amount = Decimal("0")
    rejected_amount = Decimal("0")
    manual_review = False
    violations: list[str] = []

    for expense in expenses:
        description = expense.description
        amount = expense.amount
        currency = expense.currency
        category = expense.category

        if currency != "INR":
            manual_review = True
            violations.append(
                f"Foreign currency claim requires manual review: {currency}"
            )
            continue

        if category == ExpenseCategory.OTHER:
            manual_review = True
            violations.append(
                f"Unsupported expense category: {description}"
            )
            continue

        if category == ExpenseCategory.HOTEL and amount > HOTEL_MAX_INR:
            rejected_amount += amount
            violations.append(
                f"Hotel expense exceeds ₹5,000 per night limit: "
                f"{description} (₹{amount})"
            )
            continue

        if category == ExpenseCategory.MEALS and amount > MEALS_MAX_INR:
            rejected_amount += amount
            violations.append(
                f"Meals expense exceeds ₹1,500 per day limit: "
                f"{description} (₹{amount})"
            )
            continue

        if category == ExpenseCategory.TRANSPORT:
            if _is_personal_fuel(expense):
                rejected_amount += amount
                violations.append(
                    f"Personal vehicle fuel is not reimbursable: "
                    f"{description} (₹{amount})"
                )
                continue

            if not _is_allowed_transport(expense):
                manual_review = True
                violations.append(
                    f"Unsupported transport mode: {description}"
                )
                continue

        approved_amount += amount

    return {
        "is_valid": len(violations) == 0 and not manual_review,
        "approved_amount": approved_amount,
        "rejected_amount": rejected_amount,
        "manual_review": manual_review,
        "violations": violations,
    }
