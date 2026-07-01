"""Shared pytest fixtures and helpers."""

from datetime import date
from decimal import Decimal

import pytest

from app.models import ExpenseCategory, ExpenseItem, TravelClaim
import app.tools.duplicate_checker as duplicate_checker_module


@pytest.fixture(autouse=True)
def reset_duplicate_registry() -> None:
    """Isolate duplicate checker state across tests."""
    duplicate_checker_module._SUBMITTED_CLAIM_IDS.clear()
    duplicate_checker_module._EXPENSE_SIGNATURES.clear()
    yield
    duplicate_checker_module._SUBMITTED_CLAIM_IDS.clear()
    duplicate_checker_module._EXPENSE_SIGNATURES.clear()


def make_expense(
    category: ExpenseCategory,
    amount: str | Decimal,
    *,
    currency: str = "INR",
    description: str = "Business expense",
    receipt_attached: bool = True,
) -> ExpenseItem:
    return ExpenseItem(
        category=category,
        amount=Decimal(str(amount)),
        currency=currency,
        description=description,
        receipt_attached=receipt_attached,
    )


def make_claim(
    claim_id: str,
    expenses: list[ExpenseItem],
    *,
    employee_id: str = "EMP001",
    employee_name: str = "Jane Doe",
    department: str = "Engineering",
    trip_start_date: date = date(2026, 3, 1),
    trip_end_date: date = date(2026, 3, 5),
    destination: str = "Delhi",
    purpose: str = "Client visit",
) -> TravelClaim:
    return TravelClaim(
        claim_id=claim_id,
        employee_id=employee_id,
        employee_name=employee_name,
        department=department,
        trip_start_date=trip_start_date,
        trip_end_date=trip_end_date,
        destination=destination,
        purpose=purpose,
        expenses=expenses,
    )


def gemini_response_payload(
    decision: str,
    *,
    approved_amount: str = "0",
    rejected_amount: str = "0",
    missing_documents: list[str] | None = None,
    violated_policies: list[str] | None = None,
    confidence: float = 0.95,
    explanation: str = "Test explanation.",
) -> str:
    import json

    return json.dumps(
        {
            "decision": decision,
            "approved_amount": approved_amount,
            "rejected_amount": rejected_amount,
            "missing_documents": missing_documents or [],
            "violated_policies": violated_policies or [],
            "confidence": confidence,
            "explanation": explanation,
        }
    )
