"""Deterministic duplicate detection for travel reimbursement claims."""

from typing import Any

from app.models import TravelClaim

_SUBMITTED_CLAIM_IDS: set[str] = set()
_EXPENSE_SIGNATURES: set[str] = set()


def _expense_line_key(expense: Any) -> tuple[str, str, str, str]:
    return (
        expense.category.value,
        str(expense.amount),
        expense.currency,
        expense.description.strip().lower(),
    )


def _build_expense_signature(claim: TravelClaim) -> str:
    line_keys = sorted(_expense_line_key(expense) for expense in claim.expenses)
    return (
        f"{claim.employee_id}|{claim.trip_start_date}|{claim.trip_end_date}|"
        f"{claim.destination.strip().lower()}|{line_keys!r}"
    )


def check_duplicates(claim: TravelClaim) -> dict[str, Any]:
    """Detect duplicate line items and repeated claim submissions."""
    duplicates: list[dict[str, Any]] = []
    seen_lines: dict[tuple[str, str, str, str], int] = {}

    for index, expense in enumerate(claim.expenses):
        line_key = _expense_line_key(expense)
        if line_key in seen_lines:
            duplicates.append(
                {
                    "type": "within_claim_line_item",
                    "expense_index": index,
                    "matches_expense_index": seen_lines[line_key],
                    "category": expense.category.value,
                    "amount": str(expense.amount),
                    "currency": expense.currency,
                    "description": expense.description,
                    "message": (
                        "Duplicate expense line detected within the same claim."
                    ),
                }
            )
        else:
            seen_lines[line_key] = index

    if claim.claim_id in _SUBMITTED_CLAIM_IDS:
        duplicates.append(
            {
                "type": "duplicate_claim_id",
                "claim_id": claim.claim_id,
                "message": (
                    f"Claim ID '{claim.claim_id}' has already been submitted."
                ),
            }
        )

    expense_signature = _build_expense_signature(claim)
    if expense_signature in _EXPENSE_SIGNATURES:
        duplicates.append(
            {
                "type": "duplicate_submission",
                "employee_id": claim.employee_id,
                "claim_id": claim.claim_id,
                "message": (
                    "An identical expense pattern was previously submitted "
                    "for this employee and trip."
                ),
            }
        )

    if not duplicates:
        _SUBMITTED_CLAIM_IDS.add(claim.claim_id)
        _EXPENSE_SIGNATURES.add(expense_signature)

    return {
        "has_duplicates": bool(duplicates),
        "is_valid": not duplicates,
        "duplicates": duplicates,
    }
