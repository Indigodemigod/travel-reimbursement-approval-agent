"""Edge-case stability tests — application must not crash."""

import json
from decimal import Decimal
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import travel_reimbursement_graph
from app.main import app
from app.models import ExpenseCategory
from app.tools.duplicate_checker import check_duplicates
from app.tools.expense_validator import validate_expenses
from app.tools.policy_lookup import lookup_policy
from app.tools.receipt_checker import check_receipts
from tests.conftest import gemini_response_payload, make_claim, make_expense

client = TestClient(app)


def _base_claim_payload(**overrides) -> dict:
    payload = {
        "claim_id": "EDGE-BASE",
        "employee_id": "EMP001",
        "employee_name": "Jane Doe",
        "department": "Engineering",
        "trip_start_date": "2026-03-01",
        "trip_end_date": "2026-03-05",
        "destination": "Delhi",
        "purpose": "Client visit",
        "expenses": [
            {
                "category": "meals",
                "amount": "400",
                "currency": "INR",
                "description": "Lunch",
                "receipt_attached": True,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _post_approve(payload: dict, mock_gemini) -> int:
    mock_gemini.return_value = gemini_response_payload(
        "manual_review",
        approved_amount="0",
        rejected_amount="0",
        explanation="Edge case processed safely.",
    )
    response = client.post("/approve", json=payload)
    return response.status_code


@pytest.fixture
def mock_gemini():
    with patch("app.llm.decision_engine.generate_json_response") as mock:
        yield mock


# --- API / model validation edge cases ---


def test_no_expenses_returns_422(mock_gemini) -> None:
    payload = _base_claim_payload(claim_id="EDGE-NO-EXP", expenses=[])
    assert _post_approve(payload, mock_gemini) == 422


def test_hundred_expenses_does_not_crash(mock_gemini) -> None:
    expenses = [
        {
            "category": "meals",
            "amount": "100",
            "currency": "INR",
            "description": f"Meal item {index}",
            "receipt_attached": True,
        }
        for index in range(100)
    ]
    payload = _base_claim_payload(claim_id="EDGE-100", expenses=expenses)
    status = _post_approve(payload, mock_gemini)
    assert status == 200


def test_max_length_description_accepted(mock_gemini) -> None:
    description = "A" * 500
    payload = _base_claim_payload(
        claim_id="EDGE-LONG-DESC",
        expenses=[
            {
                "category": "meals",
                "amount": "400",
                "currency": "INR",
                "description": description,
                "receipt_attached": True,
            }
        ],
    )
    assert _post_approve(payload, mock_gemini) == 200


def test_overlong_description_returns_422(mock_gemini) -> None:
    payload = _base_claim_payload(
        claim_id="EDGE-OVER-DESC",
        expenses=[
            {
                "category": "meals",
                "amount": "400",
                "currency": "INR",
                "description": "X" * 501,
                "receipt_attached": True,
            }
        ],
    )
    assert _post_approve(payload, mock_gemini) == 422


def test_unicode_description_does_not_crash(mock_gemini) -> None:
    payload = _base_claim_payload(
        claim_id="EDGE-UNICODE",
        destination="北京, 中国",
        purpose="Réunion client à Zürich — café expenses",
        expenses=[
            {
                "category": "meals",
                "amount": "450",
                "currency": "INR",
                "description": "Обед с клиентом naïve café 北京",
                "receipt_attached": True,
            }
        ],
    )
    assert _post_approve(payload, mock_gemini) == 200


def test_emoji_description_does_not_crash(mock_gemini) -> None:
    payload = _base_claim_payload(
        claim_id="EDGE-EMOJI",
        expenses=[
            {
                "category": "transport",
                "amount": "350",
                "currency": "INR",
                "description": "🚕 Uber taxi to client 🏢",
                "receipt_attached": True,
            }
        ],
    )
    assert _post_approve(payload, mock_gemini) == 200


def test_huge_amount_does_not_crash(mock_gemini) -> None:
    payload = _base_claim_payload(
        claim_id="EDGE-HUGE",
        expenses=[
            {
                "category": "hotel",
                "amount": "999999999999.99",
                "currency": "INR",
                "description": "Ultra luxury suite",
                "receipt_attached": True,
            }
        ],
    )
    status = _post_approve(payload, mock_gemini)
    assert status == 200


def test_zero_amount_returns_422(mock_gemini) -> None:
    payload = _base_claim_payload(
        claim_id="EDGE-ZERO",
        expenses=[
            {
                "category": "meals",
                "amount": "0",
                "currency": "INR",
                "description": "Free sample",
                "receipt_attached": True,
            }
        ],
    )
    assert _post_approve(payload, mock_gemini) == 422


def test_negative_amount_returns_422(mock_gemini) -> None:
    payload = _base_claim_payload(
        claim_id="EDGE-NEG",
        expenses=[
            {
                "category": "meals",
                "amount": "-100",
                "currency": "INR",
                "description": "Refund attempt",
                "receipt_attached": True,
            }
        ],
    )
    assert _post_approve(payload, mock_gemini) == 422


def test_duplicate_receipts_same_line_does_not_crash(mock_gemini) -> None:
    expense = {
        "category": "meals",
        "amount": "500",
        "currency": "INR",
        "description": "Duplicate lunch line",
        "receipt_attached": True,
    }
    payload = _base_claim_payload(
        claim_id="EDGE-DUP-RECEIPT",
        expenses=[expense, expense],
    )
    status = _post_approve(payload, mock_gemini)
    assert status == 200


def test_empty_string_fields_return_422(mock_gemini) -> None:
    payload = _base_claim_payload(claim_id="EDGE-EMPTY", employee_name="")
    assert _post_approve(payload, mock_gemini) == 422


def test_null_values_return_422(mock_gemini) -> None:
    payload = _base_claim_payload(claim_id="EDGE-NULL")
    payload["employee_name"] = None
    assert _post_approve(payload, mock_gemini) == 422


def test_unexpected_category_returns_422(mock_gemini) -> None:
    payload = _base_claim_payload(
        claim_id="EDGE-BAD-CAT",
        expenses=[
            {
                "category": "spaceship",
                "amount": "1000",
                "currency": "INR",
                "description": "Rocket travel",
                "receipt_attached": True,
            }
        ],
    )
    assert _post_approve(payload, mock_gemini) == 422


# --- Tool-level stability ---


def test_extremely_large_policy_query_does_not_crash() -> None:
    query = "hotel meals taxi flight " * 5000
    result = lookup_policy(query)

    assert isinstance(result, dict)
    assert "policy_text" in result
    assert "section_titles" in result
    assert isinstance(result["section_titles"], list)


def test_validate_expenses_hundred_items() -> None:
    expenses = [
        make_expense(ExpenseCategory.MEALS, "50", description=f"Snack {i}")
        for i in range(100)
    ]
    result = validate_expenses(expenses)

    assert result["approved_amount"] == Decimal("5000")
    assert isinstance(result["violations"], list)


def test_receipt_checker_hundred_missing() -> None:
    expenses = [
        make_expense(
            ExpenseCategory.MEALS,
            "600",
            description=f"No receipt {i}",
            receipt_attached=False,
        )
        for i in range(100)
    ]
    result = check_receipts(expenses)

    assert result["is_valid"] is False
    assert len(result["missing_receipts"]) == 100


def test_duplicate_checker_many_identical_lines() -> None:
    expense = make_expense(ExpenseCategory.MEALS, "100", description="Same item")
    claim = make_claim(
        "EDGE-DUP-MANY",
        [expense] * 50,
    )
    result = check_duplicates(claim)

    assert result["has_duplicates"] is True
    assert len(result["duplicates"]) == 49


# --- Graph pipeline stability ---


def test_graph_invoke_unicode_claim_end_to_end(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "approved",
        approved_amount="450",
        rejected_amount="0",
        explanation="Unicode claim approved.",
    )
    claim = make_claim(
        "EDGE-GRAPH-UNI",
        [
            make_expense(
                ExpenseCategory.MEALS,
                "450",
                description="東京ラーメン 🍜",
            )
        ],
        destination="東京",
        purpose="技術会議",
    )
    result = travel_reimbursement_graph.invoke(
        {
            "claim": claim,
            "duplicate_result": None,
            "validation_result": None,
            "receipt_result": None,
            "policy_context": None,
            "policy_section_titles": [],
            "ai_decision": None,
            "final_decision": None,
            "current_step": "start",
            "errors": [],
        }
    )

    assert result["final_decision"] is not None
    assert result["current_step"] == "decision_complete"


def test_graph_gemini_failure_still_returns_decision(mock_gemini) -> None:
    mock_gemini.side_effect = RuntimeError("Gemini down")
    claim = make_claim(
        "EDGE-FALLBACK",
        [make_expense(ExpenseCategory.MEALS, "400", description="Lunch")],
    )
    result = travel_reimbursement_graph.invoke(
        {
            "claim": claim,
            "duplicate_result": None,
            "validation_result": None,
            "receipt_result": None,
            "policy_context": None,
            "policy_section_titles": [],
            "ai_decision": None,
            "final_decision": None,
            "current_step": "start",
            "errors": [],
        }
    )

    assert result["final_decision"] is not None
    assert result["errors"]


def test_malformed_json_body_returns_422() -> None:
    response = client.post(
        "/approve",
        content=b"{not-valid-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_missing_required_top_level_fields_returns_422(mock_gemini) -> None:
    response = client.post("/approve", json={"claim_id": "EDGE-MISSING"})
    assert response.status_code == 422
