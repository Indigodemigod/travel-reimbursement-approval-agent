"""API integration tests using FastAPI TestClient."""

from decimal import Decimal
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import ApprovalStatus, ExpenseCategory
from tests.conftest import gemini_response_payload

client = TestClient(app)


def _claim_payload(
    claim_id: str,
    expenses: list[dict],
    *,
    employee_id: str = "EMP001",
) -> dict:
    return {
        "claim_id": claim_id,
        "employee_id": employee_id,
        "employee_name": "Jane Doe",
        "department": "Engineering",
        "trip_start_date": "2026-03-01",
        "trip_end_date": "2026-03-05",
        "destination": "Delhi",
        "purpose": "Client visit",
        "expenses": expenses,
    }


def _expense_payload(
    category: str,
    amount: str,
    *,
    currency: str = "INR",
    description: str = "Business expense",
    receipt_attached: bool = True,
) -> dict:
    return {
        "category": category,
        "amount": amount,
        "currency": currency,
        "description": description,
        "receipt_attached": receipt_attached,
    }


@pytest.fixture
def mock_gemini():
    with patch("app.llm.decision_engine.generate_json_response") as mock:
        yield mock


def test_get_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "Travel Reimbursement Approval Agent"
    assert body["version"] == "1.0.0"


def test_get_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "Travel Reimbursement Approval Agent"


def test_post_approve_approved_claim(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "approved",
        approved_amount="400",
        rejected_amount="0",
        explanation="Approved.",
    )
    payload = _claim_payload(
        "API-APPROVED-1",
        [_expense_payload("meals", "400", description="Lunch")],
    )

    response = client.post("/approve", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == ApprovalStatus.APPROVED.value
    assert Decimal(body["approved_amount"]) == Decimal("400")
    mock_gemini.assert_called_once()


def test_post_approve_rejected_claim(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "rejected",
        approved_amount="0",
        rejected_amount="6000",
        violated_policies=["Hotel limit exceeded"],
        explanation="Rejected due to hotel policy.",
    )
    payload = _claim_payload(
        "API-REJECTED-1",
        [_expense_payload("hotel", "6000", description="Luxury hotel")],
    )

    response = client.post("/approve", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == ApprovalStatus.REJECTED.value
    assert Decimal(body["rejected_amount"]) == Decimal("6000")


def test_post_approve_manual_review(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "manual_review",
        approved_amount="0",
        rejected_amount="0",
        confidence=0.8,
        explanation="Manual review required.",
    )
    payload = _claim_payload(
        "API-MANUAL-1",
        [
            _expense_payload(
                "hotel",
                "200",
                currency="USD",
                description="US hotel",
            )
        ],
    )

    response = client.post("/approve", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == ApprovalStatus.MANUAL_REVIEW.value


def test_post_approve_duplicate_claim(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "manual_review",
        approved_amount="0",
        rejected_amount="0",
        confidence=0.8,
        explanation="Duplicate detected.",
    )
    payload = _claim_payload(
        "API-DUP-1",
        [_expense_payload("meals", "400", description="Lunch")],
    )

    first = client.post("/approve", json=payload)
    second = client.post("/approve", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["decision"] == ApprovalStatus.MANUAL_REVIEW.value


def test_post_approve_missing_receipts(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "manual_review",
        approved_amount="0",
        rejected_amount="0",
        missing_documents=["Dinner receipt missing"],
        confidence=0.82,
        explanation="Missing receipt documentation.",
    )
    payload = _claim_payload(
        "API-RECEIPT-1",
        [
            _expense_payload(
                "meals",
                "600",
                description="Dinner",
                receipt_attached=False,
            )
        ],
    )

    response = client.post("/approve", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == ApprovalStatus.MANUAL_REVIEW.value
    assert body["missing_documents"]


def test_post_approve_invalid_payload() -> None:
    payload = _claim_payload("API-INVALID-1", [])
    payload["trip_end_date"] = "2026-02-01"

    response = client.post("/approve", json=payload)

    assert response.status_code == 422


def test_post_approve_gemini_fallback(mock_gemini) -> None:
    mock_gemini.side_effect = RuntimeError("Gemini unavailable")
    payload = _claim_payload(
        "API-FALLBACK-1",
        [_expense_payload("meals", "400", description="Lunch")],
    )

    response = client.post("/approve", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {
        ApprovalStatus.APPROVED.value,
        ApprovalStatus.MANUAL_REVIEW.value,
        ApprovalStatus.REJECTED.value,
    }
    assert body["explanation"]
