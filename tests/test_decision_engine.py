"""Tests for AI decision engine."""

import json
from decimal import Decimal
from unittest.mock import patch

import pytest

from app.llm.decision_engine import (
    _parse_json_response,
    generate_ai_decision,
    parse_ai_decision_json,
)
from app.models import ApprovalStatus, ExpenseCategory
from tests.conftest import gemini_response_payload, make_claim, make_expense


@pytest.fixture
def sample_claim():
    return make_claim(
        "AI-TEST-1",
        [make_expense(ExpenseCategory.MEALS, "400", description="Lunch")],
    )


@pytest.fixture
def sample_tool_outputs():
    return {
        "validation_result": {
            "is_valid": True,
            "approved_amount": Decimal("400"),
            "rejected_amount": Decimal("0"),
            "manual_review": False,
            "violations": [],
        },
        "receipt_result": {"is_valid": True, "missing_receipts": []},
        "duplicate_result": {
            "has_duplicates": False,
            "is_valid": True,
            "duplicates": [],
        },
        "policy_context": "## 5. Meals & Daily Allowance",
        "policy_section_titles": ["## 5. Meals & Daily Allowance"],
    }


def _call_generate_ai_decision(claim, outputs, gemini_payload: str):
    with patch(
        "app.llm.decision_engine.generate_json_response",
        return_value=gemini_payload,
    ):
        return generate_ai_decision(
            claim=claim,
            validation_result=outputs["validation_result"],
            receipt_result=outputs["receipt_result"],
            duplicate_result=outputs["duplicate_result"],
            policy_context=outputs["policy_context"],
            policy_section_titles=outputs["policy_section_titles"],
        )


def test_generate_ai_decision_approved(sample_claim, sample_tool_outputs) -> None:
    payload = gemini_response_payload(
        "approved",
        approved_amount="400",
        rejected_amount="0",
        explanation="Approved per policy.",
    )
    decision = _call_generate_ai_decision(sample_claim, sample_tool_outputs, payload)

    assert decision.decision == ApprovalStatus.APPROVED
    assert decision.approved_amount == Decimal("400")
    assert decision.rejected_amount == Decimal("0")


def test_generate_ai_decision_rejected(sample_claim, sample_tool_outputs) -> None:
    payload = gemini_response_payload(
        "rejected",
        approved_amount="0",
        rejected_amount="400",
        violated_policies=["Meals limit exceeded"],
        explanation="Rejected due to policy violation.",
    )
    decision = _call_generate_ai_decision(sample_claim, sample_tool_outputs, payload)

    assert decision.decision == ApprovalStatus.REJECTED
    assert decision.rejected_amount == Decimal("400")
    assert decision.violated_policies == ["Meals limit exceeded"]


def test_generate_ai_decision_manual_review(sample_claim, sample_tool_outputs) -> None:
    payload = gemini_response_payload(
        "manual_review",
        approved_amount="0",
        rejected_amount="0",
        confidence=0.8,
        explanation="Needs human review.",
    )
    decision = _call_generate_ai_decision(sample_claim, sample_tool_outputs, payload)

    assert decision.decision == ApprovalStatus.MANUAL_REVIEW
    assert decision.confidence == 0.8


def test_generate_ai_decision_partially_approved(
    sample_claim, sample_tool_outputs
) -> None:
    payload = gemini_response_payload(
        "partially_approved",
        approved_amount="300",
        rejected_amount="100",
        explanation="Partial approval applied.",
    )
    decision = _call_generate_ai_decision(sample_claim, sample_tool_outputs, payload)

    assert decision.decision == ApprovalStatus.PARTIALLY_APPROVED
    assert decision.approved_amount == Decimal("300")
    assert decision.rejected_amount == Decimal("100")


def test_generate_ai_decision_fallback_path_propagates_error(
    sample_claim, sample_tool_outputs
) -> None:
    with patch(
        "app.llm.decision_engine.generate_json_response",
        side_effect=RuntimeError("Gemini unavailable"),
    ):
        with pytest.raises(RuntimeError, match="Gemini unavailable"):
            generate_ai_decision(
                claim=sample_claim,
                validation_result=sample_tool_outputs["validation_result"],
                receipt_result=sample_tool_outputs["receipt_result"],
                duplicate_result=sample_tool_outputs["duplicate_result"],
                policy_context=sample_tool_outputs["policy_context"],
                policy_section_titles=sample_tool_outputs["policy_section_titles"],
            )


def test_parse_invalid_gemini_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_json_response("not-json")


def test_parse_ai_decision_missing_fields(sample_tool_outputs) -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        parse_ai_decision_json(
            {"decision": "approved"},
            sample_tool_outputs["validation_result"],
            sample_tool_outputs["receipt_result"],
        )


def test_parse_ai_decision_invalid_confidence(sample_tool_outputs) -> None:
    payload = json.loads(gemini_response_payload("approved", confidence=1.5))
    with pytest.raises(ValueError, match="Confidence out of range"):
        parse_ai_decision_json(
            payload,
            sample_tool_outputs["validation_result"],
            sample_tool_outputs["receipt_result"],
        )


def test_parse_json_response_with_markdown_fence() -> None:
    parsed = _parse_json_response(
        '```json\n{"decision": "approved", "value": 1}\n```'
    )

    assert parsed["decision"] == "approved"
    assert parsed["value"] == 1
