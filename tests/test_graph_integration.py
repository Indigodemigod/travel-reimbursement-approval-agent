"""LangGraph workflow integration tests."""

from decimal import Decimal
from unittest.mock import patch

import pytest

from app.agent.graph import travel_reimbursement_graph
from app.models import ApprovalStatus, ExpenseCategory
from tests.conftest import gemini_response_payload, make_claim, make_expense


EXPECTED_NODE_ORDER = [
    "check_duplicates",
    "validate_expenses",
    "check_receipts",
    "retrieve_policy",
    "ai_decision",
    "finalize_decision",
]

EXPECTED_CURRENT_STEPS = [
    "duplicate_check",
    "expense_validation",
    "receipt_validation",
    "policy_lookup",
    "ai_decision",
    "decision_complete",
]


def _initial_state(claim):
    return {
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


def _stream_node_snapshots(initial_state: dict) -> list[tuple[str, dict]]:
    snapshots: list[tuple[str, dict]] = []
    for event in travel_reimbursement_graph.stream(initial_state):
        node_name = next(iter(event))
        snapshots.append((node_name, event[node_name]))
    return snapshots


def _approved_claim():
    return make_claim(
        "INT-APPROVED-1",
        [
            make_expense(ExpenseCategory.MEALS, "400", description="Lunch"),
            make_expense(ExpenseCategory.TRANSPORT, "300", description="Uber taxi"),
        ],
        purpose="Client visit",
        destination="Delhi",
    )


@patch("app.llm.decision_engine.generate_json_response")
def test_workflow_node_execution_order(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "approved",
        approved_amount="700",
        rejected_amount="0",
        explanation="Approved.",
    )
    claim = _approved_claim()
    snapshots = _stream_node_snapshots(_initial_state(claim))
    executed_nodes = [name for name, _ in snapshots]

    assert executed_nodes == EXPECTED_NODE_ORDER


@patch("app.llm.decision_engine.generate_json_response")
def test_workflow_current_step_after_each_node(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "approved",
        approved_amount="700",
        rejected_amount="0",
        explanation="Approved.",
    )
    claim = _approved_claim()
    snapshots = _stream_node_snapshots(_initial_state(claim))
    current_steps = [state["current_step"] for _, state in snapshots]

    assert current_steps == EXPECTED_CURRENT_STEPS


@patch("app.llm.decision_engine.generate_json_response")
def test_agent_state_after_every_node_gemini_success(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "approved",
        approved_amount="700",
        rejected_amount="0",
        confidence=0.97,
        explanation="Approved per policy.",
    )
    claim = _approved_claim()
    snapshots = _stream_node_snapshots(_initial_state(claim))

    by_node = dict(snapshots)

    dup_state = by_node["check_duplicates"]
    assert dup_state["duplicate_result"] is not None
    assert dup_state["duplicate_result"]["has_duplicates"] is False
    assert dup_state["validation_result"] is None
    assert dup_state["ai_decision"] is None

    val_state = by_node["validate_expenses"]
    assert val_state["validation_result"] is not None
    assert val_state["validation_result"]["is_valid"] is True
    assert val_state["validation_result"]["approved_amount"] == Decimal("700")

    receipt_state = by_node["check_receipts"]
    assert receipt_state["receipt_result"] is not None
    assert receipt_state["receipt_result"]["is_valid"] is True

    policy_state = by_node["retrieve_policy"]
    assert policy_state["policy_context"]
    assert len(policy_state["policy_section_titles"]) >= 1

    ai_state = by_node["ai_decision"]
    assert ai_state["ai_decision"] is not None
    assert ai_state["ai_decision"].decision == ApprovalStatus.APPROVED
    assert ai_state["errors"] == []

    final_state = by_node["finalize_decision"]
    assert final_state["final_decision"] is not None
    assert final_state["final_decision"].decision == ApprovalStatus.APPROVED
    assert final_state["final_decision"] is ai_state["ai_decision"]
    assert final_state["current_step"] == "decision_complete"


@patch(
    "app.llm.decision_engine.generate_json_response",
    side_effect=RuntimeError("Gemini unavailable"),
)
def test_workflow_gemini_fallback_path(mock_gemini) -> None:
    claim = _approved_claim()
    result = travel_reimbursement_graph.invoke(_initial_state(claim))

    assert result["ai_decision"] is None
    assert len(result["errors"]) == 1
    assert "Gemini decision failed" in result["errors"][0]
    assert result["final_decision"] is not None
    assert result["final_decision"].decision == ApprovalStatus.APPROVED
    assert result["current_step"] == "decision_complete"
    mock_gemini.assert_called_once()


@patch("app.llm.decision_engine.generate_json_response")
def test_workflow_gemini_success_path_end_to_end(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "rejected",
        approved_amount="0",
        rejected_amount="6000",
        violated_policies=["Hotel exceeds limit"],
        confidence=0.95,
        explanation="Rejected due to ## 4. Hotel Accommodation.",
    )
    claim = make_claim(
        "INT-REJECT-1",
        [make_expense(ExpenseCategory.HOTEL, "6000", description="Luxury hotel")],
    )
    result = travel_reimbursement_graph.invoke(_initial_state(claim))

    assert result["ai_decision"] is not None
    assert result["ai_decision"].decision == ApprovalStatus.REJECTED
    assert result["final_decision"] is result["ai_decision"]
    assert result["validation_result"]["is_valid"] is False
    assert result["errors"] == []
    mock_gemini.assert_called_once()


@patch("app.llm.decision_engine.generate_json_response")
def test_workflow_invoke_reaches_end(mock_gemini) -> None:
    mock_gemini.return_value = gemini_response_payload(
        "approved",
        approved_amount="400",
        rejected_amount="0",
        explanation="Approved.",
    )
    claim = make_claim(
        "INT-END-1",
        [make_expense(ExpenseCategory.MEALS, "400", description="Lunch")],
    )
    snapshots = _stream_node_snapshots(_initial_state(claim))

    assert snapshots[-1][0] == "finalize_decision"
    assert snapshots[-1][1]["current_step"] == "decision_complete"
    assert snapshots[-1][1]["final_decision"] is not None
