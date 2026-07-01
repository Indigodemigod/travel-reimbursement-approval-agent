"""LangGraph workflow for travel reimbursement approval."""

import logging
import time
from collections.abc import Callable
from decimal import Decimal

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.llm.decision_engine import generate_ai_decision
from app.models import ApprovalDecision, ApprovalStatus
from app.tools.duplicate_checker import check_duplicates as run_duplicate_check
from app.tools.expense_validator import validate_expenses as run_expense_validation
from app.tools.policy_lookup import lookup_policy
from app.tools.receipt_checker import check_receipts as run_receipt_check

logger = logging.getLogger(__name__)


def _with_node_logging(
    node_name: str,
    handler: Callable[[AgentState], AgentState],
) -> Callable[[AgentState], AgentState]:
    """Wrap a graph node with enter/leave timing logs."""

    def wrapped(state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info("Entering node: %s", node_name)
        try:
            updated_state = handler(state)
            elapsed = time.perf_counter() - start
            logger.info("Leaving node: %s (elapsed: %.3fs)", node_name, elapsed)
            return updated_state
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "Leaving node: %s with error (elapsed: %.3fs)",
                node_name,
                elapsed,
            )
            raise

    return wrapped


def check_duplicates(state: AgentState) -> AgentState:
    """Detect duplicate line items or repeated claim submissions."""
    claim = state["claim"]
    state["duplicate_result"] = run_duplicate_check(claim)
    state["current_step"] = "duplicate_check"
    return state


def validate_expenses(state: AgentState) -> AgentState:
    """Validate claim expenses against policy limits and category rules."""
    claim = state["claim"]
    state["validation_result"] = run_expense_validation(claim.expenses)
    state["current_step"] = "expense_validation"
    return state


def check_receipts(state: AgentState) -> AgentState:
    """Check that required receipts are attached for high-value items."""
    claim = state["claim"]
    state["receipt_result"] = run_receipt_check(claim.expenses)
    state["current_step"] = "receipt_validation"
    return state


def retrieve_policy(state: AgentState) -> AgentState:
    """Retrieve relevant travel policy sections for the claim."""
    claim = state["claim"]

    query_parts = [claim.purpose, claim.destination]
    for expense in claim.expenses:
        query_parts.append(expense.category.value)
        query_parts.append(expense.description)

    query = " ".join(part.strip() for part in query_parts if part.strip())
    policy_result = lookup_policy(query)
    state["policy_context"] = policy_result["policy_text"]
    state["policy_section_titles"] = policy_result["section_titles"]
    state["current_step"] = "policy_lookup"
    return state


def ai_decision(state: AgentState) -> AgentState:
    """Run Gemini reasoning over deterministic tool outputs."""
    validation_result = state["validation_result"] or {}
    receipt_result = state["receipt_result"] or {}
    duplicate_result = state["duplicate_result"] or {}
    policy_context = state["policy_context"] or ""
    policy_section_titles = state.get("policy_section_titles") or []
    claim = state["claim"]

    try:
        state["ai_decision"] = generate_ai_decision(
            claim=claim,
            validation_result=validation_result,
            receipt_result=receipt_result,
            duplicate_result=duplicate_result,
            policy_context=policy_context,
            policy_section_titles=policy_section_titles,
        )
    except Exception as exc:
        fallback_reason = f"Gemini decision failed in ai_decision node: {exc}"
        logger.warning(fallback_reason, exc_info=True)
        state["errors"] = [*state.get("errors", []), fallback_reason]
        state["ai_decision"] = None

    state["current_step"] = "ai_decision"
    return state


def _build_rule_based_decision(
    validation_result: dict,
    receipt_result: dict,
    policy_context: str | None,
    duplicate_result: dict | None = None,
) -> ApprovalDecision:
    """Deterministic fallback decision when Gemini is unavailable."""
    if duplicate_result and duplicate_result.get("has_duplicates"):
        decision = ApprovalStatus.MANUAL_REVIEW
    elif validation_result.get("manual_review"):
        decision = ApprovalStatus.MANUAL_REVIEW
    elif not receipt_result.get("is_valid", True):
        decision = ApprovalStatus.MANUAL_REVIEW
    elif not validation_result.get("is_valid", True):
        decision = ApprovalStatus.REJECTED
    else:
        decision = ApprovalStatus.APPROVED

    confidence_by_decision = {
        ApprovalStatus.APPROVED: 0.97,
        ApprovalStatus.REJECTED: 0.95,
        ApprovalStatus.MANUAL_REVIEW: 0.80,
    }

    missing_receipts = receipt_result.get("missing_receipts", [])
    missing_documents = [
        (
            f"{item['description']} ({item['category']}): "
            f"{item['currency']} {item['amount']} — receipt required"
        )
        for item in missing_receipts
    ]

    violations = validation_result.get("violations", [])

    if decision == ApprovalStatus.APPROVED:
        explanation = (
            "Claim approved. Expenses comply with travel policy and all "
            "required receipts are attached."
        )
    elif decision == ApprovalStatus.REJECTED:
        policy_note = (
            " Relevant policy sections were reviewed."
            if policy_context
            else ""
        )
        explanation = (
            f"Claim rejected due to {len(violations)} policy violation(s): "
            + "; ".join(violations)
            + policy_note
        )
    elif duplicate_result and duplicate_result.get("has_duplicates"):
        explanation = (
            "Claim requires manual review due to duplicate expense lines "
            "or a repeated submission."
        )
    elif validation_result.get("manual_review"):
        explanation = (
            "Claim requires manual review due to expenses flagged for "
            "foreign currency, unsupported category, or unrecognized transport."
        )
    else:
        explanation = (
            f"Claim requires manual review due to {len(missing_documents)} "
            "missing receipt(s) for expenses above the documentation threshold."
        )

    return ApprovalDecision(
        decision=decision,
        approved_amount=validation_result.get("approved_amount", Decimal("0")),
        rejected_amount=validation_result.get("rejected_amount", Decimal("0")),
        missing_documents=missing_documents,
        violated_policies=violations,
        confidence=confidence_by_decision[decision],
        explanation=explanation,
    )


def finalize_decision(state: AgentState) -> AgentState:
    """Select the AI decision or fall back to deterministic rules."""
    if state.get("ai_decision") is not None:
        state["final_decision"] = state["ai_decision"]
    else:
        logger.warning(
            "No ai_decision available; using rule-based fallback in finalize_decision"
        )
        state["final_decision"] = _build_rule_based_decision(
            validation_result=state["validation_result"] or {},
            receipt_result=state["receipt_result"] or {},
            policy_context=state.get("policy_context"),
            duplicate_result=state.get("duplicate_result"),
        )

    state["current_step"] = "decision_complete"
    return state


def build_graph() -> StateGraph:
    """Construct the reimbursement approval StateGraph (uncompiled)."""
    workflow = StateGraph(AgentState)

    node_handlers = {
        "check_duplicates": check_duplicates,
        "validate_expenses": validate_expenses,
        "check_receipts": check_receipts,
        "retrieve_policy": retrieve_policy,
        "ai_decision": ai_decision,
        "finalize_decision": finalize_decision,
    }

    for node_name, handler in node_handlers.items():
        workflow.add_node(node_name, _with_node_logging(node_name, handler))

    workflow.add_edge(START, "check_duplicates")
    workflow.add_edge("check_duplicates", "validate_expenses")
    workflow.add_edge("validate_expenses", "check_receipts")
    workflow.add_edge("check_receipts", "retrieve_policy")
    workflow.add_edge("retrieve_policy", "ai_decision")
    workflow.add_edge("ai_decision", "finalize_decision")
    workflow.add_edge("finalize_decision", END)

    return workflow


travel_reimbursement_graph = build_graph().compile()
