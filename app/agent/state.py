"""LangGraph state for a single travel reimbursement approval request."""

from typing import TypedDict

from app.models import ApprovalDecision, TravelClaim


class AgentState(TypedDict):
    """Complete lifecycle state for one reimbursement approval run.

    Fields:
        claim: Submitted travel reimbursement claim under review.
        duplicate_result: Duplicate detection output for the claim.
        validation_result: Expense validation output (limits, categories, amounts).
        receipt_result: Receipt check output (missing documentation flags).
        policy_context: Combined text of retrieved policy sections.
        policy_section_titles: Titles of retrieved policy sections for citation.
        ai_decision: Gemini-generated decision before finalization.
        final_decision: Structured approval outcome when the workflow completes.
        current_step: Active or most recently completed graph node name.
        errors: Non-fatal issues collected during processing.
    """

    claim: TravelClaim
    duplicate_result: dict | None
    validation_result: dict | None
    receipt_result: dict | None
    policy_context: str | None
    policy_section_titles: list[str]
    ai_decision: ApprovalDecision | None
    final_decision: ApprovalDecision | None
    current_step: str
    errors: list[str]
