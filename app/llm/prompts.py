"""Prompt templates for travel reimbursement decision reasoning."""

import json
from typing import Any

from app.models import TravelClaim

DECISION_RULES = """
DECISION RULES (apply in order of priority):
1. Use ONLY the policy context provided below. Do NOT invent policy rules.
2. Treat expense validation, receipt validation, and duplicate detection outputs
   as authoritative facts.
3. Do NOT recalculate expense limits yourself. Use approved_amount and rejected_amount
   from the validation result unless decision is partially_approved.
4. If duplicate_result.has_duplicates is true, prefer decision = manual_review.
5. If validation_result.manual_review is true, prefer decision = manual_review.
6. If receipt_result.is_valid is false, prefer decision = manual_review unless
   policy clearly supports full rejection with high confidence.
7. If validation_result.is_valid is false with clear violations, prefer rejected.
8. If all checks pass and policy supports reimbursement, prefer approved.
9. For ambiguous, conflicting, or incomplete information, choose manual_review.
10. missing_documents must be derived from receipt_result.missing_receipts only.
11. violated_policies must be derived from validation_result.violations only.
12. Never hallucinate amounts, policies, or documents not present in the inputs.
13. In the explanation, cite policy sections ONLY using titles from
    RETRIEVED POLICY SECTION TITLES. Do NOT invent section numbers or names.

ALLOWED decision values:
- approved
- rejected
- partially_approved
- manual_review
"""

RESPONSE_SCHEMA = """
Return ONLY valid JSON with this exact structure (no markdown, no extra keys):
{
  "decision": "approved | rejected | partially_approved | manual_review",
  "approved_amount": <number>,
  "rejected_amount": <number>,
  "missing_documents": [<string>],
  "violated_policies": [<string>],
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<concise human-readable rationale referencing policy section titles when applicable>"
}
"""


def _serialize_for_prompt(data: Any) -> str:
    """Serialize tool outputs for safe inclusion in the prompt."""

    def default_encoder(value: Any) -> Any:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    return json.dumps(data, indent=2, default=default_encoder)


def build_decision_prompt(
    claim: TravelClaim,
    validation_result: dict[str, Any],
    receipt_result: dict[str, Any],
    duplicate_result: dict[str, Any],
    policy_context: str,
    policy_section_titles: list[str],
) -> str:
    """Build the Gemini decision prompt from claim data and tool outputs."""
    claim_json = claim.model_dump(mode="json")
    section_titles_text = (
        _serialize_for_prompt(policy_section_titles)
        if policy_section_titles
        else "[]"
    )

    return f"""You are a Travel Reimbursement Approval Agent for an enterprise HR system.

Your task is to review a travel reimbursement claim and produce a structured approval decision.

CRITICAL INSTRUCTIONS:
- Base your reasoning ONLY on the claim details, tool validation results, and policy context below.
- Do NOT hallucinate policies, amounts, receipts, violations, or section names.
- If information is insufficient or ambiguous, return decision = manual_review.
- Use manual_review instead of forcing approved or rejected when uncertain.
- Keep the explanation concise, factual, and reference specific validation findings.
- When citing policy, use ONLY titles from RETRIEVED POLICY SECTION TITLES.
  Example: "Rejected because Hotel expense exceeds ## 4. Hotel Accommodation."
- Never cite a section title that is not listed below.

{DECISION_RULES}

--- CLAIM DETAILS ---
{_serialize_for_prompt(claim_json)}

--- DUPLICATE DETECTION RESULT (deterministic tool output) ---
{_serialize_for_prompt(duplicate_result)}

--- EXPENSE VALIDATION RESULT (deterministic tool output) ---
{_serialize_for_prompt(validation_result)}

--- RECEIPT VALIDATION RESULT (deterministic tool output) ---
{_serialize_for_prompt(receipt_result)}

--- RETRIEVED POLICY SECTION TITLES (only cite these in explanations) ---
{section_titles_text}

--- RETRIEVED POLICY TEXT (only authoritative policy content) ---
{policy_context or "No relevant policy section was retrieved."}

{RESPONSE_SCHEMA}
"""
