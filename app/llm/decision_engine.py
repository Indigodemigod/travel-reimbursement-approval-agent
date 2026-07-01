"""AI-assisted decision engine using Gemini over deterministic tool outputs."""

import json
import logging
import re
from decimal import Decimal
from typing import Any

from app.llm.gemini_client import generate_json_response
from app.llm.prompts import build_decision_prompt
from app.models import ApprovalDecision, ApprovalStatus, TravelClaim

logger = logging.getLogger(__name__)

_VALID_DECISIONS = {
    "approved": ApprovalStatus.APPROVED,
    "rejected": ApprovalStatus.REJECTED,
    "partially_approved": ApprovalStatus.PARTIALLY_APPROVED,
    "manual_review": ApprovalStatus.MANUAL_REVIEW,
    "needs_review": ApprovalStatus.MANUAL_REVIEW,
}

_REQUIRED_FIELDS = (
    "decision",
    "approved_amount",
    "rejected_amount",
    "missing_documents",
    "violated_policies",
    "confidence",
    "explanation",
)


def _parse_json_response(raw_response: str) -> dict[str, Any]:
    """Parse Gemini JSON, tolerating optional markdown fences."""
    text = raw_response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response is not a JSON object")
    return parsed


def _to_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _normalize_missing_documents(
    ai_documents: list[Any],
    receipt_result: dict[str, Any],
) -> list[str]:
    if ai_documents:
        return [str(item) for item in ai_documents]

    return [
        (
            f"{item['description']} ({item['category']}): "
            f"{item['currency']} {item['amount']} — receipt required"
        )
        for item in receipt_result.get("missing_receipts", [])
    ]


def _normalize_violated_policies(
    ai_violations: list[Any],
    validation_result: dict[str, Any],
) -> list[str]:
    if ai_violations:
        return [str(item) for item in ai_violations]
    return [str(item) for item in validation_result.get("violations", [])]


def parse_ai_decision_json(
    parsed: dict[str, Any],
    validation_result: dict[str, Any],
    receipt_result: dict[str, Any],
) -> ApprovalDecision:
    """Convert validated Gemini JSON into an ApprovalDecision model."""
    missing = {field for field in _REQUIRED_FIELDS if field not in parsed}
    if missing:
        raise ValueError(f"Gemini JSON missing required fields: {sorted(missing)}")

    decision_key = str(parsed["decision"]).strip().lower()
    if decision_key not in _VALID_DECISIONS:
        raise ValueError(f"Invalid decision value from Gemini: {parsed['decision']}")

    explanation = str(parsed["explanation"]).strip()
    if not explanation:
        raise ValueError("Gemini returned an empty explanation")

    confidence = float(parsed["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence out of range: {confidence}")

    return ApprovalDecision(
        decision=_VALID_DECISIONS[decision_key],
        approved_amount=_to_decimal(parsed["approved_amount"]),
        rejected_amount=_to_decimal(parsed["rejected_amount"]),
        missing_documents=_normalize_missing_documents(
            parsed.get("missing_documents", []),
            receipt_result,
        ),
        violated_policies=_normalize_violated_policies(
            parsed.get("violated_policies", []),
            validation_result,
        ),
        confidence=confidence,
        explanation=explanation,
    )


def generate_ai_decision(
    claim: TravelClaim,
    validation_result: dict[str, Any],
    receipt_result: dict[str, Any],
    duplicate_result: dict[str, Any],
    policy_context: str,
    policy_section_titles: list[str],
) -> ApprovalDecision:
    """Build prompt, call Gemini 2.5 Flash, and return a structured decision."""
    logger.info("Validation result: %s", validation_result)
    logger.info("Receipt result: %s", receipt_result)
    logger.info("Duplicate result: %s", duplicate_result)
    logger.info(
        "Policy lookup context length: %d characters",
        len(policy_context or ""),
    )
    logger.info("Policy section titles: %s", policy_section_titles)

    prompt = build_decision_prompt(
        claim=claim,
        validation_result=validation_result,
        receipt_result=receipt_result,
        duplicate_result=duplicate_result,
        policy_context=policy_context,
        policy_section_titles=policy_section_titles,
    )
    logger.info("Gemini prompt length: %d characters", len(prompt))

    raw_response = generate_json_response(prompt)
    logger.info("Gemini raw response: %s", raw_response)

    parsed = _parse_json_response(raw_response)
    decision = parse_ai_decision_json(parsed, validation_result, receipt_result)
    logger.info("Gemini decision parsed successfully: %s", decision.decision.value)
    return decision
