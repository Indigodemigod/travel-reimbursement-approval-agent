#!/usr/bin/env python
"""End-to-end travel reimbursement scenario runner.

Calls POST /approve for each scenario and prints a PASS/FAIL summary table.
Gemini is mocked to return decisions aligned with deterministic tool outputs.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app
from app.models import ApprovalStatus, TravelClaim
from app.tools.expense_validator import validate_expenses
from app.tools.receipt_checker import check_receipts
import app.tools.duplicate_checker as duplicate_checker_module

client = TestClient(app)


@dataclass
class Scenario:
    name: str
    payload: dict[str, Any]
    expected_status: int
    expected_decision: str | None = None
    notes: str = ""


def _expense(
    category: str,
    amount: str,
    *,
    currency: str = "INR",
    description: str = "Business expense",
    receipt_attached: bool = True,
) -> dict[str, Any]:
    return {
        "category": category,
        "amount": amount,
        "currency": currency,
        "description": description,
        "receipt_attached": receipt_attached,
    }


def _claim(
    claim_id: str,
    expenses: list[dict[str, Any]],
    *,
    purpose: str = "Client visit",
    destination: str = "Delhi",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "employee_id": "EMP001",
        "employee_name": "Jane Doe",
        "department": "Engineering",
        "trip_start_date": "2026-03-01",
        "trip_end_date": "2026-03-05",
        "destination": destination,
        "purpose": purpose,
        "expenses": expenses,
    }


def _build_gemini_json(payload: dict[str, Any], decision: str) -> str:
    claim = TravelClaim(**payload)
    validation_result = validate_expenses(claim.expenses)
    receipt_result = check_receipts(claim.expenses)

    return json.dumps(
        {
            "decision": decision,
            "approved_amount": str(validation_result["approved_amount"]),
            "rejected_amount": str(validation_result["rejected_amount"]),
            "missing_documents": [
                f"{item['description']} ({item['category']})"
                for item in receipt_result.get("missing_receipts", [])
            ],
            "violated_policies": validation_result.get("violations", []),
            "confidence": 0.95,
            "explanation": f"Decision: {decision} based on tool validation outputs.",
        }
    )


def _reset_duplicate_registry() -> None:
    duplicate_checker_module._SUBMITTED_CLAIM_IDS.clear()
    duplicate_checker_module._EXPENSE_SIGNATURES.clear()


SCENARIOS: list[Scenario] = [
    Scenario(
        "01_simple_approved",
        _claim(
            "E2E-01",
            [_expense("meals", "400", description="Team lunch")],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Clean claim within all limits",
    ),
    Scenario(
        "02_hotel_above_limit",
        _claim(
            "E2E-02",
            [_expense("hotel", "6000", description="Luxury hotel stay")],
        ),
        200,
        ApprovalStatus.REJECTED.value,
        "Hotel exceeds 5000 INR/night",
    ),
    Scenario(
        "03_meals_above_limit",
        _claim(
            "E2E-03",
            [_expense("meals", "2000", description="Client dinner")],
        ),
        200,
        ApprovalStatus.REJECTED.value,
        "Meals exceed 1500 INR/day",
    ),
    Scenario(
        "04_taxi_reimbursement",
        _claim(
            "E2E-04",
            [_expense("transport", "450", description="Uber taxi to client")],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Allowed taxi transport",
    ),
    Scenario(
        "05_personal_petrol",
        _claim(
            "E2E-05",
            [_expense("transport", "2000", description="Petrol for personal car")],
        ),
        200,
        ApprovalStatus.REJECTED.value,
        "Personal fuel not reimbursable",
    ),
    Scenario(
        "06_foreign_currency",
        _claim(
            "E2E-06",
            [_expense("hotel", "180", currency="USD", description="US hotel")],
        ),
        200,
        ApprovalStatus.MANUAL_REVIEW.value,
        "Foreign currency requires review",
    ),
    Scenario(
        "07_duplicate_expenses",
        _claim(
            "E2E-07",
            [
                _expense("meals", "500", description="Lunch"),
                _expense("meals", "500", description="Lunch"),
            ],
        ),
        200,
        ApprovalStatus.MANUAL_REVIEW.value,
        "Duplicate line items in claim",
    ),
    Scenario(
        "08_missing_receipts",
        _claim(
            "E2E-08",
            [
                _expense(
                    "meals",
                    "700",
                    description="Dinner",
                    receipt_attached=False,
                )
            ],
        ),
        200,
        ApprovalStatus.MANUAL_REVIEW.value,
        "Receipt missing above 500 INR",
    ),
    Scenario(
        "09_multiple_violations",
        _claim(
            "E2E-09",
            [
                _expense("hotel", "6500", description="Hotel"),
                _expense("meals", "1800", description="Meals"),
            ],
        ),
        200,
        ApprovalStatus.REJECTED.value,
        "Hotel and meals both exceed limits",
    ),
    Scenario(
        "10_partial_approval",
        _claim(
            "E2E-10",
            [
                _expense("hotel", "4500", description="Valid hotel"),
                _expense("meals", "2000", description="Excessive meals"),
            ],
        ),
        200,
        ApprovalStatus.PARTIALLY_APPROVED.value,
        "Some expenses valid, some rejected",
    ),
    Scenario(
        "11_manual_review_other",
        _claim("E2E-11", [_expense("other", "500", description="Gift hamper")]),
        200,
        ApprovalStatus.MANUAL_REVIEW.value,
        "Unsupported OTHER category",
    ),
    Scenario(
        "12_unsupported_transport",
        _claim(
            "E2E-12",
            [_expense("transport", "5000", description="Helicopter charter")],
        ),
        200,
        ApprovalStatus.MANUAL_REVIEW.value,
        "Unrecognized transport mode",
    ),
    Scenario(
        "13_flight_only",
        _claim(
            "E2E-13",
            [_expense("flight", "8500", description="Economy flight to Mumbai")],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Flight within policy",
    ),
    Scenario(
        "14_hotel_and_meals",
        _claim(
            "E2E-14",
            [
                _expense("hotel", "4800", description="Hotel"),
                _expense("meals", "1200", description="Meals"),
            ],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Valid hotel and meals combo",
    ),
    Scenario(
        "15_hotel_and_taxi",
        _claim(
            "E2E-15",
            [
                _expense("hotel", "4200", description="Hotel"),
                _expense("transport", "350", description="Ola cab"),
            ],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Valid hotel and taxi combo",
    ),
    Scenario(
        "16_train_travel",
        _claim(
            "E2E-16",
            [_expense("transport", "1200", description="Train to Pune")],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Train is allowed transport",
    ),
    Scenario(
        "17_metro_travel",
        _claim(
            "E2E-17",
            [_expense("transport", "80", description="Delhi metro card")],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Metro is allowed transport",
    ),
    Scenario(
        "18_large_claim",
        _claim(
            "E2E-18",
            [
                _expense("flight", "12000", description="Flight"),
                _expense("hotel", "4900", description="Hotel night 1"),
                _expense("hotel", "4900", description="Hotel night 2"),
                _expense("meals", "1400", description="Meals day 1"),
                _expense("meals", "1300", description="Meals day 2"),
                _expense("transport", "900", description="Taxi"),
            ],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Large but compliant claim (>50k total)",
    ),
    Scenario(
        "19_mixed_currencies",
        _claim(
            "E2E-19",
            [
                _expense("hotel", "4500", description="INR hotel"),
                _expense("meals", "50", currency="USD", description="US meal"),
            ],
        ),
        200,
        ApprovalStatus.MANUAL_REVIEW.value,
        "Mixed INR and foreign currency",
    ),
    Scenario(
        "20_invalid_payload",
        {
            "claim_id": "E2E-20",
            "employee_id": "EMP001",
            "employee_name": "Jane Doe",
            "department": "Engineering",
            "trip_start_date": "2026-03-10",
            "trip_end_date": "2026-03-01",
            "destination": "Delhi",
            "purpose": "Invalid dates",
            "expenses": [_expense("meals", "400", description="Lunch")],
        },
        422,
        None,
        "trip_end_date before trip_start_date",
    ),
    Scenario(
        "21_bus_travel",
        _claim(
            "E2E-21",
            [_expense("transport", "600", description="Intercity bus")],
        ),
        200,
        ApprovalStatus.APPROVED.value,
        "Bus is allowed transport",
    ),
    Scenario(
        "22_receipt_and_violation",
        _claim(
            "E2E-22",
            [
                _expense(
                    "hotel",
                    "5500",
                    description="Hotel no receipt",
                    receipt_attached=False,
                )
            ],
        ),
        200,
        ApprovalStatus.REJECTED.value,
        "Policy violation dominates missing receipt",
    ),
]


def _format_request(payload: dict[str, Any]) -> str:
    expenses = payload.get("expenses", [])
    if not expenses:
        return "invalid payload"
    first = expenses[0]
    extra = f" +{len(expenses) - 1} more" if len(expenses) > 1 else ""
    return (
        f"{payload.get('claim_id')} | "
        f"{first.get('category')} {first.get('amount')} {first.get('currency')}"
        f"{extra}"
    )


def run_scenarios() -> int:
    results: list[dict[str, Any]] = []
    failures = 0

    print("=" * 120)
    print("TRAVEL REIMBURSEMENT E2E SCENARIO RUNNER")
    print("=" * 120)

    for scenario in SCENARIOS:
        _reset_duplicate_registry()

        def _mock_gemini(_prompt: str) -> str:
            return _build_gemini_json(
                scenario.payload,
                scenario.expected_decision or ApprovalStatus.MANUAL_REVIEW.value,
            )

        start = time.perf_counter()
        with patch(
            "app.llm.decision_engine.generate_json_response",
            side_effect=_mock_gemini,
        ):
            response = client.post("/approve", json=scenario.payload)
        elapsed = time.perf_counter() - start

        actual_decision = None
        if response.status_code == 200:
            actual_decision = response.json().get("decision")

        if scenario.expected_status == 422:
            passed = response.status_code == 422
            expected_display = "HTTP 422"
            actual_display = f"HTTP {response.status_code}"
        else:
            passed = (
                response.status_code == scenario.expected_status
                and actual_decision == scenario.expected_decision
            )
            expected_display = scenario.expected_decision or "N/A"
            actual_display = actual_decision or f"HTTP {response.status_code}"

        if not passed:
            failures += 1

        row = {
            "scenario": scenario.name,
            "request": _format_request(scenario.payload),
            "expected": expected_display,
            "actual": actual_display,
            "pass_fail": "PASS" if passed else "FAIL",
            "time_s": f"{elapsed:.3f}",
            "notes": scenario.notes,
        }
        results.append(row)

        print(f"\n[{scenario.name}]")
        print(f"  Request : {row['request']}")
        print(f"  Expected: {row['expected']} ({scenario.notes})")
        print(f"  Actual  : {row['actual']}")
        print(f"  Result  : {row['pass_fail']} | Time: {row['time_s']}s")

    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(
        f"{'Scenario':<28} {'Request':<35} {'Expected':<18} "
        f"{'Actual':<18} {'Result':<6} {'Time(s)':<8}"
    )
    print("-" * 120)
    for row in results:
        print(
            f"{row['scenario']:<28} {row['request']:<35} {row['expected']:<18} "
            f"{row['actual']:<18} {row['pass_fail']:<6} {row['time_s']:<8}"
        )

    print("-" * 120)
    print(f"Total: {len(results)} | Passed: {len(results) - failures} | Failed: {failures}")
    print("=" * 120)

    return failures


if __name__ == "__main__":
    sys.exit(run_scenarios())
