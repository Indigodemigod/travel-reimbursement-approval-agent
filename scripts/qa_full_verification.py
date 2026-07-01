#!/usr/bin/env python
"""Full QA verification script — Phases 3 through 7."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.agent.graph import travel_reimbursement_graph  # noqa: E402
from app.main import app  # noqa: E402
from app.models import TravelClaim  # noqa: E402
from app.tools.policy_lookup import lookup_policy  # noqa: E402
from tests.conftest import gemini_response_payload  # noqa: E402
import app.tools.duplicate_checker as duplicate_checker_module  # noqa: E402

client = TestClient(app)

EXPECTED_NODES = [
    "check_duplicates",
    "validate_expenses",
    "check_receipts",
    "retrieve_policy",
    "ai_decision",
    "finalize_decision",
]


def _reset_dup() -> None:
    duplicate_checker_module._SUBMITTED_CLAIM_IDS.clear()
    duplicate_checker_module._EXPENSE_SIGNATURES.clear()


def _base_claim(claim_id: str, expenses: list[dict], **kwargs) -> dict:
    payload = {
        "claim_id": claim_id,
        "employee_id": "EMP001",
        "employee_name": "Jane Doe",
        "department": "Engineering",
        "trip_start_date": "2026-03-01",
        "trip_end_date": "2026-03-05",
        "destination": "Delhi",
        "purpose": "Client visit",
        "expenses": expenses,
    }
    payload.update(kwargs)
    return payload


def _exp(cat: str, amt: str, **kw) -> dict:
    return {
        "category": cat,
        "amount": amt,
        "currency": kw.get("currency", "INR"),
        "description": kw.get("description", "Business expense"),
        "receipt_attached": kw.get("receipt_attached", True),
    }


def _initial_state(claim: TravelClaim) -> dict:
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


def phase3_api_tests() -> list[dict]:
    print("\n" + "=" * 80)
    print("PHASE 3 - API ENDPOINT VERIFICATION")
    print("=" * 80)
    results = []

    def run(name: str, fn) -> None:
        _reset_dup()
        try:
            passed, detail = fn()
            status = "PASS" if passed else "FAIL"
        except Exception as exc:
            passed, detail, status = False, str(exc), "FAIL"
        results.append({"test": name, "result": status, "detail": detail})
        print(f"  [{status}] {name}: {detail}")

    def t_root():
        r = client.get("/")
        return r.status_code == 200 and "service" in r.json(), f"HTTP {r.status_code}"

    def t_health():
        r = client.get("/health")
        j = r.json()
        return r.status_code == 200 and j.get("status") == "healthy", f"HTTP {r.status_code}"

    def t_approve_ok(mock):
        mock.return_value = gemini_response_payload("approved", approved_amount="400")
        r = client.post("/approve", json=_base_claim("QA-OK", [_exp("meals", "400")]))
        return r.status_code == 200 and r.json()["decision"] == "approved", r.json().get("decision")

    def t_invalid():
        p = _base_claim("QA-INV", [_exp("meals", "400")])
        p["trip_end_date"] = "2026-02-01"
        r = client.post("/approve", json=p)
        return r.status_code == 422, f"HTTP {r.status_code}"

    def t_duplicate(mock):
        mock.return_value = gemini_response_payload("manual_review")
        p = _base_claim("QA-DUP", [_exp("meals", "400")])
        client.post("/approve", json=p)
        r = client.post("/approve", json=p)
        return r.status_code == 200 and r.json()["decision"] == "manual_review", r.json().get("decision")

    def t_missing_receipts(mock):
        mock.return_value = gemini_response_payload("manual_review")
        r = client.post(
            "/approve",
            json=_base_claim("QA-REC", [_exp("meals", "700", receipt_attached=False)]),
        )
        return r.status_code == 200 and r.json()["decision"] == "manual_review", r.json().get("decision")

    def t_hotel_violation(mock):
        mock.return_value = gemini_response_payload("rejected", rejected_amount="6000")
        r = client.post("/approve", json=_base_claim("QA-HOTEL", [_exp("hotel", "6000")]))
        return r.status_code == 200 and r.json()["decision"] == "rejected", r.json().get("decision")

    def t_meal_violation(mock):
        mock.return_value = gemini_response_payload("rejected", rejected_amount="2000")
        r = client.post("/approve", json=_base_claim("QA-MEAL", [_exp("meals", "2000")]))
        return r.status_code == 200 and r.json()["decision"] == "rejected", r.json().get("decision")

    def t_foreign(mock):
        mock.return_value = gemini_response_payload("manual_review")
        r = client.post(
            "/approve",
            json=_base_claim("QA-FX", [_exp("hotel", "200", currency="USD")]),
        )
        return r.status_code == 200 and r.json()["decision"] == "manual_review", r.json().get("decision")

    def t_manual_review(mock):
        mock.return_value = gemini_response_payload("manual_review")
        r = client.post("/approve", json=_base_claim("QA-OTHER", [_exp("other", "500")]))
        return r.status_code == 200 and r.json()["decision"] == "manual_review", r.json().get("decision")

    def t_partial(mock):
        mock.return_value = gemini_response_payload(
            "partially_approved", approved_amount="4500", rejected_amount="2000"
        )
        r = client.post(
            "/approve",
            json=_base_claim("QA-PART", [_exp("hotel", "4500"), _exp("meals", "2000")]),
        )
        return r.status_code == 200 and r.json()["decision"] == "partially_approved", r.json().get("decision")

    def t_fallback(mock):
        mock.side_effect = RuntimeError("Gemini unavailable")
        r = client.post("/approve", json=_base_claim("QA-FALL", [_exp("meals", "400")]))
        j = r.json()
        return r.status_code == 200 and j.get("decision") and j.get("explanation"), j.get("decision")

    run("GET /", t_root)
    run("GET /health", t_health)

    with patch("app.llm.decision_engine.generate_json_response") as mock:
        for name, fn in [
            ("POST /approve (approved)", lambda: t_approve_ok(mock)),
            ("POST /approve (invalid payload)", t_invalid),
            ("POST /approve (duplicate claim)", lambda: t_duplicate(mock)),
            ("POST /approve (missing receipts)", lambda: t_missing_receipts(mock)),
            ("POST /approve (hotel violation)", lambda: t_hotel_violation(mock)),
            ("POST /approve (meal violation)", lambda: t_meal_violation(mock)),
            ("POST /approve (foreign currency)", lambda: t_foreign(mock)),
            ("POST /approve (manual review)", lambda: t_manual_review(mock)),
            ("POST /approve (partial approval)", lambda: t_partial(mock)),
            ("POST /approve (Gemini fallback)", lambda: t_fallback(mock)),
        ]:
            run(name, fn)

    # Swagger OpenAPI schema reachable
    r = client.get("/openapi.json")
    swagger_ok = r.status_code == 200 and "/approve" in r.text
    results.append({"test": "Swagger OpenAPI schema", "result": "PASS" if swagger_ok else "FAIL", "detail": f"HTTP {r.status_code}"})
    print(f"  [{'PASS' if swagger_ok else 'FAIL'}] Swagger OpenAPI schema: HTTP {r.status_code}")

    return results


def phase4_langgraph() -> list[dict]:
    print("\n" + "=" * 80)
    print("PHASE 4 - LANGGRAPH NODE EXECUTION")
    print("=" * 80)
    _reset_dup()
    claim = TravelClaim(**_base_claim("QA-GRAPH", [_exp("meals", "400", description="Lunch")]))
    results = []
    with patch(
        "app.llm.decision_engine.generate_json_response",
        return_value=gemini_response_payload("approved", approved_amount="400"),
    ):
        snapshots = []
        for event in travel_reimbursement_graph.stream(_initial_state(claim)):
            node = next(iter(event))
            snapshots.append(node)
            print(f"  -> {node} (current_step={event[node]['current_step']})")

    counts = {n: snapshots.count(n) for n in EXPECTED_NODES}
    all_once = all(counts[n] == 1 for n in EXPECTED_NODES)
    order_ok = snapshots == EXPECTED_NODES

    print(f"\n  Node counts: {counts}")
    print(f"  Order correct: {order_ok}")
    print(f"  Each node exactly once: {all_once}")

    results.append({"test": "Node execution order", "result": "PASS" if order_ok else "FAIL"})
    results.append({"test": "Each node runs once", "result": "PASS" if all_once else "FAIL"})
    return results


def phase5_fallback() -> list[dict]:
    print("\n" + "=" * 80)
    print("PHASE 5 - GEMINI FALLBACK BEHAVIOUR")
    print("=" * 80)
    from app.models import ExpenseCategory
    from tests.conftest import make_claim, make_expense

    failure_cases = [
        ("timeout", {"side_effect": TimeoutError("Gemini timeout")}),
        ("invalid JSON", {"return_value": "not-valid-json{{{"}),
        ("missing fields", {"return_value": json.dumps({"decision": "approved"})}),
        ("empty response", {"return_value": ""}),
        ("API exception", {"side_effect": RuntimeError("API rate limit")}),
    ]
    results = []

    for name, mock_kwargs in failure_cases:
        _reset_dup()
        claim = make_claim("QA-FB-" + name[:3].replace(" ", ""), [make_expense(ExpenseCategory.MEALS, "400")])

        with patch("app.llm.decision_engine.generate_json_response", **mock_kwargs):
            result = travel_reimbursement_graph.invoke(_initial_state(claim))
            fd = result.get("final_decision")
            ok = fd is not None and fd.decision is not None and fd.explanation
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}: decision={fd.decision.value if fd else None}, errors={len(result.get('errors', []))}")
            results.append({"test": f"Fallback: {name}", "result": status})

    return results


def phase6_edge_cases() -> list[dict]:
    print("\n" + "=" * 80)
    print("PHASE 6 - EDGE CASE SPOT CHECKS")
    print("=" * 80)
    results = []

    cases = [
        ("Empty expense list", _base_claim("EC-1", []), 422, None),
        (
            "100 expenses",
            _base_claim("EC-2", [_exp("meals", "10", description=f"Item {i}") for i in range(100)]),
            200,
            None,
        ),
        (
            "Unicode descriptions",
            _base_claim("EC-3", [_exp("meals", "400", description="東京ラーメン 🍜")]),
            200,
            None,
        ),
        (
            "Huge description (500 chars)",
            _base_claim("EC-4", [_exp("meals", "400", description="A" * 500)]),
            200,
            None,
        ),
        ("Zero amount", _base_claim("EC-5", [_exp("meals", "0")]), 422, None),
        ("Negative amount", _base_claim("EC-6", [_exp("meals", "-50")]), 422, None),
        (
            "Unknown category",
            _base_claim("EC-7", [{"category": "spaceship", "amount": "100", "currency": "INR", "description": "x", "receipt_attached": True}]),
            422,
            None,
        ),
        (
            "Unknown transport",
            _base_claim("EC-8", [_exp("transport", "500", description="Private helicopter")]),
            200,
            "manual_review",
        ),
        (
            "Very large claim",
            _base_claim(
                "EC-9",
                [_exp("flight", "15000")] + [_exp("hotel", "4900") for _ in range(5)],
            ),
            200,
            None,
        ),
        (
            "Duplicate receipts (duplicate lines)",
            _base_claim("EC-10", [_exp("meals", "400"), _exp("meals", "400")]),
            200,
            "manual_review",
        ),
    ]

    with patch("app.llm.decision_engine.generate_json_response") as mock:
        mock.return_value = gemini_response_payload("manual_review")
        for name, payload, exp_status, exp_decision in cases:
            _reset_dup()
            r = client.post("/approve", json=payload)
            ok = r.status_code == exp_status
            if exp_status == 200 and exp_decision:
                ok = ok and r.json().get("decision") == exp_decision
            status = "PASS" if ok else "FAIL"
            detail = f"HTTP {r.status_code}"
            if r.status_code == 200:
                detail += f" decision={r.json().get('decision')}"
            print(f"  [{status}] {name}: {detail}")
            results.append({"test": name, "result": status, "detail": detail})

    return results


def phase7_benchmark() -> dict:
    print("\n" + "=" * 80)
    print("PHASE 7 - PERFORMANCE BENCHMARK")
    print("=" * 80)
    latencies: list[float] = []
    gemini_latencies: list[float] = []
    policy_latencies: list[float] = []
    failures = 0

    def timed_gemini(prompt: str) -> str:
        start = time.perf_counter()
        time.sleep(0.002)  # simulate minimal LLM latency
        gemini_latencies.append(time.perf_counter() - start)
        return gemini_response_payload("approved", approved_amount="400")

    payload = _base_claim("BENCH", [_exp("meals", "400")])

    for n_requests in [50, 100]:
        _reset_dup()
        batch: list[float] = []
        batch_fail = 0
        with patch("app.llm.decision_engine.generate_json_response", side_effect=timed_gemini):
            for i in range(n_requests):
                p = dict(payload)
                p["claim_id"] = f"BENCH-{n_requests}-{i}"
                start = time.perf_counter()
                pol_start = time.perf_counter()
                lookup_policy("hotel meals taxi " + str(i))
                policy_latencies.append(time.perf_counter() - pol_start)
                r = client.post("/approve", json=p)
                elapsed = time.perf_counter() - start
                batch.append(elapsed)
                if r.status_code != 200:
                    batch_fail += 1
        latencies.extend(batch)
        failures += batch_fail
        print(
            f"  {n_requests} requests: avg={statistics.mean(batch)*1000:.1f}ms "
            f"max={max(batch)*1000:.1f}ms fail={batch_fail}"
        )

    stats = {
        "total_requests": len(latencies),
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "success_pct": (len(latencies) - failures) / len(latencies) * 100,
        "failures": failures,
        "avg_gemini_ms": statistics.mean(gemini_latencies) * 1000 if gemini_latencies else 0,
        "avg_policy_ms": statistics.mean(policy_latencies) * 1000 if policy_latencies else 0,
    }
    print(f"\n  Aggregate: avg={stats['avg_latency_ms']:.1f}ms max={stats['max_latency_ms']:.1f}ms "
          f"success={stats['success_pct']:.1f}% failures={stats['failures']}")
    print(f"  Avg Gemini (mocked): {stats['avg_gemini_ms']:.2f}ms")
    print(f"  Avg policy lookup: {stats['avg_policy_ms']:.2f}ms")
    return stats


def main() -> int:
    p3 = phase3_api_tests()
    p4 = phase4_langgraph()
    p5 = phase5_fallback()
    p6 = phase6_edge_cases()
    p7 = phase7_benchmark()

    all_results = p3 + p4 + p5 + p6
    failed = sum(1 for r in all_results if r["result"] == "FAIL")

    print("\n" + "=" * 80)
    print("PHASE 3-7 SUMMARY")
    print("=" * 80)
    print(f"  Checks run: {len(all_results)}")
    print(f"  Passed: {len(all_results) - failed}")
    print(f"  Failed: {failed}")

    return failed


if __name__ == "__main__":
    sys.exit(main())
