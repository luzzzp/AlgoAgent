from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

from algoagent.agent import AlgoAgent
from algoagent.schema import AgentResult, AgentStatus, CandidateResult, CheckStatus, ProblemBundle


def evaluate_problems(agent: AlgoAgent, problems: list[ProblemBundle]) -> dict[str, Any]:
    results = [agent.solve(bundle.spec, bundle.tests) for bundle in problems]
    return {
        "summary": summarize_results(results),
        "problems": [_serialize_agent_result(result) for result in results],
    }


def summarize_results(results: list[AgentResult]) -> dict[str, Any]:
    if not results:
        return {
            "num_problems": 0,
            "initial_compile_rate": 0.0,
            "final_compile_rate": 0.0,
            "repair_test_pass_rate": 0.0,
            "held_out_test_pass_rate": 0.0,
            "verified_success_rate": 0.0,
            "avg_repair_turns": 0.0,
            "theoretical_tle_rate": 0.0,
            "theoretical_mle_rate": 0.0,
            "complexity_unknown_rate": 0.0,
            "explanation_success_rate": 0.0,
            "failure_breakdown": {},
        }

    initial_candidates = [_first_candidate(result) for result in results]
    final_candidates = [_final_candidate(result) for result in results]
    final_repair_candidates = [_final_repair_candidate(result) for result in results]
    solved = [result for result in results if result.status == AgentStatus.SOLVED]
    failures: dict[str, int] = {}
    for result in results:
        if result.failure_reason:
            failures[result.failure_reason] = failures.get(result.failure_reason, 0) + 1

    return {
        "num_problems": len(results),
        "initial_compile_rate": _rate(candidate is not None and candidate.compiled for candidate in initial_candidates),
        "final_compile_rate": _rate(candidate is not None and candidate.compiled for candidate in final_candidates),
        "repair_test_pass_rate": mean(_candidate_pass_rate(candidate) for candidate in final_repair_candidates),
        "held_out_test_pass_rate": mean(_candidate_pass_rate(result.held_out_result) for result in results),
        "verified_success_rate": _rate(result.status == AgentStatus.SOLVED for result in results),
        "avg_repair_turns": mean(max(0, result.attempts - 1) for result in results),
        "theoretical_tle_rate": _rate(
            result.resource_verdict.time_status == CheckStatus.FAIL for result in results
        ),
        "theoretical_mle_rate": _rate(
            result.resource_verdict.memory_status == CheckStatus.FAIL for result in results
        ),
        "complexity_unknown_rate": _rate(result.resource_verdict.unknown for result in results),
        "explanation_success_rate": (
            _rate(bool(result.explanation) for result in solved) if solved else 0.0
        ),
        "failure_breakdown": failures,
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _first_candidate(result: AgentResult) -> CandidateResult | None:
    for record in result.attempt_records:
        if record.repair_result is not None:
            return record.repair_result
    return None


def _final_candidate(result: AgentResult) -> CandidateResult | None:
    if result.held_out_result is not None:
        return result.held_out_result
    for record in reversed(result.attempt_records):
        if record.repair_result is not None:
            return record.repair_result
    return None


def _final_repair_candidate(result: AgentResult) -> CandidateResult | None:
    for record in reversed(result.attempt_records):
        if record.repair_result is not None:
            return record.repair_result
    return None


def _candidate_pass_rate(candidate: CandidateResult | None) -> float:
    if candidate is None or not candidate.tests:
        return 0.0
    return _rate(test.passed for test in candidate.tests)


def _rate(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(1 for value in materialized if value) / len(materialized)


def _serialize_agent_result(result: AgentResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["status"] = result.status.value
    payload["resource_verdict"]["time_status"] = result.resource_verdict.time_status.value
    payload["resource_verdict"]["memory_status"] = result.resource_verdict.memory_status.value
    for record in payload["attempt_records"]:
        record["resource_verdict"]["time_status"] = record["resource_verdict"]["time_status"].value
        record["resource_verdict"]["memory_status"] = record["resource_verdict"]["memory_status"].value
    if result.code:
        payload["code_preview"] = result.code[:500]
    payload.pop("code", None)
    return payload
