from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.agent import AlgoAgent
from algoagent.evaluation import _serialize_agent_result, summarize_results
from algoagent.executor import CppExecutor
from algoagent.hf_model import HuggingFaceModel
from algoagent.schema import AgentResult, load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HF base model or LoRA adapter on AlgoAgent problems.")
    parser.add_argument("--problems", required=True, help="Problem JSON file or directory.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", default="", help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--out", required=True, help="Path to the JSON report.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most this many problems; 0 means all.")
    parser.add_argument("--max-repair-turns", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-explanations", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip problem ids already present in --out.")
    parser.add_argument("--save-failed-code", action="store_true", help="Store generated attempt code for offline DPO data construction.")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    if args.limit:
        problems = problems[: args.limit]

    model = HuggingFaceModel(
        args.model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
    )
    agent = AlgoAgent(
        model,
        executor=CppExecutor(compiler=args.compiler),
        max_repair_turns=args.max_repair_turns,
        explain_on_success=not args.skip_explanations,
        capture_attempt_code=args.save_failed_code,
    )

    metadata = {
        "model": args.model,
        "adapter": args.adapter,
        "problems": args.problems,
        "limit": args.limit,
        "max_repair_turns": args.max_repair_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "load_in_4bit": args.load_in_4bit,
        "skip_explanations": args.skip_explanations,
        "compiler": args.compiler,
        "resume": args.resume,
        "save_failed_code": args.save_failed_code,
    }
    results: list[AgentResult] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_problems = _load_existing_problems(out_path) if args.resume else []
    completed_ids = {str(item.get("problem_id")) for item in existing_problems}

    for index, bundle in enumerate(problems, start=1):
        if bundle.spec.id in completed_ids:
            print(f"[{index}/{len(problems)}] {bundle.spec.id} skipped (already in report)", flush=True)
            continue
        print(f"[{index}/{len(problems)}] {bundle.spec.id}", flush=True)
        result = agent.solve(bundle.spec, bundle.tests)
        results.append(result)
        report = _write_report(out_path, metadata, existing_problems, results)
        summary = report["summary"]
        print(
            "  "
            f"status={result.status.value}; "
            f"verified_success_rate={summary['verified_success_rate']:.3f}; "
            f"final_compile_rate={summary['final_compile_rate']:.3f}",
            flush=True,
        )

    report = _write_report(out_path, metadata, existing_problems, results)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


def _write_report(
    path: Path,
    metadata: dict[str, object],
    existing_problems: list[dict[str, object]],
    results: list[AgentResult],
) -> dict[str, object]:
    serialized = [*existing_problems, *[_serialize_agent_result(result) for result in results]]
    report = {
        "metadata": metadata,
        "summary": _summarize_serialized(serialized),
        "problems": serialized,
    }
    path.write_text(json.dumps(_json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _load_existing_problems(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = payload.get("problems", [])
    if not isinstance(problems, list):
        return []
    return [item for item in problems if isinstance(item, dict) and item.get("problem_id")]


def _summarize_serialized(problems: list[dict[str, object]]) -> dict[str, object]:
    if not problems:
        return summarize_results([])
    failures: dict[str, int] = {}
    for problem in problems:
        reason = problem.get("failure_reason")
        if reason:
            failures[str(reason)] = failures.get(str(reason), 0) + 1
    solved = [problem for problem in problems if problem.get("status") == "SOLVED"]
    return {
        "num_problems": len(problems),
        "initial_compile_rate": _rate(_candidate_compiled(_first_candidate(problem)) for problem in problems),
        "final_compile_rate": _rate(_candidate_compiled(_final_candidate(problem)) for problem in problems),
        "repair_test_pass_rate": _mean(_candidate_pass_rate(_final_repair_candidate(problem)) for problem in problems),
        "held_out_test_pass_rate": _mean(_candidate_pass_rate(problem.get("held_out_result")) for problem in problems),
        "verified_success_rate": _rate(problem.get("status") == "SOLVED" for problem in problems),
        "avg_repair_turns": _mean(max(0, int(problem.get("attempts") or 0) - 1) for problem in problems),
        "theoretical_tle_rate": _rate(_resource_status(problem, "time_status") == "FAIL" for problem in problems),
        "theoretical_mle_rate": _rate(_resource_status(problem, "memory_status") == "FAIL" for problem in problems),
        "complexity_unknown_rate": _rate(
            _resource_status(problem, "time_status") == "UNKNOWN"
            or _resource_status(problem, "memory_status") == "UNKNOWN"
            for problem in problems
        ),
        "explanation_success_rate": _rate(bool(problem.get("explanation")) for problem in solved) if solved else 0.0,
        "failure_breakdown": failures,
    }


def _first_candidate(problem: dict[str, object]) -> dict[str, object] | None:
    for record in problem.get("attempt_records") or []:
        if isinstance(record, dict) and isinstance(record.get("repair_result"), dict):
            return record["repair_result"]
    return None


def _final_candidate(problem: dict[str, object]) -> dict[str, object] | None:
    held_out = problem.get("held_out_result")
    if isinstance(held_out, dict):
        return held_out
    return _final_repair_candidate(problem)


def _final_repair_candidate(problem: dict[str, object]) -> dict[str, object] | None:
    records = problem.get("attempt_records") or []
    for record in reversed(records):
        if isinstance(record, dict) and isinstance(record.get("repair_result"), dict):
            return record["repair_result"]
    return None


def _candidate_compiled(candidate: dict[str, object] | None) -> bool:
    return bool(candidate and candidate.get("compiled"))


def _candidate_pass_rate(candidate: object) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    tests = candidate.get("tests")
    if not isinstance(tests, list) or not tests:
        return 0.0
    return _rate(isinstance(test, dict) and test.get("passed") for test in tests)


def _resource_status(problem: dict[str, object], key: str) -> str:
    verdict = problem.get("resource_verdict")
    if not isinstance(verdict, dict):
        return ""
    return str(verdict.get(key) or "")


def _mean(values) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _rate(values) -> float:
    materialized = list(values)
    return sum(1 for value in materialized if value) / len(materialized) if materialized else 0.0


def _json_ready(payload):
    if hasattr(payload, "value"):
        return payload.value
    if hasattr(payload, "__dataclass_fields__"):
        return _json_ready(asdict(payload))
    if isinstance(payload, dict):
        return {key: _json_ready(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_ready(value) for value in payload]
    return payload


if __name__ == "__main__":
    main()
