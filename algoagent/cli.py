from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from algoagent.agent import AlgoAgent
from algoagent.evaluation import evaluate_problems, write_report
from algoagent.executor import CppExecutor
from algoagent.model_client import RuleBasedModel
from algoagent.schema import AgentStatus, load_problem, load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="AlgoAgent command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="Solve one problem JSON file")
    _add_model_args(solve_parser)
    solve_parser.add_argument("--problem", required=True)
    solve_parser.add_argument("--max-repair-turns", type=int, default=3)
    solve_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a directory of problem JSON files")
    _add_model_args(eval_parser)
    eval_parser.add_argument("--problems", required=True)
    eval_parser.add_argument("--max-repair-turns", type=int, default=3)
    eval_parser.add_argument("--out", default="")
    eval_parser.add_argument("--compiler", default="g++")

    args = parser.parse_args()
    if args.command == "solve":
        bundle = load_problem(args.problem)
        agent = AlgoAgent(_make_model(args.backend, args.model), max_repair_turns=args.max_repair_turns)
        result = agent.solve(bundle.spec, bundle.tests)
        print(_format_solve_result(result, as_json=args.json))
        return

    if args.command == "evaluate":
        problems = load_problems(args.problems)
        agent = AlgoAgent(
            _make_model(args.backend, args.model),
            executor=CppExecutor(compiler=args.compiler),
            max_repair_turns=args.max_repair_turns,
        )
        report = evaluate_problems(agent, problems)
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
        if args.out:
            write_report(report, Path(args.out))


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["rule", "hf"], default="rule")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")


def _make_model(backend: str, model_name: str):
    if backend == "rule":
        return RuleBasedModel()
    if backend == "hf":
        from algoagent.hf_model import HuggingFaceModel

        return HuggingFaceModel(model_name)
    raise ValueError(f"Unsupported backend: {backend}")


def _format_solve_result(result, as_json: bool = False) -> str:
    verdict = {
        **asdict(result.resource_verdict),
        "time_status": result.resource_verdict.time_status.value,
        "memory_status": result.resource_verdict.memory_status.value,
    }
    if result.status == AgentStatus.SOLVED:
        payload = {
            "status": result.status.value,
            "attempts": result.attempts,
            "explanation": result.explanation,
            "complexity": asdict(result.complexity) if result.complexity else None,
            "resource_verdict": verdict,
            "code": result.code,
        }
    else:
        payload = {
            "status": result.status.value,
            "attempts": result.attempts,
            "failure_reason": result.failure_reason,
            "diagnostic_summary": result.diagnostic_summary,
            "resource_verdict": verdict,
        }
    if as_json:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if result.status == AgentStatus.FAILED:
        return _format_failed_result(payload)
    return _format_solved_result(payload)


def _format_solved_result(payload: dict) -> str:
    complexity = payload.get("complexity") or {}
    verdict = payload["resource_verdict"]
    return "\n".join(
        [
            f"Status: {payload['status']}",
            f"Attempts: {payload['attempts']}",
            "",
            "Explanation:",
            payload.get("explanation") or "",
            "",
            "Complexity:",
            f"- Time: {complexity.get('time_complexity', 'unknown')}",
            f"- Space: {complexity.get('space_complexity', 'unknown')}",
            f"- Estimated operations: {complexity.get('estimated_operations')}",
            f"- Estimated memory bytes: {complexity.get('estimated_memory_bytes')}",
            "",
            "Resource Verdict:",
            f"- Time: {verdict['time_status']} ({verdict['estimated_operations']}/{verdict['operation_budget']})",
            f"- Memory: {verdict['memory_status']} ({verdict['estimated_memory_bytes']}/{verdict['memory_budget_bytes']})",
            "",
            "Code:",
            "```cpp",
            payload.get("code") or "",
            "```",
        ]
    )


def _format_failed_result(payload: dict) -> str:
    verdict = payload["resource_verdict"]
    return "\n".join(
        [
            f"Status: {payload['status']}",
            f"Attempts: {payload['attempts']}",
            f"Failure Reason: {payload.get('failure_reason')}",
            "",
            "Diagnostic Summary:",
            payload.get("diagnostic_summary") or "",
            "",
            "Resource Verdict:",
            f"- Time: {verdict['time_status']} ({verdict['estimated_operations']}/{verdict['operation_budget']})",
            f"- Memory: {verdict['memory_status']} ({verdict['estimated_memory_bytes']}/{verdict['memory_budget_bytes']})",
        ]
    )


if __name__ == "__main__":
    main()
