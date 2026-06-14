from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.agent import PythonAlgoAgent
from algoagent.hf_model import HuggingFaceModel
from algoagent.schema import AgentStatus, load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AlgoAgent-Py end-to-end.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-repair-turns", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    if args.limit:
        problems = problems[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = _load_existing(out_path) if args.resume else []
    done = {item["problem_id"] for item in results}
    model = HuggingFaceModel(args.model, args.adapter, temperature=args.temperature, load_in_4bit=args.load_in_4bit)
    agent = PythonAlgoAgent(model, max_repair_turns=args.max_repair_turns)
    for idx, bundle in enumerate(problems, start=1):
        if bundle.spec.id in done:
            continue
        print(f"[{idx}/{len(problems)}] {bundle.spec.id}", flush=True)
        result = agent.solve(bundle.spec, bundle.tests)
        results.append(_serialize(result))
        _write(out_path, results, args)
        print(f"  status={result.status.value}; attempts={result.attempts}", flush=True)
    _write(out_path, results, args)
    print(json.dumps(_summary(results), indent=2, ensure_ascii=False))


def _serialize(result) -> dict:
    return {
        "problem_id": result.problem_id,
        "status": result.status.value,
        "attempts": result.attempts,
        "failure_reason": result.failure_reason,
        "diagnostic_summary": result.diagnostic_summary,
        "attempt_records": [
            {
                "turn": record.turn,
                "visible_pass_rate": record.visible_result.pass_rate if record.visible_result else 0.0,
                "eval_pass_rate": record.eval_result.pass_rate if record.eval_result else 0.0,
            }
            for record in result.attempt_records
        ],
    }


def _summary(results: list[dict]) -> dict[str, float | int | dict[str, int]]:
    failures: dict[str, int] = {}
    for result in results:
        if result.get("failure_reason"):
            failures[result["failure_reason"]] = failures.get(result["failure_reason"], 0) + 1
    return {
        "num_problems": len(results),
        "sample_solved_rate": _rate(item["status"] in {AgentStatus.SAMPLE_SOLVED.value, AgentStatus.VERIFIED_SOLVED.value} for item in results),
        "verified_success_rate": _rate(item["status"] == AgentStatus.VERIFIED_SOLVED.value for item in results),
        "avg_repair_turns": sum(max(0, int(item.get("attempts", 0)) - 1) for item in results) / len(results) if results else 0.0,
        "failure_breakdown": failures,
    }


def _rate(values) -> float:
    materialized = list(values)
    return sum(1 for value in materialized if value) / len(materialized) if materialized else 0.0


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("problems", []) if isinstance(payload.get("problems"), list) else []


def _write(path: Path, results: list[dict], args) -> None:
    payload = {"metadata": vars(args), "summary": _summary(results), "problems": results}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

