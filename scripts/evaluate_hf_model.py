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
    }
    results: list[AgentResult] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for index, bundle in enumerate(problems, start=1):
        print(f"[{index}/{len(problems)}] {bundle.spec.id}", flush=True)
        result = agent.solve(bundle.spec, bundle.tests)
        results.append(result)
        _write_report(out_path, metadata, results)
        summary = summarize_results(results)
        print(
            "  "
            f"status={result.status.value}; "
            f"verified_success_rate={summary['verified_success_rate']:.3f}; "
            f"final_compile_rate={summary['final_compile_rate']:.3f}",
            flush=True,
        )

    report = _write_report(out_path, metadata, results)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


def _write_report(path: Path, metadata: dict[str, object], results: list[AgentResult]) -> dict[str, object]:
    report = {
        "metadata": metadata,
        "summary": summarize_results(results),
        "problems": [_serialize_agent_result(result) for result in results],
    }
    path.write_text(json.dumps(_json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


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
