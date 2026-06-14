from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.executor import PythonExecutor
from algoagent.hf_model import HuggingFaceModel, SOLVER_SYSTEM_PROMPT
from algoagent.schema import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate solver format adherence.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
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
    executor = PythonExecutor()
    for idx, bundle in enumerate(problems, start=1):
        if bundle.spec.id in done:
            continue
        print(f"[{idx}/{len(problems)}] {bundle.spec.id}", flush=True)
        prompt = bundle.spec.prompt(bundle.tests.visible_tests)
        text = model._generate(
            SOLVER_SYSTEM_PROMPT,
            prompt
            + "\nUse exact sections: Solution Explanation, Time Complexity, Space Complexity, and ```python code block.",
        )
        code = _extract_code(text)
        syntax_valid = executor.evaluate(code, [], bundle.spec.time_limit_sec).syntax_valid if code else False
        result = {
            "problem_id": bundle.spec.id,
            "format_valid": has_explanation(text) and has_complexity(text) and has_python_block(text),
            "chinese_explanation": has_chinese_explanation(text),
            "complexity_field": has_complexity(text),
            "python_code_block": has_python_block(text),
            "syntax_valid": syntax_valid,
            "response_preview": text[:1200],
        }
        results.append(result)
        _write(out_path, results, args)
        print(f"  format={result['format_valid']}; syntax={syntax_valid}", flush=True)
    _write(out_path, results, args)
    print(json.dumps(_summary(results), indent=2, ensure_ascii=False))


def has_explanation(text: str) -> bool:
    return bool(re.search(r"solution\s+explanation\s*:?", text, flags=re.I))


def has_chinese_explanation(text: str) -> bool:
    return has_explanation(text) and bool(re.search(r"[\u4e00-\u9fff]", text))


def has_complexity(text: str) -> bool:
    return bool(re.search(r"time\s+complexity\s*:?", text, flags=re.I)) and bool(
        re.search(r"space\s+complexity\s*:?", text, flags=re.I)
    )


def has_python_block(text: str) -> bool:
    return bool(re.search(r"```(?:python|py)\s*.*?```", text, flags=re.I | re.S))


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python|py)\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def _summary(results: list[dict]) -> dict[str, float | int]:
    return {
        "num_problems": len(results),
        "format_valid_rate": _rate(item["format_valid"] for item in results),
        "chinese_explanation_rate": _rate(item["chinese_explanation"] for item in results),
        "complexity_field_rate": _rate(item["complexity_field"] for item in results),
        "python_code_block_rate": _rate(item["python_code_block"] for item in results),
        "syntax_valid_rate": _rate(item["syntax_valid"] for item in results),
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
    payload = {
        "metadata": vars(args),
        "summary": _summary(results),
        "problems": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

