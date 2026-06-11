from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.executor import CppExecutor
from algoagent.hf_model import HuggingFaceModel
from algoagent.schema import ProblemBundle, load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SFT format adherence for HF models.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
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
    executor = CppExecutor(compiler=args.compiler)
    results = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "adapter": args.adapter,
        "problems": args.problems,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "load_in_4bit": args.load_in_4bit,
        "compiler": args.compiler,
    }

    for index, bundle in enumerate(problems, start=1):
        print(f"[{index}/{len(problems)}] {bundle.spec.id}", flush=True)
        completion = generate_format_response(model, bundle)
        result = evaluate_completion(bundle, completion, executor)
        results.append(result)
        report = {"metadata": metadata, "summary": summarize(results), "problems": results}
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            "  "
            f"format_valid={result['format_valid']}; "
            f"cpp_block={result['cpp_code_block']}; "
            f"compiled={result['compiled']}",
            flush=True,
        )

    report = {"metadata": metadata, "summary": summarize(results), "problems": results}
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


def generate_format_response(model: HuggingFaceModel, bundle: ProblemBundle) -> str:
    prompt = (
        f"{bundle.spec.prompt()}\n\n"
        "Use this exact response structure:\n"
        "Solution Explanation:\n"
        "...\n\n"
        "Time Complexity: O(...)\n"
        "Space Complexity: O(...)\n"
        "```cpp\n"
        "...\n"
        "```"
    )
    return model._generate(
        "You are AlgoAgent. Return a structured competitive programming solution.",
        prompt,
    )


def evaluate_completion(bundle: ProblemBundle, completion: str, executor: CppExecutor) -> dict[str, Any]:
    explanation_field = has_explanation_field(completion)
    complexity_field = has_complexity_fields(completion)
    cpp_code_block = has_cpp_code_block(completion)
    code = extract_cpp_code(completion)
    compiled = bool(code) and executor.evaluate(code, [], default_timeout_sec=bundle.spec.time_limit_sec).compiled
    return {
        "problem_id": bundle.spec.id,
        "format_valid": explanation_field and complexity_field and cpp_code_block,
        "explanation_field": explanation_field,
        "complexity_field": complexity_field,
        "cpp_code_block": cpp_code_block,
        "compiled": compiled,
        "response_preview": completion[:1000],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "num_problems": len(results),
        "format_valid_rate": rate(item["format_valid"] for item in results),
        "explanation_field_rate": rate(item["explanation_field"] for item in results),
        "complexity_field_rate": rate(item["complexity_field"] for item in results),
        "cpp_code_block_rate": rate(item["cpp_code_block"] for item in results),
        "compile_rate": rate(item["compiled"] for item in results),
    }


def has_explanation_field(text: str) -> bool:
    return bool(re.search(r"solution\s+explanation\s*:", text, flags=re.I))


def has_complexity_fields(text: str) -> bool:
    return bool(re.search(r"time\s+complexity\s*:", text, flags=re.I)) and bool(
        re.search(r"space\s+complexity\s*:", text, flags=re.I)
    )


def has_cpp_code_block(text: str) -> bool:
    return bool(re.search(r"```(?:cpp|c\+\+)\s*.*?```", text, flags=re.I | re.S))


def extract_cpp_code(text: str) -> str:
    match = re.search(r"```(?:cpp|c\+\+)\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    include_at = text.find("#include")
    return text[include_at:].strip() if include_at >= 0 else ""


def rate(values) -> float:
    materialized = list(values)
    return sum(1 for value in materialized if value) / len(materialized) if materialized else 0.0


if __name__ == "__main__":
    main()
