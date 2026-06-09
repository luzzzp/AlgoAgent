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


DEFAULT_DATASET = "likaixin/TACO-verified"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TACO-verified rows to AlgoAgent problem JSON.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="data/problems/taco_verified")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many dataset rows before conversion.")
    parser.add_argument("--limit", type=int, default=0, help="Convert at most this many usable rows; 0 means no limit.")
    parser.add_argument("--repair-ratio", type=float, default=0.7)
    parser.add_argument("--min-tests", type=int, default=2)
    parser.add_argument("--max-repair-tests", type=int, default=20)
    parser.add_argument("--max-eval-tests", type=int, default=10)
    parser.add_argument("--target-language", default="cpp17")
    parser.add_argument("--include-function-tasks", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The converter requires Hugging Face datasets. Install with `pip install datasets` "
            "or `pip install -r requirements-train.txt`."
        ) from exc

    dataset = load_dataset(args.dataset, split=args.split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = {"function_task": 0, "bad_tests": 0, "no_solution": 0, "conversion_error": 0}
    for index, row in enumerate(dataset):
        if index < args.offset:
            continue
        if args.limit and written >= args.limit:
            break
        try:
            converted = convert_row(
                row,
                index=index,
                repair_ratio=args.repair_ratio,
                min_tests=args.min_tests,
                max_repair_tests=args.max_repair_tests,
                max_eval_tests=args.max_eval_tests,
                target_language=args.target_language,
                include_function_tasks=args.include_function_tasks,
            )
        except ConversionError as exc:
            skipped[exc.reason] = skipped.get(exc.reason, 0) + 1
            continue
        except Exception:
            skipped["conversion_error"] += 1
            continue
        target = out_dir / f"{converted['problem']['id']}.json"
        target.write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "offset": args.offset,
        "written": written,
        "skipped": skipped,
        "target_language": args.target_language,
        "repair_ratio": args.repair_ratio,
        "min_tests": args.min_tests,
        "max_repair_tests": args.max_repair_tests,
        "max_eval_tests": args.max_eval_tests,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


class ConversionError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def convert_row(
    row: dict[str, Any],
    index: int,
    repair_ratio: float = 0.7,
    min_tests: int = 2,
    max_repair_tests: int = 20,
    max_eval_tests: int = 10,
    target_language: str = "cpp17",
    include_function_tasks: bool = False,
) -> dict[str, Any]:
    input_output = _loads_jsonish(row.get("input_output"))
    if not isinstance(input_output, dict):
        raise ConversionError("bad_tests")
    if input_output.get("fn_name") and not include_function_tasks:
        raise ConversionError("function_task")

    inputs = _as_list(input_output.get("inputs"))
    outputs = _as_list(input_output.get("outputs"))
    if len(inputs) < min_tests or len(inputs) != len(outputs):
        raise ConversionError("bad_tests")

    solutions = _solutions_from_row(row)
    if not solutions:
        raise ConversionError("no_solution")

    title = str(row.get("name") or f"TACO problem {index}")
    problem_id = _slug(f"taco_{index}_{title}")
    repair_count = min(len(inputs) - 1, max(1, int(round(len(inputs) * repair_ratio))))
    tests = [
        {"stdin": _ensure_text(stdin), "expected_stdout": _ensure_text(stdout)}
        for stdin, stdout in zip(inputs, outputs)
    ]
    repair_tests = tests[:repair_count]
    eval_tests = tests[repair_count:]
    if max_repair_tests > 0:
        repair_tests = repair_tests[:max_repair_tests]
    if max_eval_tests > 0:
        eval_tests = eval_tests[:max_eval_tests]
    if not repair_tests or not eval_tests:
        raise ConversionError("bad_tests")
    expected_time = _first_non_empty(row, "Expected Time Complexity", "expected_time_complexity")
    expected_space = _first_non_empty(row, "Expected Auxiliary Space", "expected_space_complexity")
    expected_complexity = ", ".join(item for item in [expected_time, expected_space] if item)

    return {
        "problem": {
            "id": problem_id,
            "title": title,
            "statement": str(row.get("question") or ""),
            "input_format": "",
            "output_format": "",
            "constraints": _extract_constraint_lines(str(row.get("question") or "")),
            "language": target_language,
            "time_limit_sec": _parse_time_limit(row.get("time_limit")),
            "memory_limit_mb": _parse_memory_limit(row.get("memory_limit")),
        },
        "tests": {
            "repair_tests": repair_tests,
            "eval_tests": eval_tests,
        },
        "oracle": {
            "difficulty": str(row.get("difficulty") or ""),
            "tags": _extract_tags(row),
            "expected_complexity": expected_complexity,
            "reference_solution": _first_cpp_solution(solutions),
            "solutions": solutions,
            "source": str(row.get("source") or "taco-verified"),
            "url": str(row.get("url") or ""),
        },
    }


def _solutions_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _loads_jsonish(row.get("solutions"))
    items = _as_list(raw)
    solutions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        code = _solution_code(item)
        if not code or code in seen:
            continue
        seen.add(code)
        solutions.append({"language": _detect_language(code), "code": code, "verified": False})
    return solutions


def _solution_code(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("code", "solution", "text"):
            if item.get(key):
                return str(item[key])
    return ""


def _loads_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _ensure_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _detect_language(code: str) -> str:
    lowered = code.lower()
    if "#include" in lowered or "using namespace std" in lowered:
        return "cpp17"
    if "def " in lowered or "input(" in lowered or "sys.stdin" in lowered:
        return "python3"
    return "unknown"


def _first_cpp_solution(solutions: list[dict[str, Any]]) -> str:
    for solution in solutions:
        if solution["language"] == "cpp17" and solution.get("verified"):
            return str(solution["code"])
    return ""


def _extract_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in ("tags", "raw_tags", "skill_types"):
        value = _loads_jsonish(row.get(key))
        for item in _as_list(value):
            if item and str(item) not in tags:
                tags.append(str(item))
    return tags


def _extract_constraint_lines(statement: str) -> list[str]:
    lines = []
    for line in statement.splitlines():
        lowered = line.lower()
        if "constraint" in lowered or "<=" in line or "\u2264" in line:
            compact = line.strip()
            if compact:
                lines.append(compact)
    return lines


def _parse_time_limit(value: Any) -> float:
    if value is None or value == "":
        return 2.0
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
    return float(match.group(0)) if match else 2.0


def _parse_memory_limit(value: Any) -> int:
    if value is None or value == "":
        return 256
    match = re.search(r"[0-9]+", str(value))
    return int(match.group(0)) if match else 256


def _first_non_empty(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:120] or "taco_problem"


if __name__ == "__main__":
    main()
