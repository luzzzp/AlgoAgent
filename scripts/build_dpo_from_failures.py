from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problems
from scripts.make_datasets import _format_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO pairs from failed model attempts and verified C++ oracle.")
    parser.add_argument("--reports", nargs="+", required=True, help="Evaluation reports generated with --save-failed-code.")
    parser.add_argument("--problems", required=True, help="Problem directory containing verified cpp17 oracle solutions.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-records", type=int, default=0, help="Maximum records to write; 0 means no limit.")
    parser.add_argument(
        "--failure-reasons",
        nargs="*",
        default=["REPAIR_TEST_FAILED", "COMPILE_FAILED", "REPAIR_TEST_TIMEOUT", "HELD_OUT_TEST_FAILED"],
    )
    args = parser.parse_args()

    report_paths = [Path(item) for item in args.reports]
    bundles = {bundle.spec.id: bundle for bundle in load_problems(args.problems)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, stats = build_dpo_records(
        report_paths,
        bundles,
        max_records=args.max_records,
        failure_reasons=set(args.failure_reasons),
    )
    write_jsonl(out_dir / "dpo.jsonl", records)
    manifest = {
        "stage": "build_dpo_from_failures",
        "reports": [str(path) for path in report_paths],
        "problems": args.problems,
        **stats,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_dpo_records(
    report_paths: list[Path],
    bundles: dict[str, ProblemBundle],
    max_records: int = 0,
    failure_reasons: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    allowed_reasons = failure_reasons or set()
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    stats = {
        "scanned_failures": 0,
        "written": 0,
        "skipped_no_problem": 0,
        "skipped_no_verified_cpp": 0,
        "skipped_no_failed_code": 0,
        "skipped_duplicate": 0,
        "skipped_same_as_chosen": 0,
        "skipped_failure_reason": 0,
    }
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for problem in payload.get("problems", []):
            if not isinstance(problem, dict) or problem.get("status") == "SOLVED":
                continue
            reason = str(problem.get("failure_reason") or "")
            if allowed_reasons and reason not in allowed_reasons:
                stats["skipped_failure_reason"] += 1
                continue
            stats["scanned_failures"] += 1
            problem_id = str(problem.get("problem_id") or "")
            bundle = bundles.get(problem_id)
            if bundle is None:
                stats["skipped_no_problem"] += 1
                continue
            chosen = _format_answer(bundle)
            chosen_code = _cpp_solution(bundle)
            if not chosen_code:
                stats["skipped_no_verified_cpp"] += 1
                continue
            rejected_code = _failed_code(problem)
            if not rejected_code:
                stats["skipped_no_failed_code"] += 1
                continue
            if normalize_code(rejected_code) == normalize_code(chosen_code):
                stats["skipped_same_as_chosen"] += 1
                continue
            key = (problem_id, normalize_code(rejected_code))
            if key in seen:
                stats["skipped_duplicate"] += 1
                continue
            seen.add(key)
            records.append(
                {
                    "prompt": bundle.spec.prompt(),
                    "chosen": chosen,
                    "rejected": _format_rejected(problem, rejected_code),
                    "problem_id": problem_id,
                    "failure_reason": reason,
                    "diagnostic_summary": str(problem.get("diagnostic_summary") or ""),
                }
            )
            stats["written"] += 1
            if max_records and len(records) >= max_records:
                return records, stats
    return records, stats


def _failed_code(problem: dict[str, Any]) -> str:
    records = problem.get("attempt_records") or []
    for record in reversed(records):
        if isinstance(record, dict) and record.get("generated_code"):
            return str(record["generated_code"])
    return ""


def _format_rejected(problem: dict[str, Any], code: str) -> str:
    complexity = _last_complexity(problem)
    time_complexity = complexity.get("time_complexity") or "unknown"
    space_complexity = complexity.get("space_complexity") or "unknown"
    return (
        "Solution Explanation:\n"
        "The attempted solution failed validation and should not be preferred.\n\n"
        f"Time Complexity: {time_complexity}\n"
        f"Space Complexity: {space_complexity}\n"
        f"```cpp\n{code.strip()}\n```"
    )


def _last_complexity(problem: dict[str, Any]) -> dict[str, Any]:
    records = problem.get("attempt_records") or []
    for record in reversed(records):
        if isinstance(record, dict) and isinstance(record.get("complexity"), dict):
            return record["complexity"]
    return {}


def _cpp_solution(bundle: ProblemBundle) -> str:
    for solution in bundle.oracle.solutions:
        if solution.language == "cpp17" and solution.verified:
            return solution.code
    return bundle.oracle.reference_solution


def normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
