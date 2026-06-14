from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.executor import PythonExecutor
from algoagent.schema import (
    ExecutionReport,
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    load_problems,
    problem_bundle_to_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Python oracle solutions against all tests.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-solutions-per-problem", type=int, default=3)
    args = parser.parse_args()

    executor = PythonExecutor()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = verified = failed = no_solution = 0
    failed_records = []
    for bundle in load_problems(args.problems):
        total += 1
        updated, ok, has_solution, failure_detail = verify_bundle(bundle, executor, args.max_solutions_per_problem)
        no_solution += 0 if has_solution else 1
        verified += 1 if ok else 0
        failed += 1 if has_solution and not ok else 0
        if failure_detail:
            failed_records.append(failure_detail)
        (out_dir / f"{updated.spec.id}.json").write_text(
            json.dumps(problem_bundle_to_dict(updated), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    report = {
        "stage": "verify_python_oracles",
        "total": total,
        "verified": verified,
        "failed": failed,
        "no_solution": no_solution,
        "failed_log": "_failed.jsonl" if failed_records else "",
    }
    (out_dir / "_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed_records:
        with (out_dir / "_failed.jsonl").open("w", encoding="utf-8") as handle:
            for record in failed_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


def verify_bundle(
    bundle: ProblemBundle,
    executor: PythonExecutor,
    max_solutions: int,
) -> tuple[ProblemBundle, bool, bool, dict | None]:
    tests = [*bundle.tests.visible_tests, *bundle.tests.reward_tests, *bundle.tests.eval_tests]
    solutions = [solution for solution in bundle.oracle.solutions if solution.language == "python3"]
    has_solution = bool(solutions)
    new_solutions: list[OracleSolution] = []
    ok = False
    reference = bundle.oracle.reference_solution
    attempts = []
    for solution in solutions[:max_solutions]:
        result = executor.evaluate(solution.code, tests, default_timeout_sec=bundle.spec.time_limit_sec, suite_name="oracle")
        is_verified = result.all_passed
        ok = ok or is_verified
        if is_verified and not reference:
            reference = solution.code
        new_solutions.append(OracleSolution(solution.language, solution.code, is_verified))
        attempts.append(_attempt_summary(len(attempts), result))
    new_solutions.extend(solutions[max_solutions:])
    oracle = OracleMetadata(
        difficulty=bundle.oracle.difficulty,
        tags=bundle.oracle.tags,
        expected_complexity=bundle.oracle.expected_complexity,
        reference_solution=reference,
        source=bundle.oracle.source,
        url=bundle.oracle.url,
        solutions=new_solutions,
    )
    failure_detail = None
    if has_solution and not ok:
        failure_detail = {
            "problem_id": bundle.spec.id,
            "title": bundle.spec.title,
            "time_limit_sec": bundle.spec.time_limit_sec,
            "num_tests": len(tests),
            "attempts": attempts,
        }
    return ProblemBundle(spec=bundle.spec, tests=bundle.tests, oracle=oracle), ok, has_solution, failure_detail


def _attempt_summary(index: int, result: ExecutionReport) -> dict:
    first_failed = next((run for run in result.runs if not run.passed), None)
    return {
        "solution_index": index,
        "syntax_valid": result.syntax_valid,
        "syntax_error": _preview(result.syntax_error),
        "pass_rate": result.pass_rate,
        "first_failed": (
            {
                "test_id": first_failed.test_id,
                "suite": first_failed.suite,
                "timed_out": first_failed.timed_out,
                "returncode": first_failed.returncode,
                "expected": _preview(first_failed.expected),
                "actual": _preview(first_failed.actual),
                "stderr": _preview(first_failed.stderr),
            }
            if first_failed
            else None
        ),
    }


def _preview(text: str, limit: int = 500) -> str:
    return (text or "").replace("\n", "\\n")[:limit]


if __name__ == "__main__":
    main()
