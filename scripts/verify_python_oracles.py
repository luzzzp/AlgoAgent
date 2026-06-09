from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.python_executor import PythonExecutor
from algoagent.schema import OracleSolution, ProblemBundle, load_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Python oracle solutions with repair/eval tests.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-solutions-per-problem", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Verify at most this many problem files; 0 means no limit.")
    parser.add_argument("--python", default=None)
    args = parser.parse_args()

    report = verify_python_oracles(
        Path(args.problems),
        Path(args.out_dir),
        max_solutions_per_problem=args.max_solutions_per_problem,
        limit=args.limit,
        interpreter=args.python,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def verify_python_oracles(
    problems_dir: Path,
    out_dir: Path,
    max_solutions_per_problem: int = 3,
    limit: int = 0,
    interpreter: str | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    executor = PythonExecutor(interpreter=interpreter)
    stats = {
        "total": 0,
        "verified": 0,
        "no_python_solution": 0,
        "failed_python_solution": 0,
        "copied_manifests": 0,
    }
    for path in sorted(problems_dir.glob("*.json")):
        if path.name.startswith("_"):
            shutil.copy2(path, out_dir / path.name)
            stats["copied_manifests"] += 1
            continue
        if limit and stats["total"] >= limit:
            continue
        stats["total"] += 1
        bundle = load_problem(path)
        updated, ok = verify_bundle(bundle, executor, max_solutions_per_problem)
        if ok:
            stats["verified"] += 1
        elif not any(solution.language == "python3" for solution in bundle.oracle.solutions):
            stats["no_python_solution"] += 1
        else:
            stats["failed_python_solution"] += 1
        (out_dir / path.name).write_text(json.dumps(bundle_to_json(updated), indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {"stage": "verify_python_oracles", **stats}
    (out_dir / "_python_oracle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def verify_bundle(
    bundle: ProblemBundle,
    executor: PythonExecutor,
    max_solutions_per_problem: int = 3,
) -> tuple[ProblemBundle, bool]:
    tests = [*bundle.tests.repair_tests, *bundle.tests.eval_tests]
    solutions: list[OracleSolution] = []
    verified = False
    checked = 0
    for solution in bundle.oracle.solutions:
        if solution.language != "python3" or checked >= max_solutions_per_problem or verified:
            solutions.append(solution)
            continue
        checked += 1
        result = executor.evaluate(solution.code, tests, default_timeout_sec=bundle.spec.time_limit_sec, suite_name="oracle")
        is_verified = result.all_passed
        verified = verified or is_verified
        solutions.append(OracleSolution(language=solution.language, code=solution.code, verified=is_verified))

    updated = ProblemBundle(
        spec=bundle.spec,
        tests=bundle.tests,
        oracle=type(bundle.oracle)(
            difficulty=bundle.oracle.difficulty,
            tags=bundle.oracle.tags,
            expected_complexity=bundle.oracle.expected_complexity,
            reference_solution=bundle.oracle.reference_solution,
            solutions=solutions,
            source=bundle.oracle.source,
            url=bundle.oracle.url,
        ),
    )
    return updated, verified


def bundle_to_json(bundle: ProblemBundle) -> dict[str, object]:
    payload = {
        "problem": asdict(bundle.spec),
        "tests": {
            "repair_tests": [asdict(test) for test in bundle.tests.repair_tests],
            "eval_tests": [asdict(test) for test in bundle.tests.eval_tests],
        },
        "oracle": asdict(bundle.oracle),
    }
    return payload


if __name__ == "__main__":
    main()
