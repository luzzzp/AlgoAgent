from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AlgoAgent evaluation failures.")
    parser.add_argument("--reports", nargs="+", required=True, help="One or more evaluate_hf_model JSON reports.")
    parser.add_argument("--out-md", default="reports/failure_analysis.md")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--max-examples-per-category", type=int, default=5)
    args = parser.parse_args()

    analysis = analyze_reports([Path(item) for item in args.reports], args.max_examples_per_category)
    md = render_markdown(analysis)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def analyze_reports(paths: list[Path], max_examples_per_category: int = 5) -> dict[str, Any]:
    report_summaries: list[dict[str, Any]] = []
    failure_reason_counts: Counter[str] = Counter()
    diagnostic_category_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_problems = 0
    solved = 0

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        problems = [item for item in payload.get("problems", []) if isinstance(item, dict)]
        summary = payload.get("summary", {})
        report_summaries.append(
            {
                "path": str(path),
                "num_problems": len(problems),
                "summary": summary,
            }
        )
        total_problems += len(problems)
        for problem in problems:
            if problem.get("status") == "SOLVED":
                solved += 1
                continue
            failure_reason = str(problem.get("failure_reason") or "UNKNOWN_FAILURE")
            category = classify_failure(problem)
            failure_reason_counts[failure_reason] += 1
            diagnostic_category_counts[category] += 1
            if len(examples[category]) < max_examples_per_category:
                examples[category].append(example_from_problem(path, problem, category))

    failed = total_problems - solved
    return {
        "reports": report_summaries,
        "total_problems": total_problems,
        "solved": solved,
        "failed": failed,
        "verified_success_rate": solved / total_problems if total_problems else 0.0,
        "failure_reason_counts": dict(failure_reason_counts.most_common()),
        "diagnostic_category_counts": dict(diagnostic_category_counts.most_common()),
        "examples": dict(examples),
    }


def classify_failure(problem: dict[str, Any]) -> str:
    reason = str(problem.get("failure_reason") or "")
    if reason in {"THEORETICAL_TLE", "THEORETICAL_MLE"}:
        return reason
    final_repair = final_repair_candidate(problem)
    if final_repair is None:
        return reason or "NO_CANDIDATE"
    if not final_repair.get("compiled"):
        return "COMPILE_ERROR"

    failed_tests = failed_tests_from_candidate(final_repair)
    if any(test.get("timed_out") for test in failed_tests):
        return "TIMEOUT"
    if reason == "HELD_OUT_TEST_FAILED":
        return "HELD_OUT_FAILED"
    if not failed_tests:
        held_out = problem.get("held_out_result")
        if isinstance(held_out, dict):
            held_out_failures = failed_tests_from_candidate(held_out)
            if any(test.get("timed_out") for test in held_out_failures):
                return "HELD_OUT_TIMEOUT"
            if held_out_failures:
                return "HELD_OUT_FAILED"
        return reason or "UNKNOWN_FAILURE"

    first = failed_tests[0]
    expected = str(first.get("expected") or "")
    actual = str(first.get("actual") or "")
    if not actual and expected:
        return "EMPTY_OUTPUT"
    if actual and not expected:
        return "EXTRA_OUTPUT"
    if actual.split() == expected.split() and actual != expected:
        return "OUTPUT_FORMAT"
    return "WRONG_ANSWER"


def example_from_problem(path: Path, problem: dict[str, Any], category: str) -> dict[str, Any]:
    final_repair = final_repair_candidate(problem)
    first_failed = {}
    if isinstance(final_repair, dict):
        failures = failed_tests_from_candidate(final_repair)
        if failures:
            first_failed = trim_test(failures[0])
    return {
        "report": str(path),
        "problem_id": problem.get("problem_id"),
        "status": problem.get("status"),
        "failure_reason": problem.get("failure_reason"),
        "category": category,
        "attempts": problem.get("attempts"),
        "diagnostic_summary": trim_text(str(problem.get("diagnostic_summary") or ""), 500),
        "first_failed_test": first_failed,
        "last_trace": last_trace(problem),
    }


def final_repair_candidate(problem: dict[str, Any]) -> dict[str, Any] | None:
    records = problem.get("attempt_records") or []
    for record in reversed(records):
        if isinstance(record, dict) and isinstance(record.get("repair_result"), dict):
            return record["repair_result"]
    return None


def failed_tests_from_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    tests = candidate.get("tests")
    if not isinstance(tests, list):
        return []
    return [test for test in tests if isinstance(test, dict) and not test.get("passed")]


def trim_test(test: dict[str, Any]) -> dict[str, Any]:
    return {
        "test_id": test.get("test_id"),
        "suite": test.get("suite"),
        "timed_out": test.get("timed_out"),
        "expected": trim_text(str(test.get("expected") or ""), 300),
        "actual": trim_text(str(test.get("actual") or ""), 300),
        "stderr": trim_text(str(test.get("stderr") or ""), 300),
    }


def last_trace(problem: dict[str, Any]) -> dict[str, Any]:
    traces = problem.get("traces") or []
    for trace in reversed(traces):
        if isinstance(trace, dict):
            return {
                "attempt": trace.get("attempt"),
                "stage": trace.get("stage"),
                "message": trim_text(str(trace.get("message") or ""), 500),
            }
    return {}


def trim_text(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def render_markdown(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# AlgoAgent Failure Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Reports: {len(analysis['reports'])}")
    lines.append(f"- Total problems: {analysis['total_problems']}")
    lines.append(f"- Solved: {analysis['solved']}")
    lines.append(f"- Failed: {analysis['failed']}")
    lines.append(f"- Verified success rate: {analysis['verified_success_rate']:.3f}")
    lines.append("")
    lines.append("## Report Summaries")
    lines.append("")
    lines.append("| Report | Problems | Verified Success | Compile | Repair Pass | Held-out Pass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for report in analysis["reports"]:
        summary = report.get("summary") or {}
        lines.append(
            "| "
            f"{Path(report['path']).name} | "
            f"{report['num_problems']} | "
            f"{float(summary.get('verified_success_rate') or 0):.3f} | "
            f"{float(summary.get('final_compile_rate') or 0):.3f} | "
            f"{float(summary.get('repair_test_pass_rate') or 0):.3f} | "
            f"{float(summary.get('held_out_test_pass_rate') or 0):.3f} |"
        )
    lines.append("")
    lines.append("## Failure Reasons")
    lines.append("")
    lines.extend(counter_table(analysis["failure_reason_counts"], "Reason"))
    lines.append("")
    lines.append("## Diagnostic Categories")
    lines.append("")
    lines.extend(counter_table(analysis["diagnostic_category_counts"], "Category"))
    lines.append("")
    lines.append("## Representative Examples")
    lines.append("")
    for category, items in analysis["examples"].items():
        lines.append(f"### {category}")
        lines.append("")
        for item in items:
            lines.append(f"- Problem: `{item.get('problem_id')}`")
            lines.append(f"  - Report: `{Path(str(item.get('report'))).name}`")
            lines.append(f"  - Failure reason: `{item.get('failure_reason')}`")
            lines.append(f"  - Attempts: `{item.get('attempts')}`")
            failed = item.get("first_failed_test") or {}
            if failed:
                lines.append(f"  - First failed test: `{failed.get('test_id')}` timed_out={failed.get('timed_out')}")
                lines.append(f"  - Expected: `{single_line(str(failed.get('expected') or ''))}`")
                lines.append(f"  - Actual: `{single_line(str(failed.get('actual') or ''))}`")
                stderr = single_line(str(failed.get("stderr") or ""))
                if stderr:
                    lines.append(f"  - Stderr: `{stderr}`")
            trace = item.get("last_trace") or {}
            if trace:
                lines.append(f"  - Last trace: `{trace.get('stage')}` - {single_line(str(trace.get('message') or ''))}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def counter_table(counter: dict[str, int], label: str) -> list[str]:
    lines = [f"| {label} | Count |", "|---|---:|"]
    if not counter:
        lines.append("| none | 0 |")
        return lines
    for key, value in counter.items():
        lines.append(f"| {key} | {value} |")
    return lines


def single_line(text: str) -> str:
    return trim_text(" ".join(text.split()), 160).replace("`", "'")


if __name__ == "__main__":
    main()
