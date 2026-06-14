from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import (
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    ProblemSpec,
    TestCase,
    TestSuite,
    problem_bundle_to_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TACO-verified style rows to AlgoAgent-Py JSON.")
    parser.add_argument("--dataset", default="likaixin/TACO-verified")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-visible-tests", type=int, default=2)
    parser.add_argument("--max-reward-tests", type=int, default=20)
    parser.add_argument("--max-eval-tests", type=int, default=10)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to use this converter.") from exc

    dataset = load_dataset(args.dataset, split=args.split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = dataset.select(range(args.offset, min(len(dataset), args.offset + args.limit))) if args.limit else dataset
    written = 0
    skipped = 0
    skipped_records = []
    for idx, row in enumerate(rows, start=args.offset):
        try:
            bundle = convert_row(row, idx, args.max_visible_tests, args.max_reward_tests, args.max_eval_tests)
        except ValueError as exc:
            skipped += 1
            skipped_records.append(_skipped_record(row, idx, str(exc)))
            continue
        path = out_dir / f"{bundle.spec.id}.json"
        path.write_text(json.dumps(problem_bundle_to_dict(bundle), indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
    manifest = {
        "stage": "convert_taco_python",
        "dataset": args.dataset,
        "split": args.split,
        "written": written,
        "skipped": skipped,
        "skipped_log": "_skipped.jsonl" if skipped_records else "",
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if skipped_records:
        with (out_dir / "_skipped.jsonl").open("w", encoding="utf-8") as handle:
            for record in skipped_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2))


def convert_row(
    row: dict[str, Any],
    index: int,
    max_visible: int,
    max_reward: int,
    max_eval: int,
) -> ProblemBundle:
    title = str(row.get("title") or row.get("name") or f"taco_problem_{index}")
    statement = str(row.get("statement") or row.get("question") or row.get("description") or "")
    if not statement:
        raise ValueError("missing_statement")
    inputs, outputs = _extract_tests(row)
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("missing_or_mismatched_tests")
    cases = [TestCase(stdin=i, expected_stdout=o) for i, o in zip(inputs, outputs)]
    visible = cases[:max_visible]
    reward = cases[max_visible : max_visible + max_reward]
    eval_cases = cases[max_visible + max_reward : max_visible + max_reward + max_eval]
    if not reward and len(cases) > len(visible):
        reward = cases[len(visible) :]
    if not eval_cases and reward:
        eval_cases = reward[-min(len(reward), max_eval) :]
        reward = reward[: max(0, len(reward) - len(eval_cases))]
    solutions = [
        OracleSolution(language="python3", code=code, verified=False)
        for code in _extract_python_solutions(row)
    ]
    spec = ProblemSpec(
        id=_safe_id(index, title),
        title=title,
        statement=statement,
        input_format=str(row.get("input_format") or ""),
        output_format=str(row.get("output_format") or ""),
        constraints=_as_list(row.get("constraints")),
        language="python3",
        time_limit_sec=float(row.get("time_limit_sec") or row.get("time_limit") or 2.0),
        memory_limit_mb=int(row.get("memory_limit_mb") or row.get("memory_limit") or 256),
    )
    return ProblemBundle(
        spec=spec,
        tests=TestSuite(visible_tests=visible, reward_tests=reward, eval_tests=eval_cases),
        oracle=OracleMetadata(
            difficulty=str(row.get("difficulty") or ""),
            tags=_as_list(row.get("tags")),
            expected_complexity=str(row.get("expected_complexity") or ""),
            solutions=solutions,
            source="TACO-verified",
            url=str(row.get("url") or ""),
        ),
    )


def _extract_tests(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    io = row.get("input_output")
    if isinstance(io, str):
        try:
            io = json.loads(io)
        except json.JSONDecodeError:
            io = {}
    if isinstance(io, dict):
        return _as_list(io.get("inputs")), _as_list(io.get("outputs"))
    return _as_list(row.get("inputs")), _as_list(row.get("outputs"))


def _extract_python_solutions(row: dict[str, Any]) -> list[str]:
    candidates = row.get("solutions") or row.get("python_solutions") or row.get("code")
    if isinstance(candidates, str):
        try:
            decoded = json.loads(candidates)
            candidates = decoded
        except json.JSONDecodeError:
            candidates = [candidates]
    if isinstance(candidates, list):
        return [str(item) for item in candidates if "def " in str(item) or "input" in str(item)]
    return []


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(item) for item in decoded]
        except json.JSONDecodeError:
            pass
        return [value]
    return [str(value)]


def _safe_id(index: int, title: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"taco_{index}_{slug[:80] or 'problem'}"


def _skipped_record(row: dict[str, Any], index: int, reason: str) -> dict[str, Any]:
    title = row.get("title") or row.get("name") or f"taco_problem_{index}"
    statement = row.get("statement") or row.get("question") or row.get("description") or ""
    input_output = row.get("input_output", "")
    return {
        "index": index,
        "reason": reason,
        "title": str(title),
        "keys": sorted(str(key) for key in row.keys()),
        "statement_preview": _preview(statement),
        "input_output_preview": _preview(input_output),
        "solutions_preview": _preview(row.get("solutions") or row.get("python_solutions") or row.get("code") or ""),
    }


def _preview(value: Any, limit: int = 500) -> str:
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except TypeError:
            value = str(value)
    value = value.replace("\n", "\\n")
    return value[:limit]


if __name__ == "__main__":
    main()
