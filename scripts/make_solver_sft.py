from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Solver SFT data for AlgoAgent-Py.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    records = [_record(bundle) for bundle in bundles if bundle.oracle.best_solution("python3")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "solver_sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "make_solver_sft", "records": len(records)}, indent=2))


def _record(bundle: ProblemBundle) -> dict[str, str]:
    return {
        "instruction": (
            "Solve the algorithm problem in Python 3. "
            "The Solution Explanation must be Chinese. "
            "Use exact sections: Solution Explanation, Time Complexity, Space Complexity, and a python code block."
        ),
        "input": bundle.spec.prompt(bundle.tests.visible_tests),
        "output": _answer(bundle),
    }


def _answer(bundle: ProblemBundle) -> str:
    return (
        f"Solution Explanation:\n{_explanation(bundle)}\n\n"
        f"Time Complexity: {_time_complexity(bundle)}\n"
        f"Space Complexity: {_space_complexity(bundle)}\n"
        f"```python\n{bundle.oracle.best_solution('python3').strip()}\n```"
    )


def _explanation(bundle: ProblemBundle) -> str:
    return (
        "根据题意分析输入规模、输出要求和边界条件，选择能够在时间限制内通过的算法。"
        "实现时按照题目给定的输入格式读取数据，并严格输出要求的结果。"
    )


def _time_complexity(bundle: ProblemBundle) -> str:
    complexities = re.findall(r"O\s*\([^)]+\)", bundle.oracle.expected_complexity, flags=re.I)
    return complexities[0] if complexities else "unknown"


def _space_complexity(bundle: ProblemBundle) -> str:
    complexities = re.findall(r"O\s*\([^)]+\)", bundle.oracle.expected_complexity, flags=re.I)
    return complexities[1] if len(complexities) > 1 else "unknown"


if __name__ == "__main__":
    main()

