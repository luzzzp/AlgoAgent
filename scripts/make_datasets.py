from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT/DPO/GRPO seed datasets.")
    parser.add_argument("--problems", default="data/problems/sample")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    trainable = [bundle for bundle in bundles if _cpp_solution(bundle)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "sft.jsonl", [_sft_record(bundle) for bundle in trainable])
    _write_jsonl(out_dir / "dpo.jsonl", [_dpo_record(bundle) for bundle in trainable])
    _write_jsonl(out_dir / "grpo_prompts.jsonl", [_grpo_record(bundle) for bundle in bundles])
    print(f"Wrote {len(trainable)} supervised records and {len(bundles)} GRPO prompts to {out_dir}")


def _sft_record(bundle: ProblemBundle) -> dict[str, str]:
    return {
        "instruction": "Solve the algorithm problem using C++17 within the stated resource limits.",
        "input": bundle.spec.prompt(),
        "output": _format_answer(bundle),
    }


def _dpo_record(bundle: ProblemBundle) -> dict[str, str]:
    return {
        "prompt": bundle.spec.prompt(),
        "chosen": _format_answer(bundle),
        "rejected": (
            "Time Complexity: unknown\n"
            "Space Complexity: unknown\n"
            "Code:\n```cpp\n#include <bits/stdc++.h>\nusing namespace std;\nint main(){return 0;}\n```"
        ),
    }


def _grpo_record(bundle: ProblemBundle) -> dict[str, str]:
    return {
        "problem_id": bundle.spec.id,
        "prompt": bundle.spec.prompt(),
        "reward_contract": "resource feasibility + compile + repair tests + held-out tests",
    }


def _format_answer(bundle: ProblemBundle) -> str:
    oracle = bundle.oracle
    return (
        f"Algorithm tags: {', '.join(oracle.tags) or 'not provided'}\n"
        f"Time and space complexity: {oracle.expected_complexity}\n"
        f"Code:\n```cpp\n{_cpp_solution(bundle)}\n```"
    )


def _cpp_solution(bundle: ProblemBundle) -> str:
    for solution in bundle.oracle.solutions:
        if solution.language == "cpp17" and solution.verified:
            return solution.code
    return bundle.oracle.reference_solution


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
