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
        "instruction": (
            "Solve the algorithm problem using C++17 within the stated resource limits. "
            "The Solution Explanation section must be written in Chinese. "
            "Use this exact response structure:\n"
            "Solution Explanation:\n"
            "...\n"
            "Time Complexity: O(...)\n"
            "Space Complexity: O(...)\n"
            "```cpp\n...\n```"
        ),
        "input": bundle.spec.prompt(),
        "output": _format_answer(bundle),
    }


def _dpo_record(bundle: ProblemBundle) -> dict[str, str]:
    return {
        "prompt": bundle.spec.prompt(),
        "chosen": _format_answer(bundle),
        "rejected": (
            "Solution Explanation:\n"
            "\u8be5\u5019\u9009\u4ee3\u7801\u672a\u80fd\u7a33\u5b9a\u901a\u8fc7\u9a8c\u8bc1\uff0c\u4e0d\u5e94\u4f5c\u4e3a\u4f18\u5148\u7b54\u6848\u3002\n"
            "Time Complexity: unknown\n"
            "Space Complexity: unknown\n"
            "```cpp\n#include <bits/stdc++.h>\nusing namespace std;\nint main(){return 0;}\n```"
        ),
    }


def _grpo_record(bundle: ProblemBundle) -> dict[str, str]:
    return {
        "problem_id": bundle.spec.id,
        "prompt": bundle.spec.prompt(),
        "reward_contract": "resource feasibility + compile + repair tests + held-out tests",
    }


def _format_answer(bundle: ProblemBundle) -> str:
    time_complexity, space_complexity = _complexity_fields(bundle)
    return (
        f"Solution Explanation:\n{_solution_explanation(bundle)}\n\n"
        f"Time Complexity: {time_complexity}\n"
        f"Space Complexity: {space_complexity}\n"
        f"```cpp\n{_cpp_solution(bundle)}\n```"
    )


def _solution_explanation(bundle: ProblemBundle) -> str:
    tags = ", ".join(bundle.oracle.tags)
    if tags:
        return (
            f"\u6839\u636e\u9898\u610f\u5206\u6790\u8f93\u5165\u89c4\u6a21\u548c\u8f93\u51fa\u8981\u6c42\uff0c"
            f"\u7ed3\u5408 {tags} \u7b49\u7b97\u6cd5\u601d\u60f3\u8bbe\u8ba1\u89e3\u6cd5\u3002"
            "\u5b9e\u73b0\u65f6\u9700\u8981\u6309\u7167\u9898\u76ee\u7ed9\u5b9a\u7684\u8f93\u5165\u683c\u5f0f\u8bfb\u53d6\u6570\u636e\uff0c"
            "\u5e76\u4e25\u683c\u8f93\u51fa\u8981\u6c42\u7684\u7ed3\u679c\u3002"
        )
    return (
        "\u6839\u636e\u9898\u610f\u63a8\u5bfc\u9700\u8981\u8ba1\u7b97\u7684\u76ee\u6807\uff0c"
        "\u6309\u7167\u7ea6\u675f\u9009\u62e9\u80fd\u5728\u65f6\u95f4\u548c\u7a7a\u95f4\u9650\u5236\u5185\u901a\u8fc7\u7684\u65b9\u6cd5\u3002"
        "\u5b9e\u73b0\u65f6\u9700\u8981\u51c6\u786e\u5904\u7406\u8f93\u5165\u8f93\u51fa\u683c\u5f0f\u548c\u8fb9\u754c\u60c5\u51b5\u3002"
    )


def _complexity_fields(bundle: ProblemBundle) -> tuple[str, str]:
    text = bundle.oracle.expected_complexity.strip()
    time_complexity = _labeled_complexity(text, "time")
    space_complexity = _labeled_complexity(text, "space")
    complexities = re.findall(r"O\s*\([^)]+\)", text, flags=re.I)
    if time_complexity == "unknown" and complexities:
        time_complexity = complexities[0]
    if space_complexity == "unknown" and len(complexities) >= 2:
        space_complexity = complexities[1]
    return time_complexity, space_complexity


def _labeled_complexity(text: str, label: str) -> str:
    normalized = text.replace("\uff1a", ":")
    match = re.search(rf"{label}\s*:\s*(O\s*\([^)]+\)|unknown)", normalized, flags=re.I)
    return match.group(1) if match else "unknown"


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
