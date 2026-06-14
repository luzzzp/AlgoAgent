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
    parser = argparse.ArgumentParser(description="Build Solver SFT data for AlgoAgent.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--annotations", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    annotations = _load_annotations(args.annotations) if args.annotations else {}
    records = [
        _record(bundle, annotations.get(bundle.spec.id))
        for bundle in bundles
        if bundle.oracle.best_solution("python3")
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "solver_sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "make_solver_sft", "records": len(records)}, indent=2))


def _record(bundle: ProblemBundle, annotation: dict | None = None) -> dict[str, str]:
    return {
        "instruction": (
            "Solve the algorithm problem in Python 3. "
            "The Solution Explanation must be Chinese. "
            "Use exact sections: Solution Explanation, Time Complexity, Space Complexity, and a python code block."
        ),
        "input": bundle.spec.prompt(bundle.tests.visible_tests),
        "output": _answer(bundle, annotation),
    }


def _answer(bundle: ProblemBundle, annotation: dict | None = None) -> str:
    return (
        f"Solution Explanation:\n{_explanation(bundle, annotation)}\n\n"
        f"Time Complexity: {_time_complexity(bundle, annotation)}\n"
        f"Space Complexity: {_space_complexity(bundle, annotation)}\n"
        f"```python\n{bundle.oracle.best_solution('python3').strip()}\n```"
    )


def _explanation(bundle: ProblemBundle, annotation: dict | None = None) -> str:
    if annotation and annotation.get("solution_explanation"):
        return str(annotation["solution_explanation"]).strip()
    if bundle.spec.io_mode == "callable" and bundle.spec.entry_point:
        io_sentence = (
            f"本题是函数式任务，需要实现 `{bundle.spec.entry_point}` 函数，"
            "根据传入参数计算并直接返回结果，不需要额外从标准输入读取。"
        )
    else:
        io_sentence = "本题是标准输入输出任务，需要按照题面格式从 stdin 读取数据，并将答案输出到 stdout。"

    time_complexity = _time_complexity(bundle)
    space_complexity = _space_complexity(bundle)
    complexity_parts = []
    if time_complexity != "unknown":
        complexity_parts.append(f"时间复杂度目标约为 {time_complexity}")
    if space_complexity != "unknown":
        complexity_parts.append(f"空间复杂度目标约为 {space_complexity}")
    if complexity_parts:
        complexity_sentence = "，".join(complexity_parts) + "。"
    else:
        complexity_sentence = "实现时需要根据题目约束选择能在时间限制内通过的算法。"

    return (
        f"{io_sentence}"
        f"{complexity_sentence}"
        "下面的参考实现已通过该题的可见测试、奖励测试和留出测试；"
        "本条样本用于训练模型稳定输出中文题解、复杂度字段和可执行 Python 代码。"
    )


def _time_complexity(bundle: ProblemBundle, annotation: dict | None = None) -> str:
    if annotation and annotation.get("time_complexity"):
        return str(annotation["time_complexity"]).strip()
    return _complexity_by_label(bundle.oracle.expected_complexity, "time")


def _space_complexity(bundle: ProblemBundle, annotation: dict | None = None) -> str:
    if annotation and annotation.get("space_complexity"):
        return str(annotation["space_complexity"]).strip()
    return _complexity_by_label(bundle.oracle.expected_complexity, "space")


def _complexity_by_label(text: str, label: str) -> str:
    if not text:
        return "unknown"
    pattern = rf"{label}\s*:\s*([^;]+)"
    labelled = re.search(pattern, text, flags=re.I)
    if labelled:
        complexity = _extract_o_notation(labelled.group(1))
        return complexity or labelled.group(1).strip()
    complexity = _extract_o_notation(text)
    return complexity or "unknown"


def _extract_o_notation(text: str) -> str:
    match = re.search(r"O\s*\([^)]+\)", text, flags=re.I)
    return match.group(0).strip() if match else ""


def _load_annotations(path: str) -> dict[str, dict]:
    annotations = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record.get("problem_id")
            if problem_id:
                annotations[str(problem_id)] = record
    return annotations


if __name__ == "__main__":
    main()
