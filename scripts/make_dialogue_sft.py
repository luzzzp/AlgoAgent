from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problems


FOLLOWUP_INSTRUCTION = (
    "Answer the user's follow-up question in Chinese. "
    "Only use the provided problem, verified solution, line-numbered code, and complexity context. "
    "Do not invent a new algorithm or modify code unless the user explicitly asks to restart solving."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build follow-up dialogue SFT data for AlgoAgent.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-questions-per-problem", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    records = []
    for bundle in bundles:
        if not bundle.oracle.best_solution("python3"):
            continue
        records.extend(_records(bundle, args.max_questions_per_problem))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "followup_sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "make_dialogue_sft", "records": len(records)}, indent=2))


def _records(bundle: ProblemBundle, max_questions: int) -> list[dict[str, str]]:
    code = bundle.oracle.best_solution("python3")
    if not code:
        return []
    context = _context(bundle, code)
    line_no, line_text = _representative_line(code)
    questions = [
        (
            f"第 {line_no} 行代码是什么意思？",
            (
                f"第 {line_no} 行是 `{line_text.strip()}`。"
                "它需要结合上下文理解：这一行通常承担读取数据、更新状态、调用函数或输出结果的作用。"
                "在解释代码时，应先说明这一行直接做了什么，再说明它为什么服务于整体算法。"
            ),
        ),
        (
            "这份代码的核心解题思路是什么？",
            (
                "这份代码先按照题目要求读取输入或接收函数参数，然后围绕题目约束维护必要的变量、数组或状态，"
                "最后得到并返回/输出答案。因为这份代码已经通过测试验证，解释时应重点说明它如何处理一般输入，"
                "而不是只复述样例。"
            ),
        ),
        (
            "为什么这个复杂度可以接受？",
            (
                f"当前记录的复杂度信息是：{bundle.oracle.expected_complexity or 'unknown'}。"
                "判断复杂度是否可接受时，需要把主循环次数、排序/搜索/动态规划状态数与题目的输入规模和时间限制对应起来。"
                "如果复杂度字段为 unknown，应谨慎说明需要进一步根据代码主结构估算，而不能随意编造。"
            ),
        ),
        (
            "这道题需要注意哪些边界情况？",
            (
                "应重点检查最小输入、最大输入、重复值、空结构、单元素结构、输出格式和不可行情况。"
                "如果是函数式任务，还要确认返回值类型与题目要求一致；如果是标准输入输出任务，还要确认换行和空格格式。"
            ),
        ),
        (
            "如果用户说样例过了但隐藏测试失败，应该怎么分析？",
            (
                "应先避免直接改成只匹配样例的代码，而是检查算法假设是否覆盖所有输入。"
                "常见方向包括边界条件、数据范围导致的超时、整数/浮点精度、排序稳定性、重复元素、空输入和输出格式。"
                "修复后需要重新运行可见测试，并用内部测试做不泄漏用例的验证。"
            ),
        ),
    ]
    return [
        {
            "instruction": FOLLOWUP_INSTRUCTION,
            "input": f"{context}\n\nUser follow-up:\n{question}",
            "output": answer,
        }
        for question, answer in questions[:max_questions]
    ]


def _context(bundle: ProblemBundle, code: str) -> str:
    return (
        f"Title: {bundle.spec.title}\n\n"
        f"Problem statement:\n{bundle.spec.statement}\n\n"
        f"IO mode: {bundle.spec.io_mode}\n"
        f"Entry point: {bundle.spec.entry_point or 'stdin/stdout'}\n"
        f"Complexity: {bundle.oracle.expected_complexity or 'unknown'}\n\n"
        f"Verified solution with line numbers:\n{_numbered(code)}"
    )


def _numbered(code: str) -> str:
    return "\n".join(f"{idx}: {line}" for idx, line in enumerate(code.splitlines(), start=1))


def _representative_line(code: str) -> tuple[int, str]:
    for idx, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return idx, line
    return 1, code.splitlines()[0] if code.splitlines() else ""


if __name__ == "__main__":
    main()
