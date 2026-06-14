from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problems


FOLLOWUP_SYSTEM = "你是 AlgoAgent-Py。只能基于已验证题解回答用户追问，不要擅自修改代码。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic follow-up dialogue SFT data.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-questions-per-problem", type=int, default=4)
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
    with (out_dir / "dialogue_sft.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "make_dialogue_sft", "records": len(records)}, indent=2))


def _records(bundle: ProblemBundle, max_questions: int) -> list[dict]:
    code = bundle.oracle.best_solution("python3")
    numbered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(code.splitlines(), start=1))
    context = (
        f"题目：{bundle.spec.title}\n{bundle.spec.statement}\n\n"
        f"已验证代码：\n{numbered}\n\n"
        f"复杂度：{bundle.oracle.expected_complexity or 'unknown'}\n"
    )
    qa = [
        ("这段代码的核心思路是什么？", "这段代码围绕题目要求读取输入、执行核心计算并输出答案。具体细节需要结合代码中的变量更新和循环结构理解。"),
        ("为什么这个解法不会只通过样例？", "该解法来自已通过测试验证的 oracle，回答时应关注它对一般输入的处理，而不是只匹配题目样例。"),
        ("第二行代码是什么意思？", "第 2 行代码需要结合带行号代码查看，它通常用于导入模块、读取输入或初始化变量。"),
        ("时间复杂度为什么这样估计？", "时间复杂度应根据主循环、排序、搜索或动态规划状态数量来估计，不能只看样例规模。"),
        ("边界情况应该注意什么？", "需要关注最小输入、最大输入、重复值、空结构和输出格式等边界情况。"),
    ][:max_questions]
    return [
        {
            "messages": [
                {"role": "system", "content": FOLLOWUP_SYSTEM},
                {"role": "user", "content": context + f"\n用户追问：{question}"},
                {"role": "assistant", "content": answer},
            ]
        }
        for question, answer in qa
    ]


if __name__ == "__main__":
    main()

