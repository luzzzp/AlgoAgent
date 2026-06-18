from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problems


SYSTEM_PROMPT = (
    "你是算法题解标注员。你只基于题面和已验证 Python 代码生成中文题解与复杂度。"
    "不要修改代码，不要编造不存在的数据结构；不确定时可以写 unknown。"
)


class AnnotationModel(Protocol):
    def annotate(self, bundle: ProblemBundle, code: str) -> str:
        ...


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chinese solution annotations for Solver SFT.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["hf", "mock"], default="hf")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _loaded_ids(out_path) if args.resume else set()
    model = _build_model(args)

    written = 0
    skipped = 0
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for bundle in bundles:
            if bundle.spec.id in done:
                skipped += 1
                continue
            code = bundle.oracle.best_solution("python3")
            if not code:
                skipped += 1
                continue
            raw = model.annotate(bundle, code)
            record = _parse_annotation(bundle, raw)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1

    print(json.dumps({"stage": "generate_solution_annotations", "written": written, "skipped": skipped}, indent=2))


def _build_model(args) -> AnnotationModel:
    if args.backend == "mock":
        return MockAnnotationModel()
    return HfAnnotationModel(args.model, args.max_new_tokens, args.load_in_4bit)


class MockAnnotationModel:
    def annotate(self, bundle: ProblemBundle, code: str) -> str:
        return json.dumps(
            {
                "solution_explanation": "先根据题目输入构造必要变量，再按照参考代码中的状态转移或循环逻辑计算答案，最后输出结果。",
                "time_complexity": "O(n)",
                "space_complexity": "O(n)",
            },
            ensure_ascii=False,
        )


class HfAnnotationModel:
    def __init__(self, model_name: str, max_new_tokens: int, load_in_4bit: bool):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit("Install requirements-train.txt before using HF annotation backend.") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def annotate(self, bundle: ProblemBundle, code: str) -> str:
        prompt = _annotation_prompt(bundle, code)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def _annotation_prompt(bundle: ProblemBundle, code: str) -> str:
    visible = "\n\n".join(
        f"Input:\n{case.stdin}\nExpected:\n{case.expected_stdout}" for case in bundle.tests.visible_tests[:2]
    )
    return (
        "请为下面算法题生成 SFT 用标注。要求只输出 JSON，不要 Markdown。\n"
        "JSON 字段：solution_explanation, time_complexity, space_complexity。\n"
        "solution_explanation 必须是中文，2-5 句话，要结合题意和代码思路，不要写空泛模板。\n"
        "time_complexity / space_complexity 尽量根据代码主循环、递归、排序、DP 状态和题目约束估计，格式优先用 O(...)；"
        "如果无法可靠判断，才写 unknown。\n\n"
        f"Statement:\n{_clip(bundle.spec.statement, 3500)}\n\n"
        f"IO mode: {bundle.spec.io_mode}\n"
        f"Entry point: {bundle.spec.entry_point or 'stdin/stdout'}\n"
        f"Known complexity metadata: {bundle.oracle.expected_complexity or 'unknown'}\n\n"
        f"Visible tests:\n{_clip(visible, 1200)}\n\n"
        f"Verified Python code:\n```python\n{_clip(code, 7000)}\n```"
    )


def _parse_annotation(bundle: ProblemBundle, raw: str) -> dict:
    payload = _extract_json(raw)
    explanation = str(payload.get("solution_explanation") or "").strip()
    time_complexity = str(payload.get("time_complexity") or "").strip() or "unknown"
    space_complexity = str(payload.get("space_complexity") or "").strip() or "unknown"
    valid = bool(explanation and re.search(r"[\u4e00-\u9fff]", explanation))
    if not valid:
        explanation = _fallback_explanation(bundle)
    return {
        "problem_id": bundle.spec.id,
        "solution_explanation": explanation,
        "time_complexity": time_complexity,
        "space_complexity": space_complexity,
        "valid": valid,
        "raw_response": raw[:2000],
    }


def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _fallback_explanation(bundle: ProblemBundle) -> str:
    if bundle.spec.io_mode == "callable" and bundle.spec.entry_point:
        return f"本题需要实现 `{bundle.spec.entry_point}` 函数，根据参数计算并返回题目要求的结果。"
    return "本题需要按照题面读取输入，依据约束设计可通过时间限制的算法，并输出题目要求的结果。"


def _loaded_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("problem_id"):
                ids.add(str(payload["problem_id"]))
    return ids


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


if __name__ == "__main__":
    main()
