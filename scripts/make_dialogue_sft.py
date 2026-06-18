from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problems


FOLLOWUP_INSTRUCTION = (
    "Answer the user's follow-up question in Chinese. "
    "Only use the provided problem, verified solution, line-numbered code, and complexity context. "
    "Do not invent a new algorithm or modify code unless the user explicitly asks to restart solving."
)

FOLLOWUP_TASK_HEADER = (
    "Task Type: ANSWER_FOLLOW_UP\n"
    "Interaction: User follow-up after verified solution\n\n"
)

SYSTEM_PROMPT = (
    "\u4f60\u662f\u7b97\u6cd5\u9898\u89e3\u5bf9\u8bdd\u6570\u636e\u6807\u6ce8\u5458\u3002"
    "\u4f60\u53ea\u80fd\u57fa\u4e8e\u9898\u9762\u3001\u5df2\u9a8c\u8bc1\u4ee3\u7801\u3001"
    "\u9898\u89e3\u548c\u590d\u6742\u5ea6\u751f\u6210\u4e2d\u6587\u8ffd\u95ee\u95ee\u7b54\u3002"
    "\u4e0d\u8981\u6cc4\u9732\u5185\u90e8\u6d4b\u8bd5\u7528\u4f8b\uff0c"
    "\u4e0d\u8981\u53d1\u660e\u65b0\u7b97\u6cd5\u6216\u64c5\u81ea\u4fee\u6539\u4ee3\u7801\u3002"
)

SUGGESTED_QUESTION_TYPES = [
    "line_explanation",
    "algorithm_idea",
    "complexity",
    "edge_cases",
    "hidden_failure_analysis",
    "statement_understanding",
    "variable_meaning",
    "sample_walkthrough",
    "implementation_detail",
    "debugging_strategy",
]

DONE_IDS_FILENAME = "_done_problem_ids.jsonl"


class DialogueModel(Protocol):
    def generate_dialogue(self, bundle: ProblemBundle, annotation: dict[str, Any], code: str) -> str:
        ...


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM-generated follow-up SFT data for AlgoAgent.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--annotations", default="")
    parser.add_argument("--backend", choices=["hf", "mock"], default="hf")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-questions-per-problem", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    annotations = _load_annotations(args.annotations) if args.annotations else {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "followup_sft.jsonl"
    failed_path = out_dir / "_failed_followup.jsonl"
    done = _loaded_problem_ids(out_path) if args.resume else set()
    model = _build_model(args)

    written = 0
    failed = 0
    skipped = 0
    mode = "a" if args.resume else "w"
    failed_mode = "a" if args.resume else "w"
    done_mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as output, \
        failed_path.open(failed_mode, encoding="utf-8") as failures, \
        _done_ids_path(out_path).open(done_mode, encoding="utf-8") as done_ids:
        for bundle in bundles:
            if bundle.spec.id in done:
                skipped += 1
                continue
            code = bundle.oracle.best_solution("python3")
            if not code:
                skipped += 1
                continue
            annotation = annotations.get(bundle.spec.id, {})
            raw = model.generate_dialogue(bundle, annotation, code)
            records, failure_records = _records_from_response(
                bundle,
                annotation,
                code,
                raw,
                args.max_questions_per_problem,
            )
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            for failure_record in failure_records:
                failures.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                failed += 1
            done_ids.write(json.dumps({"problem_id": bundle.spec.id}, ensure_ascii=False) + "\n")
            output.flush()
            failures.flush()
            done_ids.flush()

    report = {
        "stage": "make_dialogue_sft",
        "records": written,
        "failed_items": failed,
        "skipped_problems": skipped,
        "failed_log": "_failed_followup.jsonl" if failed else "",
    }
    (out_dir / "_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _build_model(args) -> DialogueModel:
    if args.backend == "mock":
        return MockDialogueModel(args.max_questions_per_problem)
    return HfDialogueModel(args.model, args.max_new_tokens, args.load_in_4bit, args.max_questions_per_problem)


class MockDialogueModel:
    def __init__(self, max_questions: int = 5):
        self.max_questions = max_questions

    def generate_dialogue(self, bundle: ProblemBundle, annotation: dict[str, Any], code: str) -> str:
        evidence = _evidence_items(bundle, annotation, code)
        line_item = next(item for item in evidence if item["source"] == "code")
        items = [
            {
                "category": "line_explanation",
                "question": f"\u7b2c {line_item['line']} \u884c `{line_item['text']}` \u662f\u4ec0\u4e48\u610f\u601d\uff1f",
                "answer": (
                    f"\u7b2c {line_item['line']} \u884c\u4ee3\u7801\u662f `{line_item['text']}`\uff0c"
                    "\u9700\u8981\u653e\u5728\u5df2\u9a8c\u8bc1\u4ee3\u7801\u7684\u4e0a\u4e0b\u6587\u4e2d\u7406\u89e3\u3002"
                    "\u5b83\u76f4\u63a5\u53c2\u4e0e\u8f93\u5165\u5904\u7406\u3001\u72b6\u6001\u66f4\u65b0\u6216\u7ed3\u679c\u8ba1\u7b97\u3002"
                ),
                "evidence_id": line_item["id"],
            },
            {
                "category": "statement_understanding",
                "question": f"\u8fd9\u9053\u9898\u4e3a\u4ec0\u4e48\u53ef\u4ee5\u6309 `{bundle.spec.entry_point or 'stdin/stdout'}` \u8fd9\u79cd\u8f93\u5165\u8f93\u51fa\u65b9\u5f0f\u5904\u7406\uff1f",
                "answer": _annotation_explanation(annotation)
                or "\u8fd9\u4efd\u4ee3\u7801\u6839\u636e\u9898\u9762\u7ea6\u675f\u5904\u7406\u8f93\u5165\uff0c\u518d\u8ba1\u7b97\u5e76\u8fd4\u56de\u6216\u8f93\u51fa\u7ed3\u679c\u3002",
                "evidence_id": _first_evidence_id(evidence, "statement"),
            },
            {
                "category": "complexity",
                "question": "\u8fd9\u4e2a\u65f6\u95f4\u548c\u7a7a\u95f4\u590d\u6742\u5ea6\u662f\u600e\u4e48\u5224\u65ad\u7684\uff1f",
                "answer": _complexity_answer(annotation, bundle),
                "evidence_id": _first_evidence_id(evidence, "complexity"),
            },
            {
                "category": "variable_meaning",
                "question": "\u4ee3\u7801\u91cc\u7684\u53c2\u6570\u6216\u4e2d\u95f4\u53d8\u91cf\u5e94\u8be5\u600e\u4e48\u548c\u9898\u610f\u5bf9\u5e94\uff1f",
                "answer": (
                    "\u9700\u8981\u628a\u53d8\u91cf\u548c\u9898\u9762\u4e2d\u7684\u8f93\u5165\u542b\u4e49\u3001\u72b6\u6001\u542b\u4e49\u6216\u8fd4\u56de\u503c\u8981\u6c42\u5bf9\u5e94\u8d77\u6765\uff0c"
                    "\u4e0d\u80fd\u53ea\u770b\u4ee3\u7801\u8868\u9762\u7684\u8d4b\u503c\u3002"
                ),
                "evidence_id": line_item["id"],
            },
            {
                "category": "debugging_strategy",
                "question": "\u5982\u679c\u6211\u81ea\u5df1\u6539\u5199\u8fd9\u4efd\u4ee3\u7801\u540e\u53ea\u8fc7\u4e86\u6837\u4f8b\uff0c\u5e94\u8be5\u4f18\u5148\u68c0\u67e5\u54ea\u4e9b\u5730\u65b9\uff1f",
                "answer": (
                    "\u5e94\u5148\u68c0\u67e5\u7b97\u6cd5\u5047\u8bbe\u662f\u5426\u8986\u76d6\u4e00\u822c\u8f93\u5165\uff0c"
                    "\u800c\u4e0d\u662f\u628a\u4ee3\u7801\u6539\u6210\u53ea\u5339\u914d\u6837\u4f8b\u3002"
                    "\u5e38\u89c1\u65b9\u5411\u5305\u62ec\u8fb9\u754c\u6761\u4ef6\u3001\u8d85\u65f6\u3001\u7cbe\u5ea6\u3001\u91cd\u590d\u503c\u548c\u8f93\u51fa\u683c\u5f0f\u3002"
                ),
                "evidence_id": line_item["id"],
            },
        ]
        return json.dumps(items[: self.max_questions], ensure_ascii=False)


class HfDialogueModel:
    def __init__(self, model_name: str, max_new_tokens: int, load_in_4bit: bool, max_questions: int):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit("Install requirements-train.txt before using HF dialogue backend.") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.max_questions = max_questions

    def generate_dialogue(self, bundle: ProblemBundle, annotation: dict[str, Any], code: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _generation_prompt(bundle, annotation, code, self.max_questions)},
        ]
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


def _records_from_response(
    bundle: ProblemBundle,
    annotation: dict[str, Any],
    code: str,
    raw: str,
    max_questions: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    payloads = _extract_json_array(raw)
    context = _context(bundle, annotation, code)
    evidence_map = _evidence_map(bundle, annotation, code)
    records: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []
    if not payloads:
        return [], [_failure(bundle, "parse_failed", raw, None)]
    for item in payloads[:max_questions]:
        reason = _invalid_item_reason(item, code, evidence_map)
        if reason:
            failures.append(_failure(bundle, reason, raw, item))
            continue
        question = str(item["question"]).strip()
        answer = _normalized_answer(str(item["answer"]).strip(), item, evidence_map)
        records.append(
            {
                "instruction": FOLLOWUP_INSTRUCTION,
                "input": f"{context}\n\nUser follow-up:\n{question}",
                "output": answer,
            }
        )
    return records, failures


def _invalid_item_reason(item: Any, code: str, evidence_map: dict[str, dict[str, Any]] | None = None) -> str:
    if not isinstance(item, dict):
        return "item_not_object"
    for key in ("question", "answer", "category"):
        if not str(item.get(key) or "").strip():
            return f"missing_{key}"
    evidence_id = str(item.get("evidence_id") or "").strip()
    legacy_evidence = str(item.get("evidence") or "").strip()
    if not evidence_id and not legacy_evidence:
        return "missing_evidence_id"
    question = str(item["question"]).strip()
    answer = str(item["answer"]).strip()
    if not _has_chinese(question) or not _has_chinese(answer):
        return "non_chinese"
    if evidence_map is not None and not _evidence_supported(evidence_id, legacy_evidence, evidence_map):
        return "unsupported_evidence"
    if _mentions_hidden_case(answer):
        return "hidden_case_leak"
    if str(item["category"]) == "line_explanation" and not _line_question_supported(question, answer, code, evidence_id, evidence_map or {}):
        return "line_reference_missing"
    if str(item["category"]) == "complexity" and _fabricates_unknown_complexity(answer):
        return "fabricated_unknown_complexity"
    return ""


def _generation_prompt(bundle: ProblemBundle, annotation: dict[str, Any], code: str, max_questions: int) -> str:
    visible = "\n\n".join(
        f"Input:\n{case.stdin}\nExpected:\n{case.expected_stdout}" for case in bundle.tests.visible_tests[:2]
    )
    evidence_text = "\n".join(
        f"{item['id']}: ({item['source']}) {item['text']}" for item in _evidence_items(bundle, annotation, code)
    )
    return (
        "Generate follow-up SFT data for the algorithm solution below. Output JSON array only, no Markdown.\n"
        "Each item must contain: category, question, answer, evidence_id.\n"
        "Use evidence_id exactly as one of the Evidence IDs listed below. Do not create or infer new evidence IDs.\n"
        f"Generate exactly {max_questions} diverse follow-up questions.\n"
        "Role-play as a student who is learning algorithm problem solving and has just read the solution.\n"
        "Questions must be naturally phrased and directly related to this specific statement or this specific code.\n"
        f"You may use these question types, but do not mechanically cover them in a fixed order: {', '.join(SUGGESTED_QUESTION_TYPES)}.\n"
        "Prefer concrete questions about variables, branches, loops, data structures, input/output details, sample reasoning, or why a code line is needed.\n"
        "Avoid generic questions that could apply to any algorithm problem.\n"
        "All questions and answers must be Chinese.\n"
        "Answers must be grounded in the provided problem, annotation, and verified code.\n"
        "Do not invent new concrete inputs. If asking about a sample, use only a visible test shown below.\n"
        "Do not reveal reward/eval/internal test inputs or expected outputs.\n"
        "For line_explanation, use a code evidence_id such as C12 and quote that exact code line in the answer.\n"
        "For complexity, if complexity is unknown, explain uncertainty instead of inventing O(n).\n\n"
        f"Statement:\n{_clip(bundle.spec.statement, 3500)}\n\n"
        f"IO mode: {bundle.spec.io_mode}\n"
        f"Entry point: {bundle.spec.entry_point or 'stdin/stdout'}\n"
        f"Visible tests:\n{_clip(visible, 1200)}\n\n"
        f"Solution annotation:\n{json.dumps(_annotation_view(annotation, bundle), ensure_ascii=False, indent=2)}\n\n"
        f"Evidence IDs:\n{_clip(evidence_text, 2500)}\n\n"
        f"Verified solution with line numbers:\n{_numbered(_clip(code, 7000))}"
    )


def _context(bundle: ProblemBundle, annotation: dict[str, Any], code: str) -> str:
    return FOLLOWUP_TASK_HEADER + (
        f"Problem statement:\n{bundle.spec.statement}\n\n"
        f"IO mode: {bundle.spec.io_mode}\n"
        f"Entry point: {bundle.spec.entry_point or 'stdin/stdout'}\n"
        f"Solution explanation:\n{_annotation_explanation(annotation) or 'unknown'}\n"
        f"Time Complexity: {annotation.get('time_complexity') or _complexity_by_label(bundle, 'time')}\n"
        f"Space Complexity: {annotation.get('space_complexity') or _complexity_by_label(bundle, 'space')}\n\n"
        f"Verified solution with line numbers:\n{_numbered(code)}"
    )


def _extract_json_array(raw: str) -> list[Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, flags=re.S)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    return payload if isinstance(payload, list) else []


def _load_annotations(path: str) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    if not path:
        return annotations
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            problem_id = payload.get("problem_id")
            if problem_id:
                annotations[str(problem_id)] = payload
    return annotations


def _loaded_problem_ids(path: Path) -> set[str]:
    done_path = _done_ids_path(path)
    ids = set()
    if done_path.exists():
        with done_path.open(encoding="utf-8") as handle:
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
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(payload.get("input") or "")
                match = re.search(r"Problem ID:\s*(.*?)\n", text)
                if match:
                    ids.add(match.group(1).strip())
    return ids


def _done_ids_path(path: Path) -> Path:
    return path.with_name(DONE_IDS_FILENAME)


def _failure(bundle: ProblemBundle, reason: str, raw: str, item: Any) -> dict[str, Any]:
    return {
        "problem_id": bundle.spec.id,
        "reason": reason,
        "item": item,
        "raw_response": raw[:2000],
    }


def _annotation_view(annotation: dict[str, Any], bundle: ProblemBundle) -> dict[str, str]:
    return {
        "solution_explanation": _annotation_explanation(annotation) or "",
        "time_complexity": str(annotation.get("time_complexity") or _complexity_by_label(bundle, "time")),
        "space_complexity": str(annotation.get("space_complexity") or _complexity_by_label(bundle, "space")),
    }


def _evidence_items(bundle: ProblemBundle, annotation: dict[str, Any], code: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if bundle.spec.statement:
        items.append({"id": "S1", "source": "statement", "text": _clip_one_line(bundle.spec.statement, 180)})
    explanation = _annotation_explanation(annotation)
    if explanation:
        items.append({"id": "A1", "source": "annotation", "text": _clip_one_line(explanation, 180)})
    time_complexity = str(annotation.get("time_complexity") or _complexity_by_label(bundle, "time"))
    space_complexity = str(annotation.get("space_complexity") or _complexity_by_label(bundle, "space"))
    items.append({"id": "X1", "source": "complexity", "text": f"Time Complexity: {time_complexity}"})
    items.append({"id": "X2", "source": "complexity", "text": f"Space Complexity: {space_complexity}"})
    for index, case in enumerate(bundle.tests.visible_tests[:2], start=1):
        items.append(
            {
                "id": f"V{index}",
                "source": "visible_test",
                "text": _clip_one_line(f"Input: {case.stdin} Expected: {case.expected_stdout}", 180),
            }
        )
    for line_no, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        items.append({"id": f"C{line_no}", "source": "code", "line": line_no, "text": stripped})
        if len([item for item in items if item["source"] == "code"]) >= 80:
            break
    return items


def _evidence_map(bundle: ProblemBundle, annotation: dict[str, Any], code: str) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in _evidence_items(bundle, annotation, code)}


def _first_evidence_id(items: list[dict[str, Any]], source: str) -> str:
    for item in items:
        if item["source"] == source:
            return str(item["id"])
    return str(items[0]["id"]) if items else ""


def _evidence_supported(
    evidence_id: str,
    legacy_evidence: str,
    evidence_map: dict[str, dict[str, Any]],
) -> bool:
    if evidence_id:
        return evidence_id in evidence_map
    legacy_evidence = legacy_evidence.strip()
    return bool(legacy_evidence and any(legacy_evidence == str(item["text"]) for item in evidence_map.values()))


def _annotation_explanation(annotation: dict[str, Any]) -> str:
    return str(annotation.get("solution_explanation") or "").strip()


def _complexity_answer(annotation: dict[str, Any], bundle: ProblemBundle) -> str:
    time = annotation.get("time_complexity") or _complexity_by_label(bundle, "time")
    space = annotation.get("space_complexity") or _complexity_by_label(bundle, "space")
    if time == "unknown" and space == "unknown":
        return (
            "\u5f53\u524d\u590d\u6742\u5ea6\u4fe1\u606f\u4e0d\u786e\u5b9a\uff0c"
            "\u9700\u8981\u7ed3\u5408\u4ee3\u7801\u4e3b\u5faa\u73af\u3001\u9012\u5f52\u6df1\u5ea6\u6216\u72b6\u6001\u6570\u8fdb\u4e00\u6b65\u4f30\u8ba1\uff0c"
            "\u4e0d\u5e94\u76f4\u63a5\u7f16\u9020 O(n)\u3002"
        )
    return f"\u65f6\u95f4\u590d\u6742\u5ea6\u8bb0\u5f55\u4e3a {time}\uff0c\u7a7a\u95f4\u590d\u6742\u5ea6\u8bb0\u5f55\u4e3a {space}\uff0c\u9700\u8981\u7ed3\u5408\u4ee3\u7801\u4e3b\u8981\u5faa\u73af\u548c\u6570\u636e\u7ed3\u6784\u7406\u89e3\u3002"


def _complexity_by_label(bundle: ProblemBundle, label: str) -> str:
    text = bundle.oracle.expected_complexity
    if not text:
        return "unknown"
    match = re.search(rf"{label}\s*:\s*([^;]+)", text, flags=re.I)
    return match.group(1).strip() if match else "unknown"


def _line_question_supported(
    question: str,
    answer: str,
    code: str,
    evidence_id: str = "",
    evidence_map: dict[str, dict[str, Any]] | None = None,
) -> bool:
    if evidence_id and evidence_map and evidence_id in evidence_map:
        return evidence_map[evidence_id].get("source") == "code"
    line_numbers = {str(idx) for idx, _ in enumerate(code.splitlines(), start=1)}
    if not any(number in question or number in answer for number in line_numbers):
        return False
    stripped_lines = [line.strip() for line in code.splitlines() if line.strip()]
    return any(line in answer for line in stripped_lines)


def _normalized_answer(answer: str, item: dict[str, Any], evidence_map: dict[str, dict[str, Any]]) -> str:
    evidence_id = str(item.get("evidence_id") or "").strip()
    evidence = evidence_map.get(evidence_id)
    if not evidence or evidence.get("source") != "code":
        return answer
    code_text = str(evidence.get("text") or "").strip()
    if code_text and code_text not in answer and str(item.get("category")) == "line_explanation":
        return f"{answer}\n对应代码是 `{code_text}`。"
    return answer


def _fabricates_unknown_complexity(answer: str) -> bool:
    lower = answer.lower()
    return "unknown" in lower and bool(re.search(r"O\s*\([^)]+\)", answer))


def _mentions_hidden_case(text: str) -> bool:
    lowered = text.lower()
    suspicious = ["reward-", "eval-", "visible-", "internal test input", "expected output is"]
    return any(item in lowered for item in suspicious)


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _numbered(code: str) -> str:
    return "\n".join(f"{idx}: {line}" for idx, line in enumerate(code.splitlines(), start=1))


def _representative_line(code: str) -> tuple[int, str]:
    for idx, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return idx, line
    return 1, code.splitlines()[0] if code.splitlines() else ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _clip_one_line(text: str, limit: int) -> str:
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= limit else one_line[:limit] + "...[truncated]"


if __name__ == "__main__":
    main()
