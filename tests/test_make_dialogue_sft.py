from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoagent.schema import OracleMetadata, OracleSolution, ProblemBundle, ProblemSpec, TestCase, TestSuite


SCRIPT = Path("scripts/make_dialogue_sft.py").resolve()
SPEC = importlib.util.spec_from_file_location("make_dialogue_sft", SCRIPT)
assert SPEC and SPEC.loader
make_dialogue_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_dialogue_sft)


CODE = "def add(a, b):\n    return a + b\n"


def _bundle() -> ProblemBundle:
    return ProblemBundle(
        spec=ProblemSpec(
            id="p1",
            title="Sum",
            statement="Return the sum of two numbers.",
            io_mode="callable",
            entry_point="add",
        ),
        tests=TestSuite(visible_tests=[TestCase("[2, 3]", "5")]),
        oracle=OracleMetadata(
            expected_complexity="Time: O(1); Space: O(1)",
            solutions=[OracleSolution("python3", CODE, True)],
        ),
    )


class MakeDialogueSftTest(unittest.TestCase):
    def test_mock_backend_generates_common_sft_schema(self) -> None:
        bundle = _bundle()
        annotation = {"solution_explanation": "直接返回两个数的和。", "time_complexity": "O(1)", "space_complexity": "O(1)"}
        raw = make_dialogue_sft.MockDialogueModel(max_questions=5).generate_dialogue(bundle, annotation, CODE)

        records, failures = make_dialogue_sft._records_from_response(
            bundle,
            annotation,
            CODE,
            raw,
            max_questions=5,
        )

        self.assertEqual(len(records), 5)
        self.assertEqual(failures, [])
        self.assertIn("instruction", records[0])
        self.assertIn("input", records[0])
        self.assertIn("output", records[0])
        self.assertTrue(records[0]["input"].startswith("Task Type: ANSWER_FOLLOW_UP\n"))
        self.assertIn("Interaction: User follow-up after verified solution", records[0]["input"])
        self.assertIn("Problem ID: p1", records[0]["input"])
        self.assertIn("Verified solution with line numbers", records[0]["input"])
        self.assertIn("User follow-up:", records[0]["input"])
        self.assertRegex(records[0]["output"], r"[\u4e00-\u9fff]")

    def test_extract_json_array_from_markdown_response(self) -> None:
        raw = '```json\n[{"category":"algorithm_idea","question":"思路是什么？","answer":"直接相加。","evidence_id":"C2"}]\n```'

        payload = make_dialogue_sft._extract_json_array(raw)

        self.assertEqual(payload[0]["category"], "algorithm_idea")

    def test_invalid_non_chinese_item_is_rejected(self) -> None:
        reason = make_dialogue_sft._invalid_item_reason(
            {"category": "algorithm_idea", "question": "What?", "answer": "Use sum.", "evidence_id": "C2"},
            CODE,
            {"C2": {"source": "code", "text": "return a + b"}},
        )

        self.assertEqual(reason, "non_chinese")

    def test_invalid_missing_field_is_rejected(self) -> None:
        reason = make_dialogue_sft._invalid_item_reason(
            {"category": "algorithm_idea", "question": "思路是什么？", "evidence_id": "C2"},
            CODE,
            {"C2": {"source": "code", "text": "return a + b"}},
        )

        self.assertEqual(reason, "missing_answer")

    def test_hidden_case_leak_is_rejected(self) -> None:
        reason = make_dialogue_sft._invalid_item_reason(
            {
                "category": "hidden_failure_analysis",
                "question": "隐藏测试失败怎么办？",
                "answer": "可以查看 eval-0001 的 expected output is 5。",
                "evidence_id": "C2",
            },
            CODE,
            {"C2": {"source": "code", "text": "return a + b"}},
        )

        self.assertEqual(reason, "hidden_case_leak")

    def test_line_question_must_quote_real_code(self) -> None:
        reason = make_dialogue_sft._invalid_item_reason(
            {
                "category": "line_explanation",
                "question": "第 2 行是什么意思？",
                "answer": "第 2 行用于返回结果。",
                "evidence_id": "X1",
            },
            CODE,
            {"X1": {"source": "complexity", "text": "Time Complexity: O(1)"}},
        )

        self.assertEqual(reason, "line_reference_missing")

    def test_fabricated_unknown_complexity_is_rejected(self) -> None:
        reason = make_dialogue_sft._invalid_item_reason(
            {
                "category": "complexity",
                "question": "复杂度是什么？",
                "answer": "虽然复杂度 unknown，但可以认为是 O(n)。",
                "evidence_id": "X1",
            },
            CODE,
            {"X1": {"source": "complexity", "text": "Time Complexity: unknown"}},
        )

        self.assertEqual(reason, "fabricated_unknown_complexity")

    def test_unsupported_evidence_is_rejected(self) -> None:
        reason = make_dialogue_sft._invalid_item_reason(
            {
                "category": "algorithm_idea",
                "question": "这段代码为什么这样写？",
                "answer": "因为它直接返回两个数的和。",
                "evidence_id": "NOPE",
            },
            CODE,
            {"C2": {"source": "code", "text": "return a + b"}},
        )

        self.assertEqual(reason, "unsupported_evidence")

    def test_generation_prompt_requests_diverse_student_questions(self) -> None:
        bundle = _bundle()

        prompt = make_dialogue_sft._generation_prompt(
            bundle,
            {"solution_explanation": "直接返回两个数的和。", "time_complexity": "O(1)", "space_complexity": "O(1)"},
            CODE,
            max_questions=7,
        )

        self.assertIn("Generate exactly 7 diverse follow-up questions", prompt)
        self.assertIn("Role-play as a student", prompt)
        self.assertIn("specific statement or this specific code", prompt)
        self.assertIn("do not mechanically cover them in a fixed order", prompt)
        self.assertIn("category, question, answer, evidence_id", prompt)
        self.assertIn("Use evidence_id exactly as one of the Evidence IDs", prompt)
        self.assertIn("Do not invent new concrete inputs", prompt)

    def test_code_evidence_ids_use_real_line_numbers(self) -> None:
        code = "\n".join(f"x{i} = {i}" for i in range(1, 35))
        bundle = _bundle()

        items = make_dialogue_sft._evidence_items(bundle, {}, code)
        evidence_ids = {item["id"] for item in items}

        self.assertIn("C1", evidence_ids)
        self.assertIn("C25", evidence_ids)
        self.assertIn("C34", evidence_ids)

    def test_normalized_line_answer_adds_code_quote(self) -> None:
        answer = make_dialogue_sft._normalized_answer(
            "第 2 行用于返回两个数的和。",
            {"category": "line_explanation", "evidence_id": "C2"},
            {"C2": {"source": "code", "text": "return a + b"}},
        )

        self.assertIn("`return a + b`", answer)


if __name__ == "__main__":
    unittest.main()
