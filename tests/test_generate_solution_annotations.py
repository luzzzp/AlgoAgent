from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoagent.schema import OracleMetadata, OracleSolution, ProblemBundle, ProblemSpec, TestSuite


SCRIPT = Path("scripts/generate_solution_annotations.py").resolve()
SPEC = importlib.util.spec_from_file_location("generate_solution_annotations", SCRIPT)
assert SPEC and SPEC.loader
generate_solution_annotations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_solution_annotations)


class GenerateSolutionAnnotationsTest(unittest.TestCase):
    def test_parse_annotation_extracts_json_object(self) -> None:
        bundle = ProblemBundle(
            spec=ProblemSpec(id="p", title="P", statement="Solve it."),
            tests=TestSuite(),
            oracle=OracleMetadata(solutions=[OracleSolution("python3", "print(1)", True)]),
        )
        raw = '```json\n{"solution_explanation":"读取输入后输出答案。","time_complexity":"O(1)","space_complexity":"O(1)"}\n```'

        record = generate_solution_annotations._parse_annotation(bundle, raw)

        self.assertEqual(record["problem_id"], "p")
        self.assertTrue(record["valid"])
        self.assertEqual(record["time_complexity"], "O(1)")

    def test_parse_annotation_falls_back_on_invalid_response(self) -> None:
        bundle = ProblemBundle(
            spec=ProblemSpec(
                id="p",
                title="P",
                statement="Solve it.",
                io_mode="callable",
                entry_point="solve",
            ),
            tests=TestSuite(),
            oracle=OracleMetadata(solutions=[OracleSolution("python3", "def solve(): return 1", True)]),
        )

        record = generate_solution_annotations._parse_annotation(bundle, "not json")

        self.assertFalse(record["valid"])
        self.assertIn("`solve`", record["solution_explanation"])

    def test_annotation_prompt_does_not_include_title(self) -> None:
        bundle = ProblemBundle(
            spec=ProblemSpec(id="p", title="Misleading Title", statement="Solve it."),
            tests=TestSuite(),
            oracle=OracleMetadata(solutions=[OracleSolution("python3", "print(1)", True)]),
        )

        prompt = generate_solution_annotations._annotation_prompt(bundle, "print(1)")

        self.assertNotIn("Title:", prompt)
        self.assertNotIn("Misleading Title", prompt)
        self.assertIn("Statement:", prompt)


if __name__ == "__main__":
    unittest.main()
