from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/convert_taco_python.py").resolve()
SPEC = importlib.util.spec_from_file_location("convert_taco_python", SCRIPT)
assert SPEC and SPEC.loader
convert_taco_python = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert_taco_python)


class ConvertTacoPythonTest(unittest.TestCase):
    def test_skipped_record_contains_reason_and_previews(self) -> None:
        row = {
            "title": "Bad IO",
            "statement": "Solve it.\nSecond line.",
            "input_output": {"inputs": ["1"], "outputs": []},
            "solutions": ["print(1)"],
        }

        record = convert_taco_python._skipped_record(row, 7, "missing_or_mismatched_tests")

        self.assertEqual(record["index"], 7)
        self.assertEqual(record["reason"], "missing_or_mismatched_tests")
        self.assertEqual(record["title"], "Bad IO")
        self.assertIn("input_output", record["keys"])
        self.assertIn("\\n", record["statement_preview"])

    def test_parses_human_readable_limits(self) -> None:
        self.assertEqual(convert_taco_python._parse_time_limit("1.0 seconds"), 1.0)
        self.assertEqual(convert_taco_python._parse_time_limit("0.5 seconds"), 0.5)
        self.assertEqual(convert_taco_python._parse_memory_limit("256 megabytes"), 256)
        self.assertEqual(convert_taco_python._parse_memory_limit("1 GB"), 1024)

    def test_nested_test_case_lines_are_joined(self) -> None:
        row = {
            "input_output": {
                "inputs": [["1", "5 10"], "7\n8"],
                "outputs": [["50"], "15"],
            }
        }

        inputs, outputs = convert_taco_python._extract_tests(row)

        self.assertEqual(inputs, ["1\n5 10", "7\n8"])
        self.assertEqual(outputs, ["50", "15"])


if __name__ == "__main__":
    unittest.main()
