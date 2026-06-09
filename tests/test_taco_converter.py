from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path("scripts/convert_taco_verified.py").resolve()
SPEC = importlib.util.spec_from_file_location("convert_taco_verified", SCRIPT)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class TacoConverterTest(unittest.TestCase):
    def test_converts_stdin_stdout_python_oracle(self) -> None:
        row = {
            "question": "Given n, print n.\nConstraints: 1 <= n <= 100",
            "name": "Echo Number",
            "input_output": json.dumps(
                {
                    "inputs": ["1\n", "2\n", "3\n"],
                    "outputs": ["1\n", "2\n", "3\n"],
                }
            ),
            "solutions": json.dumps(["n = int(input())\nprint(n)\n"]),
            "difficulty": "EASY",
            "tags": ["implementation"],
            "source": "unit",
            "url": "https://example.com/problem",
            "time_limit": "1 second",
            "memory_limit": "128 megabytes",
            "Expected Time Complexity": "O(1)",
        }
        converted = converter.convert_row(row, index=7, repair_ratio=0.67, min_tests=2)
        self.assertEqual(converted["problem"]["language"], "cpp17")
        self.assertEqual(converted["problem"]["time_limit_sec"], 1.0)
        self.assertEqual(converted["problem"]["memory_limit_mb"], 128)
        self.assertEqual(len(converted["tests"]["repair_tests"]), 2)
        self.assertEqual(len(converted["tests"]["eval_tests"]), 1)
        self.assertEqual(converted["oracle"]["solutions"][0]["language"], "python3")
        self.assertFalse(converted["oracle"]["solutions"][0]["verified"])
        self.assertEqual(converted["oracle"]["reference_solution"], "")

    def test_caps_repair_and_eval_tests(self) -> None:
        row = {
            "question": "Given n, print n.",
            "input_output": json.dumps(
                {
                    "inputs": [f"{i}\n" for i in range(10)],
                    "outputs": [f"{i}\n" for i in range(10)],
                }
            ),
            "solutions": json.dumps(["n = int(input())\nprint(n)\n"]),
        }
        converted = converter.convert_row(row, index=1, max_repair_tests=3, max_eval_tests=2)
        self.assertEqual(len(converted["tests"]["repair_tests"]), 3)
        self.assertEqual(len(converted["tests"]["eval_tests"]), 2)

    def test_skips_function_tasks_by_default(self) -> None:
        row = {
            "question": "Return x.",
            "input_output": json.dumps({"fn_name": "solve", "inputs": [1, 2], "outputs": [1, 2]}),
            "solutions": json.dumps(["def solve(x): return x"]),
        }
        with self.assertRaises(converter.ConversionError) as raised:
            converter.convert_row(row, index=0)
        self.assertEqual(raised.exception.reason, "function_task")


if __name__ == "__main__":
    unittest.main()
