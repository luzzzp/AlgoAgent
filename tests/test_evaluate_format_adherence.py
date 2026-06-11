from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/evaluate_format_adherence.py").resolve()
SPEC = importlib.util.spec_from_file_location("evaluate_format_adherence", SCRIPT)
assert SPEC and SPEC.loader
format_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(format_eval)


class EvaluateFormatAdherenceTest(unittest.TestCase):
    def test_detects_valid_format(self) -> None:
        text = """Solution Explanation:
Use a hash map.

Time Complexity: O(n)
Space Complexity: O(n)
```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){return 0;}
```
"""
        self.assertTrue(format_eval.has_explanation_field(text))
        self.assertTrue(format_eval.has_complexity_fields(text))
        self.assertTrue(format_eval.has_cpp_code_block(text))
        self.assertIn("int main", format_eval.extract_cpp_code(text))

    def test_summarizes_format_rates(self) -> None:
        summary = format_eval.summarize(
            [
                {
                    "format_valid": True,
                    "explanation_field": True,
                    "complexity_field": True,
                    "cpp_code_block": True,
                    "compiled": True,
                },
                {
                    "format_valid": False,
                    "explanation_field": False,
                    "complexity_field": True,
                    "cpp_code_block": False,
                    "compiled": False,
                },
            ]
        )
        self.assertEqual(summary["num_problems"], 2)
        self.assertEqual(summary["format_valid_rate"], 0.5)
        self.assertEqual(summary["complexity_field_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
