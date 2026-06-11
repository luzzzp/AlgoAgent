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
使用哈希表记录已经出现过的数，并在扫描时查找目标差值。

Time Complexity: O(n)
Space Complexity: O(n)
```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){return 0;}
```
"""
        self.assertTrue(format_eval.has_explanation_field(text))
        self.assertTrue(format_eval.has_chinese_explanation(text))
        self.assertTrue(format_eval.has_complexity_fields(text))
        self.assertTrue(format_eval.has_cpp_code_block(text))

    def test_rejects_non_chinese_explanation(self) -> None:
        text = """Solution Explanation:
Use a hash map.

Time Complexity: O(n)
Space Complexity: O(n)
```cpp
int main(){return 0;}
```
"""
        self.assertTrue(format_eval.has_explanation_field(text))
        self.assertFalse(format_eval.has_chinese_explanation(text))

    def test_accepts_markdown_headings_without_colons(self) -> None:
        text = """### Solution Explanation

题目要求根据规则推导答案，可以先分析输入规模，再选择合适的模拟或数学方法。

### Time Complexity
- 总时间复杂度为 O(n log n)。

### Space Complexity
- 额外空间复杂度为 O(n)。

```cpp
int main(){return 0;}
```
"""
        self.assertTrue(format_eval.has_explanation_field(text))
        self.assertTrue(format_eval.has_chinese_explanation(text))
        self.assertTrue(format_eval.has_complexity_fields(text))
        self.assertTrue(format_eval.has_cpp_code_block(text))

    def test_summarizes_format_rates(self) -> None:
        summary = format_eval.summarize(
            [
                {
                    "format_valid": True,
                    "explanation_field": True,
                    "chinese_explanation": True,
                    "complexity_field": True,
                    "cpp_code_block": True,
                },
                {
                    "format_valid": False,
                    "explanation_field": False,
                    "chinese_explanation": False,
                    "complexity_field": True,
                    "cpp_code_block": False,
                },
            ]
        )
        self.assertEqual(summary["num_problems"], 2)
        self.assertEqual(summary["format_valid_rate"], 0.5)
        self.assertEqual(summary["chinese_explanation_rate"], 0.5)
        self.assertEqual(summary["complexity_field_rate"], 1.0)
        self.assertNotIn("compile_rate", summary)


if __name__ == "__main__":
    unittest.main()
