from __future__ import annotations

import unittest

from algoagent.executor import CppExecutor
from algoagent.schema import TestCase


class CppExecutorTest(unittest.TestCase):
    def test_evaluate_passing_program(self) -> None:
        code = """#include <bits/stdc++.h>
using namespace std;
int main() {
    int a, b;
    cin >> a >> b;
    cout << a + b << "\\n";
    return 0;
}
"""
        result = CppExecutor().evaluate(
            code,
            [TestCase(id="sum", stdin="2 3\n", expected_stdout="5\n")],
            suite_name="repair",
        )
        self.assertTrue(result.compiled)
        self.assertTrue(result.all_passed)
        self.assertEqual(result.tests[0].test_id, "sum")
        self.assertEqual(result.tests[0].suite, "repair")

    def test_compile_error_is_reported(self) -> None:
        result = CppExecutor().evaluate("int main( {", [])
        self.assertFalse(result.compiled)
        self.assertIn("error", result.compile_error.lower())


if __name__ == "__main__":
    unittest.main()

