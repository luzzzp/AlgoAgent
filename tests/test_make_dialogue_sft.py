from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoagent.schema import OracleMetadata, OracleSolution, ProblemBundle, ProblemSpec, TestSuite


SCRIPT = Path("scripts/make_dialogue_sft.py").resolve()
SPEC = importlib.util.spec_from_file_location("make_dialogue_sft", SCRIPT)
assert SPEC and SPEC.loader
make_dialogue_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_dialogue_sft)


class MakeDialogueSftTest(unittest.TestCase):
    def test_records_use_common_sft_schema(self) -> None:
        bundle = ProblemBundle(
            spec=ProblemSpec(
                id="p",
                title="P",
                statement="Return sum.",
                io_mode="callable",
                entry_point="add",
            ),
            tests=TestSuite(),
            oracle=OracleMetadata(
                expected_complexity="Time: O(1); Space: O(1)",
                solutions=[OracleSolution("python3", "def add(a, b):\n    return a + b\n", True)],
            ),
        )

        records = make_dialogue_sft._records(bundle, max_questions=2)

        self.assertEqual(len(records), 2)
        self.assertIn("instruction", records[0])
        self.assertIn("input", records[0])
        self.assertIn("output", records[0])
        self.assertIn("Verified solution with line numbers", records[0]["input"])
        self.assertRegex(records[0]["output"], r"[\u4e00-\u9fff]")


if __name__ == "__main__":
    unittest.main()
