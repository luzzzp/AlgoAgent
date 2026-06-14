from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path("scripts/merge_sft_datasets.py").resolve()
SPEC = importlib.util.spec_from_file_location("merge_sft_datasets", SCRIPT)
assert SPEC and SPEC.loader
merge_sft_datasets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_sft_datasets)


class MergeSftDatasetsTest(unittest.TestCase):
    def test_validate_requires_common_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            with self.assertRaises(ValueError):
                merge_sft_datasets._validate({"instruction": "x"}, path)

    def test_validate_accepts_common_schema(self) -> None:
        merge_sft_datasets._validate({"instruction": "i", "input": "x", "output": "y"}, Path("ok.jsonl"))


if __name__ == "__main__":
    unittest.main()
