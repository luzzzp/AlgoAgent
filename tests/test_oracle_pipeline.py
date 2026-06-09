from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from algoagent.schema import load_problem


def _load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_script = _load_script("verify_python_oracles", "scripts/verify_python_oracles.py")
translate_script = _load_script("translate_python_to_cpp", "scripts/translate_python_to_cpp.py")


class OraclePipelineTest(unittest.TestCase):
    def test_verifies_python_oracle_and_translates_to_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            py_verified = root / "py_verified"
            cpp_verified = root / "cpp_verified"
            raw.mkdir()
            (raw / "echo.json").write_text(json.dumps(_echo_problem(), indent=2), encoding="utf-8")
            (raw / "_manifest.json").write_text("{}", encoding="utf-8")

            py_report = verify_script.verify_python_oracles(raw, py_verified, max_solutions_per_problem=2)
            self.assertEqual(py_report["verified"], 1)
            py_bundle = load_problem(py_verified / "echo.json")
            self.assertTrue(py_bundle.oracle.best_solution("python3"))
            self.assertTrue(next(sol for sol in py_bundle.oracle.solutions if sol.language == "python3").verified)

            cpp_report = translate_script.translate_python_to_cpp(
                py_verified,
                cpp_verified,
                translate_script.MockTranslator(),
                candidates_per_problem=1,
            )
            self.assertEqual(cpp_report["verified_cpp"], 1)
            cpp_bundle = load_problem(cpp_verified / "echo.json")
            self.assertTrue(cpp_bundle.oracle.best_solution("cpp17"))
            self.assertTrue(cpp_bundle.oracle.reference_solution)

    def test_python_oracle_failure_does_not_mark_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            out = Path(tmp) / "out"
            raw.mkdir()
            problem = _echo_problem()
            problem["oracle"]["solutions"][0]["code"] = "print(0)\n"
            (raw / "bad.json").write_text(json.dumps(problem, indent=2), encoding="utf-8")
            report = verify_script.verify_python_oracles(raw, out, max_solutions_per_problem=1)
            self.assertEqual(report["verified"], 0)
            bundle = load_problem(out / "bad.json")
            self.assertFalse(next(sol for sol in bundle.oracle.solutions if sol.language == "python3").verified)


def _echo_problem() -> dict:
    return {
        "problem": {
            "id": "echo",
            "title": "Echo",
            "statement": "Given n, print n.",
            "input_format": "n",
            "output_format": "n",
            "constraints": ["1 <= n <= 10"],
            "language": "cpp17",
            "time_limit_sec": 2,
            "memory_limit_mb": 256,
        },
        "tests": {
            "repair_tests": [{"stdin": "1\n", "expected_stdout": "1\n"}],
            "eval_tests": [{"stdin": "2\n", "expected_stdout": "2\n"}],
        },
        "oracle": {
            "difficulty": "easy",
            "tags": ["implementation"],
            "expected_complexity": "O(1)",
            "reference_solution": "",
            "solutions": [{"language": "python3", "code": "n = int(input())\nprint(n)\n", "verified": False}],
            "source": "unit",
            "url": "",
        },
    }


if __name__ == "__main__":
    unittest.main()

