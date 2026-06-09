from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

from algoagent.schema import CandidateResult, ExecutionResult, TestCase, TestRun, normalize_output


class PythonExecutor:
    """Run Python stdin/stdout solutions for oracle validation."""

    def __init__(self, interpreter: str | None = None, compile_timeout_sec: float = 10.0):
        self.interpreter = interpreter or sys.executable
        self.compile_timeout_sec = compile_timeout_sec

    def evaluate(
        self,
        code: str,
        tests: list[TestCase],
        default_timeout_sec: float = 2.0,
        suite_name: str = "tests",
    ) -> CandidateResult:
        with tempfile.TemporaryDirectory(prefix="algoagent_py_") as tmp:
            source = Path(tmp) / "solution.py"
            source.write_text(code, encoding="utf-8")
            syntax = self._syntax_check(source)
            if not syntax.compiled:
                return CandidateResult(compiled=False, compile_error=syntax.compile_error, tests=[])
            runs = [self._run_one(source, case, default_timeout_sec, suite_name) for case in tests]
            return CandidateResult(compiled=True, compile_error="", tests=runs)

    def _syntax_check(self, source: Path) -> ExecutionResult:
        try:
            completed = subprocess.run(
                [self.interpreter, "-m", "py_compile", str(source)],
                capture_output=True,
                text=True,
                timeout=self.compile_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                compiled=False,
                passed=False,
                timed_out=True,
                compile_error=f"Python syntax check timed out: {exc}",
            )
        if completed.returncode != 0:
            return ExecutionResult(
                compiled=False,
                passed=False,
                returncode=completed.returncode,
                stderr=completed.stderr,
                compile_error=completed.stderr.strip(),
            )
        return ExecutionResult(compiled=True, passed=True, returncode=0)

    def _run_one(
        self,
        source: Path,
        case: TestCase,
        default_timeout_sec: float,
        suite_name: str,
    ) -> TestRun:
        timeout_sec = case.timeout_sec if case.timeout_sec is not None else default_timeout_sec
        try:
            completed = subprocess.run(
                [self.interpreter, str(source)],
                input=case.stdin,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return TestRun(
                test_id=case.id,
                suite=suite_name,
                passed=False,
                expected=normalize_output(case.expected_stdout),
                actual=normalize_output(exc.stdout or ""),
                stderr=normalize_output(exc.stderr or ""),
                timed_out=True,
            )
        actual = normalize_output(completed.stdout)
        expected = normalize_output(case.expected_stdout)
        return TestRun(
            test_id=case.id,
            suite=suite_name,
            passed=completed.returncode == 0 and actual == expected,
            expected=expected,
            actual=actual,
            stderr=normalize_output(completed.stderr),
            timed_out=False,
        )

