from __future__ import annotations

from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile

from algoagent.schema import ExecutionReport, ExecutionRun, TestCase, normalize_output


class PythonExecutor:
    """Run generated Python programs with per-test timeout and output caps."""

    def __init__(
        self,
        python_bin: str | None = None,
        max_stdout_bytes: int = 200_000,
        max_stderr_bytes: int = 100_000,
    ):
        self.python_bin = python_bin or sys.executable
        self.max_stdout_bytes = max_stdout_bytes
        self.max_stderr_bytes = max_stderr_bytes

    def evaluate(
        self,
        code: str,
        tests: list[TestCase],
        default_timeout_sec: float = 2.0,
        suite_name: str = "tests",
        entry_point: str = "",
    ) -> ExecutionReport:
        with tempfile.TemporaryDirectory(prefix="algoagent_py_") as tmp:
            script = Path(tmp) / "main.py"
            script.write_text(code, encoding="utf-8")
            syntax_error = self._syntax_error(script)
            if syntax_error:
                return ExecutionReport(syntax_valid=False, syntax_error=syntax_error, runs=[])
            runner = Path(tmp) / "callable_runner.py"
            if entry_point:
                runner.write_text(_CALLABLE_RUNNER, encoding="utf-8")
            runs = [
                self._run_one(script, case, default_timeout_sec, suite_name, entry_point, runner)
                for case in tests
            ]
            return ExecutionReport(syntax_valid=True, runs=runs)

    def _syntax_error(self, script: Path) -> str:
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as exc:
            return str(exc)
        return ""

    def _run_one(
        self,
        script: Path,
        case: TestCase,
        default_timeout_sec: float,
        suite_name: str,
        entry_point: str = "",
        runner: Path | None = None,
    ) -> ExecutionRun:
        timeout = case.timeout_sec if case.timeout_sec is not None else default_timeout_sec
        command = (
            [self.python_bin, str(runner), str(script), entry_point]
            if entry_point and runner is not None
            else [self.python_bin, str(script)]
        )
        try:
            completed = subprocess.run(
                command,
                input=case.stdin.encode("utf-8"),
                capture_output=True,
                text=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionRun(
                test_id=case.id,
                suite=suite_name,
                passed=False,
                expected=normalize_output(case.expected_stdout),
                actual=normalize_output(self._cap(exc.stdout or b"", self.max_stdout_bytes)[0]),
                stderr=normalize_output(self._cap(exc.stderr or b"", self.max_stderr_bytes)[0]),
                timed_out=True,
            )

        stdout, stdout_truncated = self._cap(completed.stdout or b"", self.max_stdout_bytes)
        stderr, stderr_truncated = self._cap(completed.stderr or b"", self.max_stderr_bytes)
        actual = normalize_output(stdout)
        expected = normalize_output(case.expected_stdout)
        return ExecutionRun(
            test_id=case.id,
            suite=suite_name,
            passed=completed.returncode == 0 and actual == expected and not stdout_truncated,
            expected=expected,
            actual=actual,
            stderr=normalize_output(stderr),
            returncode=completed.returncode,
            timed_out=False,
            output_truncated=stdout_truncated or stderr_truncated,
        )

    @staticmethod
    def _cap(payload: bytes, limit: int) -> tuple[bytes, bool]:
        if len(payload) <= limit:
            return payload, False
        return payload[:limit], True


_CALLABLE_RUNNER = r"""
import json
import runpy
import sys


def main():
    script = sys.argv[1]
    entry_point = sys.argv[2]
    namespace = runpy.run_path(script)
    target = namespace.get(entry_point)
    if target is None and "Solution" in namespace:
        target = getattr(namespace["Solution"](), entry_point, None)
    if target is None:
        raise AttributeError(f"entry point not found: {entry_point}")

    raw = sys.stdin.read().strip()
    args = json.loads(raw) if raw else []
    if not isinstance(args, list):
        args = [args]
    result = target(*args)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
"""
