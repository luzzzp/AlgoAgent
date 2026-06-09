from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from algoagent.schema import CandidateResult, ExecutionResult, TestCase, TestRun, normalize_output


class CppExecutor:
    """Compile C++17 code and run it against deterministic test cases."""

    def __init__(self, compiler: str = "g++", compile_timeout_sec: float = 20.0):
        self.compiler = compiler
        self.compile_timeout_sec = compile_timeout_sec

    def evaluate(
        self,
        code: str,
        tests: list[TestCase],
        default_timeout_sec: float = 2.0,
        suite_name: str = "tests",
    ) -> CandidateResult:
        with tempfile.TemporaryDirectory(prefix="algoagent_") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "main.cpp"
            exe = tmp_path / ("main.exe" if _is_windows_exe() else "main")
            source.write_text(code, encoding="utf-8")

            compile_result = self._compile(source, exe)
            if not compile_result.compiled:
                return CandidateResult(
                    compiled=False,
                    compile_error=compile_result.compile_error,
                    tests=[],
                )

            runs = [self._run_one(exe, case, default_timeout_sec, suite_name) for case in tests]
            return CandidateResult(compiled=True, compile_error="", tests=runs)

    def _compile(self, source: Path, exe: Path) -> ExecutionResult:
        cmd = [
            self.compiler,
            "-std=c++17",
            "-O2",
            "-pipe",
            str(source),
            "-o",
            str(exe),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.compile_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                compiled=False,
                passed=False,
                timed_out=True,
                compile_error=f"Compilation timed out after {self.compile_timeout_sec}s: {exc}",
            )
        except FileNotFoundError:
            return ExecutionResult(
                compiled=False,
                passed=False,
                compile_error=f"Compiler not found: {self.compiler}",
            )

        if completed.returncode != 0:
            return ExecutionResult(
                compiled=False,
                passed=False,
                stderr=completed.stderr,
                returncode=completed.returncode,
                compile_error=completed.stderr.strip(),
            )
        return ExecutionResult(compiled=True, passed=True, returncode=0)

    def _run_one(
        self,
        exe: Path,
        case: TestCase,
        default_timeout_sec: float,
        suite_name: str,
    ) -> TestRun:
        timeout_sec = case.timeout_sec if case.timeout_sec is not None else default_timeout_sec
        try:
            completed = subprocess.run(
                [str(exe)],
                input=case.stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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


def _is_windows_exe() -> bool:
    return "\\" in str(Path.cwd()) or Path("C:/").exists()
