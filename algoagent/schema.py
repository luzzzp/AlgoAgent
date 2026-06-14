from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class AgentStatus(str, Enum):
    SAMPLE_SOLVED = "SAMPLE_SOLVED"
    VERIFIED_SOLVED = "VERIFIED_SOLVED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ProblemSpec:
    id: str
    title: str
    statement: str
    input_format: str = ""
    output_format: str = ""
    constraints: list[str] = field(default_factory=list)
    language: str = "python3"
    time_limit_sec: float = 2.0
    memory_limit_mb: int = 256
    io_mode: str = "stdin"
    entry_point: str = ""

    def prompt(self, include_visible_tests: list[TestCase] | None = None) -> str:
        constraints = "\n".join(f"- {item}" for item in self.constraints) or "- Not specified"
        mode = (
            f"Callable entry point: {self.entry_point}\n"
            "Implement this function directly. Do not read from stdin unless the statement requires it.\n"
            if self.io_mode == "callable" and self.entry_point
            else ""
        )
        visible = ""
        if include_visible_tests:
            blocks = []
            for case in include_visible_tests:
                blocks.append(
                    f"Input:\n{case.stdin.rstrip()}\nExpected output:\n{case.expected_stdout.rstrip()}"
                )
            visible = "\n\nVisible samples:\n" + "\n\n".join(blocks)
        return (
            f"Title: {self.title}\n\n"
            f"{self.statement}\n\n"
            f"Input format:\n{self.input_format}\n\n"
            f"Output format:\n{self.output_format}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Language: {self.language}\n"
            f"Time limit: {self.time_limit_sec:g} seconds\n"
            f"Memory limit: {self.memory_limit_mb} MB\n"
            f"{mode}"
            f"{visible}\n\n"
            "Return a Python 3 solution with Chinese explanation and structured complexity."
        )


@dataclass(frozen=True)
class TestCase:
    stdin: str
    expected_stdout: str
    id: str = ""
    timeout_sec: float | None = None


@dataclass(frozen=True)
class TestSuite:
    visible_tests: list[TestCase] = field(default_factory=list)
    reward_tests: list[TestCase] = field(default_factory=list)
    eval_tests: list[TestCase] = field(default_factory=list)


@dataclass(frozen=True)
class OracleSolution:
    language: str
    code: str
    verified: bool = False


@dataclass(frozen=True)
class OracleMetadata:
    difficulty: str = ""
    tags: list[str] = field(default_factory=list)
    expected_complexity: str = ""
    reference_solution: str = ""
    solutions: list[OracleSolution] = field(default_factory=list)
    source: str = ""
    url: str = ""

    def best_solution(self, language: str = "python3") -> str:
        for solution in self.solutions:
            if solution.language == language and solution.verified:
                return solution.code
        for solution in self.solutions:
            if solution.language == language:
                return solution.code
        return self.reference_solution if language == "python3" else ""


@dataclass(frozen=True)
class ProblemBundle:
    spec: ProblemSpec
    tests: TestSuite
    oracle: OracleMetadata = field(default_factory=OracleMetadata)


@dataclass(frozen=True)
class ExecutionRun:
    test_id: str
    suite: str
    passed: bool
    expected: str
    actual: str
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    output_truncated: bool = False


@dataclass(frozen=True)
class ExecutionReport:
    syntax_valid: bool
    syntax_error: str = ""
    runs: list[ExecutionRun] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.syntax_valid and all(run.passed for run in self.runs)

    @property
    def pass_rate(self) -> float:
        return sum(1 for run in self.runs if run.passed) / len(self.runs) if self.runs else 0.0


@dataclass(frozen=True)
class AgentTrace:
    turn: int
    stage: str
    message: str


@dataclass(frozen=True)
class AttemptRecord:
    turn: int
    code: str
    visible_result: ExecutionReport | None = None
    eval_result: ExecutionReport | None = None
    diagnostic: str = ""


@dataclass(frozen=True)
class AgentResult:
    problem_id: str
    status: AgentStatus
    attempts: int
    code: str | None
    explanation: str | None
    failure_reason: str | None
    diagnostic_summary: str
    attempt_records: list[AttemptRecord]
    traces: list[AgentTrace]


def normalize_output(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def load_problem(path: str | Path) -> ProblemBundle:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    try:
        return problem_bundle_from_dict(payload)
    except KeyError as exc:
        raise ValueError(f"{source} is not an AlgoAgent-Py problem JSON; missing key {exc!s}.") from exc


def load_problems(path: str | Path) -> list[ProblemBundle]:
    root = Path(path)
    if root.is_file():
        return [load_problem(root)]
    return [load_problem(item) for item in sorted(root.glob("*.json")) if not item.name.startswith("_")]


def problem_bundle_from_dict(payload: dict[str, Any]) -> ProblemBundle:
    spec_payload = payload["problem"]
    tests_payload = payload.get("tests", {})
    oracle_payload = payload.get("oracle", {})
    spec = ProblemSpec(
        id=spec_payload["id"],
        title=spec_payload["title"],
        statement=spec_payload["statement"],
        input_format=spec_payload.get("input_format", ""),
        output_format=spec_payload.get("output_format", ""),
        constraints=list(spec_payload.get("constraints", [])),
        language=spec_payload.get("language", "python3"),
        time_limit_sec=float(spec_payload.get("time_limit_sec", 2.0)),
        memory_limit_mb=int(spec_payload.get("memory_limit_mb", 256)),
        io_mode=spec_payload.get("io_mode", "stdin"),
        entry_point=spec_payload.get("entry_point", ""),
    )
    tests = TestSuite(
        visible_tests=_cases_from_dict(tests_payload.get("visible_tests", []), "visible"),
        reward_tests=_cases_from_dict(tests_payload.get("reward_tests", []), "reward"),
        eval_tests=_cases_from_dict(tests_payload.get("eval_tests", []), "eval"),
    )
    oracle = OracleMetadata(
        difficulty=oracle_payload.get("difficulty", ""),
        tags=_flat_tags(oracle_payload.get("tags", [])),
        expected_complexity=oracle_payload.get("expected_complexity", ""),
        reference_solution=oracle_payload.get("reference_solution", ""),
        source=oracle_payload.get("source", ""),
        url=oracle_payload.get("url", ""),
        solutions=[
            OracleSolution(
                language=solution.get("language", "unknown"),
                code=solution.get("code", ""),
                verified=bool(solution.get("verified", False)),
            )
            for solution in oracle_payload.get("solutions", [])
            if solution.get("code")
        ],
    )
    return ProblemBundle(spec=spec, tests=tests, oracle=oracle)


def problem_bundle_to_dict(bundle: ProblemBundle) -> dict[str, Any]:
    return {
        "problem": {
            "id": bundle.spec.id,
            "title": bundle.spec.title,
            "statement": bundle.spec.statement,
            "input_format": bundle.spec.input_format,
            "output_format": bundle.spec.output_format,
            "constraints": bundle.spec.constraints,
            "language": bundle.spec.language,
            "time_limit_sec": bundle.spec.time_limit_sec,
            "memory_limit_mb": bundle.spec.memory_limit_mb,
            "io_mode": bundle.spec.io_mode,
            "entry_point": bundle.spec.entry_point,
        },
        "tests": {
            "visible_tests": [_case_to_dict(case) for case in bundle.tests.visible_tests],
            "reward_tests": [_case_to_dict(case) for case in bundle.tests.reward_tests],
            "eval_tests": [_case_to_dict(case) for case in bundle.tests.eval_tests],
        },
        "oracle": {
            "difficulty": bundle.oracle.difficulty,
            "tags": bundle.oracle.tags,
            "expected_complexity": bundle.oracle.expected_complexity,
            "reference_solution": bundle.oracle.reference_solution,
            "source": bundle.oracle.source,
            "url": bundle.oracle.url,
            "solutions": [
                {"language": solution.language, "code": solution.code, "verified": solution.verified}
                for solution in bundle.oracle.solutions
            ],
        },
    }


def _cases_from_dict(payloads: list[dict[str, Any]], prefix: str) -> list[TestCase]:
    cases = []
    for index, payload in enumerate(payloads, start=1):
        cases.append(
            TestCase(
                stdin=payload["stdin"],
                expected_stdout=payload["expected_stdout"],
                id=payload.get("id") or payload.get("name") or f"{prefix}-{index:04d}",
                timeout_sec=float(payload["timeout_sec"]) if payload.get("timeout_sec") is not None else None,
            )
        )
    return cases


def _case_to_dict(case: TestCase) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": case.id,
        "stdin": case.stdin,
        "expected_stdout": case.expected_stdout,
    }
    if case.timeout_sec is not None:
        payload["timeout_sec"] = case.timeout_sec
    return payload


def _flat_tags(raw: Any) -> list[str]:
    tags: list[str] = []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, list):
                tags.extend(str(sub) for sub in item)
            else:
                tags.append(str(item))
    return list(dict.fromkeys(tag for tag in tags if tag))
