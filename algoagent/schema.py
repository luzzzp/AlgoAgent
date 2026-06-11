from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class AgentStatus(str, Enum):
    SOLVED = "SOLVED"
    FAILED = "FAILED"


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProblemSpec:
    id: str
    title: str
    statement: str
    input_format: str
    output_format: str
    constraints: list[str]
    language: str = "cpp17"
    time_limit_sec: float = 2.0
    memory_limit_mb: int = 256

    def prompt(self) -> str:
        constraints = "\n".join(f"- {item}" for item in self.constraints)
        return (
            f"Title: {self.title}\n\n"
            f"{self.statement}\n\n"
            f"Input format:\n{self.input_format}\n\n"
            f"Output format:\n{self.output_format}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Language: {self.language}\n"
            f"Time limit: {self.time_limit_sec:g} seconds\n"
            f"Memory limit: {self.memory_limit_mb} MB\n\n"
            "Return a C++17 solution and structured time/space complexity analysis."
        )


@dataclass(frozen=True)
class TestCase:
    stdin: str
    expected_stdout: str
    id: str = ""
    timeout_sec: float | None = None


@dataclass(frozen=True)
class TestSuite:
    repair_tests: list[TestCase] = field(default_factory=list)
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

    def best_solution(self, language: str = "cpp17") -> str:
        for solution in self.solutions:
            if solution.language == language and solution.verified:
                return solution.code
        for solution in self.solutions:
            if solution.language == language:
                return solution.code
        if language == "cpp17":
            return self.reference_solution
        return ""


@dataclass(frozen=True)
class ProblemBundle:
    spec: ProblemSpec
    tests: TestSuite
    oracle: OracleMetadata = field(default_factory=OracleMetadata)


@dataclass(frozen=True)
class ComplexityEstimate:
    time_complexity: str
    space_complexity: str
    dominant_variables: dict[str, int] = field(default_factory=dict)
    estimated_operations: int | None = None
    estimated_memory_bytes: int | None = None


@dataclass(frozen=True)
class ResourceVerdict:
    time_status: CheckStatus
    memory_status: CheckStatus
    estimated_operations: int | None
    operation_budget: int
    estimated_memory_bytes: int | None
    memory_budget_bytes: int
    reasons: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.time_status == CheckStatus.FAIL or self.memory_status == CheckStatus.FAIL

    @property
    def unknown(self) -> bool:
        return self.time_status == CheckStatus.UNKNOWN or self.memory_status == CheckStatus.UNKNOWN


@dataclass(frozen=True)
class ExecutionResult:
    compiled: bool
    passed: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    compile_error: str = ""


@dataclass(frozen=True)
class TestRun:
    test_id: str
    suite: str
    passed: bool
    expected: str
    actual: str
    stderr: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class CandidateResult:
    compiled: bool
    compile_error: str
    tests: list[TestRun]

    @property
    def all_passed(self) -> bool:
        return self.compiled and all(test.passed for test in self.tests)


@dataclass(frozen=True)
class AgentTrace:
    attempt: int
    stage: str
    message: str


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    complexity: ComplexityEstimate
    resource_verdict: ResourceVerdict
    repair_result: CandidateResult | None = None
    generated_code: str | None = None


@dataclass(frozen=True)
class AgentResult:
    problem_id: str
    status: AgentStatus
    attempts: int
    code: str | None
    explanation: str | None
    complexity: ComplexityEstimate | None
    resource_verdict: ResourceVerdict
    failure_reason: str | None
    diagnostic_summary: str | None
    attempt_records: list[AttemptRecord]
    held_out_result: CandidateResult | None
    traces: list[AgentTrace]

    @property
    def solved(self) -> bool:
        return self.status == AgentStatus.SOLVED


def load_problem(path: str | Path) -> ProblemBundle:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    try:
        return problem_bundle_from_dict(payload)
    except KeyError as exc:
        raise ValueError(f"{source} is not an AlgoAgent problem JSON; missing key {exc!s}.") from exc


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
        language=spec_payload.get("language", "cpp17"),
        time_limit_sec=float(spec_payload.get("time_limit_sec", 2.0)),
        memory_limit_mb=int(spec_payload.get("memory_limit_mb", 256)),
    )
    tests = TestSuite(
        repair_tests=_cases_from_dict(tests_payload.get("repair_tests", []), "repair"),
        eval_tests=_cases_from_dict(tests_payload.get("eval_tests", []), "eval"),
    )
    oracle = OracleMetadata(
        difficulty=oracle_payload.get("difficulty", ""),
        tags=list(oracle_payload.get("tags", [])),
        expected_complexity=oracle_payload.get("expected_complexity", ""),
        reference_solution=oracle_payload.get("reference_solution", ""),
        solutions=[
            OracleSolution(
                language=solution.get("language", "unknown"),
                code=solution.get("code", ""),
                verified=bool(solution.get("verified", False)),
            )
            for solution in oracle_payload.get("solutions", [])
            if solution.get("code")
        ],
        source=oracle_payload.get("source", ""),
        url=oracle_payload.get("url", ""),
    )
    return ProblemBundle(spec=spec, tests=tests, oracle=oracle)


def _cases_from_dict(payloads: list[dict[str, Any]], prefix: str) -> list[TestCase]:
    cases: list[TestCase] = []
    for index, payload in enumerate(payloads, start=1):
        cases.append(
            TestCase(
                id=payload.get("id") or payload.get("name") or f"{prefix}-{index:04d}",
                stdin=payload["stdin"],
                expected_stdout=payload["expected_stdout"],
                timeout_sec=(
                    float(payload["timeout_sec"]) if payload.get("timeout_sec") is not None else None
                ),
            )
        )
    return cases


def normalize_output(text: str | bytes) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)
