from __future__ import annotations

from dataclasses import dataclass, field
import re

from algoagent.executor import PythonExecutor
from algoagent.schema import ExecutionReport, ProblemBundle


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    syntax: float = 0.0
    visible: float = 0.0
    reward_tests: float = 0.0
    stability: float = 0.0
    format: float = 0.0
    explanation: float = 0.0
    penalties: float = 0.0
    reasons: list[str] = field(default_factory=list)


def score_completion(
    text: str,
    bundle: ProblemBundle,
    executor: PythonExecutor | None = None,
) -> RewardBreakdown:
    executor = executor or PythonExecutor()
    code = extract_python_code(text)
    if not code:
        return RewardBreakdown(total=-0.2, penalties=-0.2, reasons=["missing_python_code_block"])

    visible_result = executor.evaluate(
        code,
        bundle.tests.visible_tests,
        default_timeout_sec=bundle.spec.time_limit_sec,
        suite_name="visible",
    )
    reward_result = executor.evaluate(
        code,
        bundle.tests.reward_tests,
        default_timeout_sec=bundle.spec.time_limit_sec,
        suite_name="reward",
    )
    return compute_reward(text, code, visible_result, reward_result)


def compute_reward(
    text: str,
    code: str,
    visible_result: ExecutionReport,
    reward_result: ExecutionReport,
) -> RewardBreakdown:
    reasons: list[str] = []
    syntax = 0.1 if visible_result.syntax_valid else 0.0
    visible = 0.2 if visible_result.all_passed else 0.2 * visible_result.pass_rate
    reward_tests = 0.5 * reward_result.pass_rate
    any_timeout = _any_timeout(visible_result) or _any_timeout(reward_result)
    any_runtime_error = _any_runtime_error(visible_result) or _any_runtime_error(reward_result)
    stability = 0.1 if visible_result.syntax_valid and not any_timeout and not any_runtime_error else 0.0
    format_score = _format_reward(text)
    explanation = _gated_explanation_reward(text, reward_result.pass_rate)
    penalties = 0.0
    if any_timeout:
        penalties -= 0.5
        reasons.append("timeout")
    if any_runtime_error:
        penalties -= 0.3
        reasons.append("runtime_error")
    if _looks_like_sample_hardcode(code):
        penalties -= 0.5
        reasons.append("possible_sample_hardcode")
    if _has_truncated_output(visible_result) or _has_truncated_output(reward_result):
        penalties -= 0.3
        reasons.append("output_too_long")
    total = syntax + visible + reward_tests + stability + format_score + explanation + penalties
    return RewardBreakdown(
        total=round(total, 6),
        syntax=syntax,
        visible=visible,
        reward_tests=reward_tests,
        stability=stability,
        format=format_score,
        explanation=explanation,
        penalties=penalties,
        reasons=reasons,
    )


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python|py)\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return ""


def _format_reward(text: str) -> float:
    score = 0.0
    if re.search(r"solution\s+explanation\s*:", text, flags=re.I):
        score += 0.025
    if re.search(r"[\u4e00-\u9fff]", text):
        score += 0.025
    if re.search(r"time\s+complexity\s*:", text, flags=re.I):
        score += 0.025
    if re.search(r"space\s+complexity\s*:", text, flags=re.I):
        score += 0.025
    return score


def _gated_explanation_reward(text: str, reward_pass_rate: float) -> float:
    if reward_pass_rate < 0.8:
        return 0.02 if re.search(r"[\u4e00-\u9fff]", text) else 0.0
    explanation = _extract_explanation(text)
    if not explanation:
        return 0.0
    score = 0.0
    if len(explanation) >= 30:
        score += 0.06
    if re.search(r"复杂度|时间|空间|算法|遍历|动态规划|二分|贪心|图|树", explanation):
        score += 0.08
    if "根据题意" not in explanation or len(explanation) > 80:
        score += 0.06
    return min(score, 0.2)


def _extract_explanation(text: str) -> str:
    match = re.search(
        r"solution\s+explanation\s*:\s*(.*?)(?:time\s+complexity\s*:|$)",
        text,
        flags=re.I | re.S,
    )
    return match.group(1).strip() if match else ""


def _any_timeout(report: ExecutionReport) -> bool:
    return any(run.timed_out for run in report.runs)


def _any_runtime_error(report: ExecutionReport) -> bool:
    return any((run.returncode or 0) != 0 and not run.timed_out for run in report.runs)


def _has_truncated_output(report: ExecutionReport) -> bool:
    return any(run.output_truncated for run in report.runs)


def _looks_like_sample_hardcode(code: str) -> bool:
    return code.count("input()") == 0 and code.count("print(") >= 3

