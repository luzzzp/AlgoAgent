from __future__ import annotations

from dataclasses import replace
import math
import re

from algoagent.schema import CheckStatus, ComplexityEstimate, ProblemSpec, ResourceVerdict


class ComplexityFeasibilityChecker:
    """Heuristically checks whether a C++ solution fits theoretical limits."""

    def __init__(self, operations_per_second: int = 100_000_000):
        self.operations_per_second = operations_per_second

    def check(
        self,
        problem: ProblemSpec,
        estimate: ComplexityEstimate,
        code: str,
    ) -> tuple[ComplexityEstimate, ResourceVerdict]:
        variables = dict(self.extract_variable_bounds(problem.constraints))
        variables.update({key.lower(): value for key, value in estimate.dominant_variables.items()})

        operations = estimate.estimated_operations
        if operations is None:
            operations = self._estimate_expression(estimate.time_complexity, variables)
        memory_bytes = estimate.estimated_memory_bytes
        if memory_bytes is None:
            code_memory = self._estimate_memory_from_code(code, variables)
            complexity_memory = self._estimate_space_complexity(estimate.space_complexity, variables)
            known_memory = [value for value in (code_memory, complexity_memory) if value is not None]
            memory_bytes = max(known_memory) if known_memory else None

        operation_budget = int(problem.time_limit_sec * self.operations_per_second)
        memory_budget = problem.memory_limit_mb * 1024 * 1024
        time_status = _status_for_estimate(operations, operation_budget)
        memory_status = _status_for_estimate(memory_bytes, memory_budget)
        reasons: list[str] = []
        if time_status == CheckStatus.FAIL:
            reasons.append(
                f"THEORETICAL_TLE: estimated {operations} operations exceeds budget {operation_budget}."
            )
        elif time_status == CheckStatus.UNKNOWN:
            reasons.append(f"Time complexity could not be estimated from {estimate.time_complexity!r}.")
        if memory_status == CheckStatus.FAIL:
            reasons.append(
                f"THEORETICAL_MLE: estimated {memory_bytes} bytes exceeds budget {memory_budget}."
            )
        elif memory_status == CheckStatus.UNKNOWN:
            reasons.append(f"Memory usage could not be estimated from {estimate.space_complexity!r}.")

        resolved = replace(
            estimate,
            dominant_variables=variables,
            estimated_operations=operations,
            estimated_memory_bytes=memory_bytes,
        )
        return resolved, ResourceVerdict(
            time_status=time_status,
            memory_status=memory_status,
            estimated_operations=operations,
            operation_budget=operation_budget,
            estimated_memory_bytes=memory_bytes,
            memory_budget_bytes=memory_budget,
            reasons=reasons,
        )

    def extract_variable_bounds(self, constraints: list[str]) -> dict[str, int]:
        bounds: dict[str, int] = {}
        for constraint in constraints:
            text = constraint.lower().replace(",", " ")
            for variable, value in re.findall(r"\b([a-z][a-z0-9_]*)\s*<=\s*([0-9]+(?:e[0-9]+)?)", text):
                bounds[variable] = max(bounds.get(variable, 0), _parse_number(value))
            for variables, value in re.findall(
                r"(?:[0-9]+(?:e[0-9]+)?\s*<=\s*)?([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)*)\s*<=\s*([0-9]+(?:e[0-9]+)?)",
                text,
            ):
                for variable in variables.split():
                    bounds[variable] = max(bounds.get(variable, 0), _parse_number(value))
        return bounds

    def _estimate_expression(self, complexity: str, variables: dict[str, int]) -> int | None:
        expression = _complexity_expression(complexity)
        if not expression:
            return None
        if expression == "1":
            return 1
        if "!" in expression:
            variable = _first_known_variable(expression, variables)
            return math.factorial(variable) if variable is not None and variable <= 20 else None
        exponential = re.search(r"2\^([a-z][a-z0-9_]*)", expression)
        if exponential:
            value = variables.get(exponential.group(1))
            return 2**value if value is not None and value <= 60 else None

        terms = [term.strip() for term in expression.split("+")]
        values: list[int] = []
        for term in terms:
            value = self._estimate_product(term, variables)
            if value is None:
                return None
            values.append(value)
        return sum(values)

    def _estimate_product(self, expression: str, variables: dict[str, int]) -> int | None:
        expression = expression.replace(" ", "*")
        expression = re.sub(r"\*+", "*", expression).strip("*")
        if not expression:
            return None
        value = 1.0
        for token in expression.split("*"):
            if token in {"", "1"}:
                continue
            log_match = re.fullmatch(r"log(?:2)?\(?([a-z][a-z0-9_]*)\)?", token)
            if log_match:
                bound = variables.get(log_match.group(1))
                if bound is None:
                    return None
                value *= max(1, math.log2(max(2, bound)))
                continue
            power_match = re.fullmatch(r"([a-z][a-z0-9_]*)\^([0-9]+)", token)
            if power_match:
                bound = variables.get(power_match.group(1))
                if bound is None:
                    return None
                value *= bound ** int(power_match.group(2))
                continue
            if token.isdigit():
                value *= int(token)
                continue
            bound = variables.get(token)
            if bound is None:
                return None
            value *= bound
        return int(math.ceil(value))

    def _estimate_space_complexity(self, complexity: str, variables: dict[str, int]) -> int | None:
        units = self._estimate_expression(complexity, variables)
        if units is None:
            return None
        return units * 8

    def _estimate_memory_from_code(self, code: str, variables: dict[str, int]) -> int | None:
        total = 0
        found = False
        type_sizes = {
            "char": 1,
            "bool": 1,
            "short": 2,
            "int": 4,
            "float": 4,
            "long long": 8,
            "double": 8,
        }
        array_pattern = re.compile(
            r"\b(char|bool|short|int|float|long long|double)\s+\w+\s*((?:\[[a-z0-9_]+\])+)",
            re.I,
        )
        for type_name, dimensions in array_pattern.findall(code):
            count = 1
            for dimension in re.findall(r"\[([a-z0-9_]+)\]", dimensions.lower()):
                size = int(dimension) if dimension.isdigit() else variables.get(dimension)
                if size is None:
                    count = 0
                    break
                count *= size
            if count:
                total += type_sizes[type_name.lower()] * count
                found = True

        vector_pattern = re.compile(
            r"\bvector\s*<\s*(char|bool|short|int|float|long long|double)\s*>\s+\w+\s*\(\s*([a-z0-9_]+)\s*\)",
            re.I,
        )
        for type_name, dimension in vector_pattern.findall(code):
            size = int(dimension) if dimension.isdigit() else variables.get(dimension.lower())
            if size is not None:
                total += type_sizes[type_name.lower()] * size
                found = True

        matrix_pattern = re.compile(
            r"vector\s*<\s*vector\s*<\s*(char|bool|short|int|float|long long|double)\s*>\s*>\s+\w+\s*"
            r"\(\s*([a-z0-9_]+)\s*,\s*vector\s*<[^>]+>\s*\(\s*([a-z0-9_]+)",
            re.I,
        )
        for type_name, first, second in matrix_pattern.findall(code):
            rows = int(first) if first.isdigit() else variables.get(first.lower())
            cols = int(second) if second.isdigit() else variables.get(second.lower())
            if rows is not None and cols is not None:
                total += type_sizes[type_name.lower()] * rows * cols
                found = True
        return total if found else None


def _status_for_estimate(estimate: int | None, budget: int) -> CheckStatus:
    if estimate is None:
        return CheckStatus.UNKNOWN
    return CheckStatus.PASS if estimate <= budget else CheckStatus.FAIL


def _parse_number(value: str) -> int:
    return int(float(value))


def _complexity_expression(complexity: str) -> str:
    normalized = complexity.lower().replace("²", "^2").replace("³", "^3")
    normalized = normalized.replace("time", "").replace("space", "")
    match = re.search(r"o\s*\(([^)]+)\)", normalized)
    if not match:
        return ""
    expression = match.group(1).strip()
    expression = re.sub(r"\blog\s+([a-z])", r"log(\1)", expression)
    return expression


def _first_known_variable(expression: str, variables: dict[str, int]) -> int | None:
    for variable in re.findall(r"[a-z][a-z0-9_]*", expression):
        if variable in variables:
            return variables[variable]
    return None
