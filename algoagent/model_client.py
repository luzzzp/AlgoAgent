from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from algoagent.schema import ComplexityEstimate, ProblemSpec


@dataclass(frozen=True)
class ModelResponse:
    draft_reasoning: str
    code: str
    complexity: ComplexityEstimate


class ModelClient(Protocol):
    def generate_solution(
        self,
        problem: ProblemSpec,
        feedback: str | None,
        attempt: int,
    ) -> ModelResponse:
        ...

    def explain_solution(
        self,
        problem: ProblemSpec,
        verified_code: str,
        complexity: ComplexityEstimate,
    ) -> str:
        ...


class RuleBasedModel:
    """Deterministic offline stub that never reads oracle metadata."""

    def generate_solution(
        self,
        problem: ProblemSpec,
        feedback: str | None,
        attempt: int,
    ) -> ModelResponse:
        if problem.id == "two_sum":
            if attempt == 1:
                return _response(
                    "Use a hash map, but this first draft has a syntax mistake.",
                    _two_sum_buggy(),
                    "O(n)",
                    "O(n)",
                )
            return _response("Use a hash map to find complements.", _two_sum_correct(), "O(n)", "O(n)")
        if problem.id == "balanced_parentheses":
            if attempt == 1:
                return _response(
                    "Track balance, but this first draft misses negative prefixes.",
                    _balanced_parentheses_buggy(),
                    "O(n)",
                    "O(1)",
                )
            return _response(
                "Track balance and reject negative prefixes.",
                _balanced_parentheses_correct(),
                "O(n)",
                "O(1)",
            )
        if problem.id == "longest_increasing_subsequence":
            return _response(
                "Maintain the minimum possible tail for each subsequence length.",
                _lis_correct(),
                "O(n log n)",
                "O(n)",
            )
        return _response("No specialized rule is available.", _fallback_cpp(), "unknown", "unknown")

    def explain_solution(
        self,
        problem: ProblemSpec,
        verified_code: str,
        complexity: ComplexityEstimate,
    ) -> str:
        explanations = {
            "two_sum": "遍历数组并用哈希表记录已经出现的数值及下标；若补数已经出现，则输出对应下标。",
            "balanced_parentheses": "维护当前括号余额；余额一旦为负立即失败，遍历结束后余额为零才合法。",
            "longest_increasing_subsequence": "维护每个长度的递增子序列能够取得的最小结尾，并用二分更新。",
        }
        return explanations.get(problem.id, "代码已通过全部修复测试和留出测试。")


def _response(reasoning: str, code: str, time: str, space: str) -> ModelResponse:
    return ModelResponse(
        draft_reasoning=reasoning,
        code=code,
        complexity=ComplexityEstimate(time_complexity=time, space_complexity=space),
    )


def _two_sum_buggy() -> str:
    return """#include <bits/stdc++.h>
using namespace std;
int main() {
    int n; long long target;
    cin >> n >> target;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    unordered_map<long long, int> seen;
    for (int i = 0; i < n; ++i) {
        long long need = target - a[i];
        if (seen.count(need)) {
            cout << seen[need] << " " << i << "\\n"
            return 0;
        }
        seen[a[i]] = i;
    }
    cout << "-1 -1\\n";
}
"""


def _two_sum_correct() -> str:
    return """#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n; long long target;
    if (!(cin >> n >> target)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    unordered_map<long long, int> seen;
    for (int i = 0; i < n; ++i) {
        long long need = target - a[i];
        auto it = seen.find(need);
        if (it != seen.end()) {
            cout << it->second << ' ' << i << '\\n';
            return 0;
        }
        if (!seen.count(a[i])) seen[a[i]] = i;
    }
    cout << "-1 -1\\n";
    return 0;
}
"""


def _balanced_parentheses_buggy() -> str:
    return """#include <bits/stdc++.h>
using namespace std;
int main() {
    string s; cin >> s;
    int balance = 0;
    for (char c : s) balance += c == '(' ? 1 : -1;
    cout << (balance == 0 ? "YES" : "NO") << '\\n';
}
"""


def _balanced_parentheses_correct() -> str:
    return """#include <bits/stdc++.h>
using namespace std;
int main() {
    string s;
    if (!(cin >> s)) return 0;
    int balance = 0;
    for (char c : s) {
        balance += c == '(' ? 1 : -1;
        if (balance < 0) {
            cout << "NO\\n";
            return 0;
        }
    }
    cout << (balance == 0 ? "YES" : "NO") << '\\n';
}
"""


def _lis_correct() -> str:
    return """#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> tails;
    for (int i = 0; i < n; ++i) {
        long long x; cin >> x;
        auto it = lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) tails.push_back(x);
        else *it = x;
    }
    cout << tails.size() << '\\n';
}
"""


def _fallback_cpp() -> str:
    return "#include <bits/stdc++.h>\nusing namespace std;\nint main(){return 0;}\n"

