from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.executor import CppExecutor
from algoagent.schema import OracleSolution, ProblemBundle, load_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate verified Python oracle solutions to C++17 and validate them.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--backend", choices=["hf", "mock"], default="hf")
    parser.add_argument("--candidates-per-problem", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Translate at most this many problem files; 0 means no limit.")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()

    translator = build_translator(args.backend, args.model)
    report = translate_python_to_cpp(
        Path(args.problems),
        Path(args.out_dir),
        translator,
        candidates_per_problem=args.candidates_per_problem,
        limit=args.limit,
        compiler=args.compiler,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def translate_python_to_cpp(
    problems_dir: Path,
    out_dir: Path,
    translator,
    candidates_per_problem: int = 2,
    limit: int = 0,
    compiler: str = "g++",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    executor = CppExecutor(compiler=compiler)
    stats = {
        "total": 0,
        "attempted": 0,
        "verified_cpp": 0,
        "no_verified_python": 0,
        "translation_failed": 0,
        "copied_manifests": 0,
    }
    for path in sorted(problems_dir.glob("*.json")):
        if path.name.startswith("_"):
            shutil.copy2(path, out_dir / path.name)
            stats["copied_manifests"] += 1
            continue
        if limit and stats["total"] >= limit:
            continue
        stats["total"] += 1
        bundle = load_problem(path)
        py_solution = first_verified_python(bundle)
        if not py_solution:
            stats["no_verified_python"] += 1
            write_bundle(out_dir / path.name, bundle)
            continue
        stats["attempted"] += 1
        updated, ok = translate_bundle(bundle, py_solution, translator, executor, candidates_per_problem)
        if ok:
            stats["verified_cpp"] += 1
        else:
            stats["translation_failed"] += 1
        write_bundle(out_dir / path.name, updated)

    manifest = {"stage": "translate_python_to_cpp", **stats}
    (out_dir / "_cpp_translation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def translate_bundle(
    bundle: ProblemBundle,
    python_code: str,
    translator,
    executor: CppExecutor,
    candidates_per_problem: int = 2,
) -> tuple[ProblemBundle, bool]:
    tests = [*bundle.tests.repair_tests, *bundle.tests.eval_tests]
    solutions = list(bundle.oracle.solutions)
    for candidate_index in range(candidates_per_problem):
        candidate = translator.translate(bundle, python_code, candidate_index)
        cpp = extract_cpp_code(candidate)
        if not cpp:
            continue
        result = executor.evaluate(cpp, tests, default_timeout_sec=bundle.spec.time_limit_sec, suite_name="cpp_translation")
        if result.all_passed:
            solutions.append(OracleSolution(language="cpp17", code=cpp, verified=True))
            updated = replace_oracle_solutions(bundle, solutions, reference_solution=cpp)
            return updated, True
    return replace_oracle_solutions(bundle, solutions), False


def first_verified_python(bundle: ProblemBundle) -> str:
    for solution in bundle.oracle.solutions:
        if solution.language == "python3" and solution.verified:
            return solution.code
    return ""


def replace_oracle_solutions(
    bundle: ProblemBundle,
    solutions: list[OracleSolution],
    reference_solution: str | None = None,
) -> ProblemBundle:
    return ProblemBundle(
        spec=bundle.spec,
        tests=bundle.tests,
        oracle=type(bundle.oracle)(
            difficulty=bundle.oracle.difficulty,
            tags=bundle.oracle.tags,
            expected_complexity=bundle.oracle.expected_complexity,
            reference_solution=reference_solution if reference_solution is not None else bundle.oracle.reference_solution,
            solutions=solutions,
            source=bundle.oracle.source,
            url=bundle.oracle.url,
        ),
    )


def build_translator(backend: str, model_name: str):
    if backend == "mock":
        return MockTranslator()
    return HuggingFaceTranslator(model_name)


class HuggingFaceTranslator:
    def __init__(self, model_name: str, max_new_tokens: int = 4096):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, OSError) as exc:
            raise SystemExit(
                "HF translation requires a working PyTorch + Transformers environment. "
                "Your current Python cannot import torch. For this project, run `--backend hf` on a Linux GPU "
                "server, or repair/reinstall PyTorch in the current environment. "
                "For local smoke tests, use `--backend mock`."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    def translate(self, bundle: ProblemBundle, python_code: str, candidate_index: int) -> str:
        prompt = (
            "Translate the verified Python solution into a standalone C++17 stdin/stdout program.\n"
            "Preserve the algorithm and satisfy the resource limits.\n\n"
            f"{bundle.spec.prompt()}\n\n"
            f"Python solution:\n```python\n{python_code}\n```\n\n"
            "Return only one C++17 program inside a cpp markdown code block."
        )
        messages = [
            {"role": "system", "content": "You are a careful competitive programming translator."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=candidate_index > 0,
                temperature=0.2 if candidate_index > 0 else 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class MockTranslator:
    def translate(self, bundle: ProblemBundle, python_code: str, candidate_index: int) -> str:
        if "print(n)" in python_code:
            return "```cpp\n#include <bits/stdc++.h>\nusing namespace std;\nint main(){long long n; if(cin>>n) cout<<n<<'\\n';}\n```"
        return "```cpp\n#include <bits/stdc++.h>\nusing namespace std;\nint main(){return 0;}\n```"


def extract_cpp_code(text: str) -> str:
    match = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    include_at = text.find("#include")
    return text[include_at:].strip() if include_at >= 0 else ""


def write_bundle(path: Path, bundle: ProblemBundle) -> None:
    path.write_text(json.dumps(bundle_to_json(bundle), indent=2, ensure_ascii=False), encoding="utf-8")


def bundle_to_json(bundle: ProblemBundle) -> dict[str, object]:
    return {
        "problem": asdict(bundle.spec),
        "tests": {
            "repair_tests": [asdict(test) for test in bundle.tests.repair_tests],
            "eval_tests": [asdict(test) for test in bundle.tests.eval_tests],
        },
        "oracle": asdict(bundle.oracle),
    }


if __name__ == "__main__":
    main()
