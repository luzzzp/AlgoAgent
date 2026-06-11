from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoagent.schema import ProblemBundle, load_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy only problems that have a verified C++17 oracle solution.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Copy at most this many verified C++ problems; 0 means all.")
    args = parser.parse_args()

    report = filter_verified_cpp_problems(Path(args.problems), Path(args.out_dir), limit=args.limit)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def filter_verified_cpp_problems(problems_dir: Path, out_dir: Path, limit: int = 0) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "stage": "filter_verified_cpp_problems",
        "total": 0,
        "copied": 0,
        "skipped_no_verified_cpp": 0,
        "copied_manifests": 0,
    }
    for path in sorted(problems_dir.glob("*.json")):
        if path.name.startswith("_"):
            shutil.copy2(path, out_dir / path.name)
            stats["copied_manifests"] += 1
            continue
        if limit and stats["copied"] >= limit:
            continue
        stats["total"] += 1
        bundle = load_problem(path)
        if not has_verified_cpp(bundle):
            stats["skipped_no_verified_cpp"] += 1
            continue
        shutil.copy2(path, out_dir / path.name)
        stats["copied"] += 1

    (out_dir / "_verified_cpp_filter_manifest.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def has_verified_cpp(bundle: ProblemBundle) -> bool:
    return any(solution.language == "cpp17" and solution.verified for solution in bundle.oracle.solutions)


if __name__ == "__main__":
    main()
