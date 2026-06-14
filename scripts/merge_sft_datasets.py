from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AlgoAgent SFT jsonl files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    per_file: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as output:
        for item in args.inputs:
            path = Path(item)
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    _validate(record, path)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    total += 1
            per_file[str(path)] = count
    report = {"stage": "merge_sft_datasets", "records": total, "inputs": per_file, "out": str(out_path)}
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _validate(record: dict, path: Path) -> None:
    missing = [key for key in ("instruction", "input", "output") if not record.get(key)]
    if missing:
        raise ValueError(f"{path} has record missing fields: {missing}")


if __name__ == "__main__":
    main()
