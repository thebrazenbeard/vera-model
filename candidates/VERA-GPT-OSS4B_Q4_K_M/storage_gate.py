"""Two-job independent Hugging Face Storage Bucket persistence check.

Job A writes marker, exits. Job B must mount SAME bucket and read marker.
No training is performed. Use distinct job IDs and record both receipts.
"""
import argparse
import hashlib
import json
from pathlib import Path


def check(root: Path, marker: str, mode: str) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("mount missing or symlink")
    if not marker or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in marker):
        raise ValueError("invalid marker")
    path = root / ("vera-gptoss4b-storage-" + marker + ".json")
    data = json.dumps({"schema": "vera-model.storage-gate.v1", "marker": marker}, sort_keys=True)
    if mode == "write":
        with path.open("x", encoding="utf8") as f:
            f.write(data)
            f.flush()
    elif mode != "read":
        raise ValueError("bad mode")
    if path.read_text(encoding="utf8") != data:
        raise RuntimeError("storage marker mismatch")
    return {"mode": mode, "verified": True, "sha256": hashlib.sha256(data.encode()).hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--marker", required=True)
    p.add_argument("--mode", choices=("write", "read"), required=True)
    args = p.parse_args()
    print(json.dumps(check(args.root, args.marker, args.mode)))


if __name__ == "__main__":
    main()
