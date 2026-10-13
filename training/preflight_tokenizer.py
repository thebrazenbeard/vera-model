"""Lightweight exact-tokenizer training preflight; no torch/CUDA imports.

Verifies pinned donor tokenizer bytes, immutable Vera SFT+heldout corpus,
assistant-only masking and all 68 examples using the Rust tokenizer backend.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from train_vera_lora import (
    assistant_only_tokens, verify_corpus, verify_tokenizer_assets, digest,
)


class RustTokenizerAdapter:
    bos_token = "<s>"
    eos_token = "</s>"
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        return {"input_ids": self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens).ids}


def validate_sft_tokenization(tokenizer, rows: list[dict], limit: int) -> dict:
    counts = []
    for row in rows:
        ids, labels = assistant_only_tokens(
            tokenizer, row["messages"], limit)
        if len(ids) != len(labels):
            raise ValueError("token/label length mismatch for " + row["id"])
        if not any(x == -100 for x in labels) or not any(x != -100 for x in labels):
            raise ValueError("missing prompt or assistant labels for " + row["id"])
        counts.append({"id": row["id"], "tokens": len(ids),
                       "learned_tokens": sum(x != -100 for x in labels)})
    if len(counts) != 68:
        raise ValueError("not the exact pilot training window")
    return {
        "rows_checked": len(counts),
        "min_tokens": min(x["tokens"] for x in counts),
        "max_tokens": max(x["tokens"] for x in counts),
        "total_tokens": sum(x["tokens"] for x in counts),
        "total_learned_tokens": sum(x["learned_tokens"] for x in counts),
        "all_assistant_only_masking_valid": True,
        "all_512_token_limits_passed": max(x["tokens"] for x in counts) <= limit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise SystemExit("HOLD: exact output receipt exists or parent missing")
    root = Path(__file__).resolve().parent
    recipe_file = root / "qlora-recipe-v1.json"
    pins_file = root / "donor-checkpoint-manifest.json"
    recipe = json.loads(recipe_file.read_text(encoding="utf8"))
    pins = json.loads(pins_file.read_text(encoding="utf8"))
    try:
        count = verify_tokenizer_assets(args.tokenizer_dir,
                                        pins["tokenizer_assets"])
        from tokenizers import Tokenizer
        raw = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
        for token, idx in (("<s>", 1), ("</s>", 2),
                           ("[INST]", 3), ("[/INST]", 4)):
            if raw.token_to_id(token) != idx:
                raise ValueError("token vocabulary incompatible: " + token)
        rows, manifest = verify_corpus(args.corpus, recipe)
        result = validate_sft_tokenization(
            RustTokenizerAdapter(raw), rows,
            int(recipe["trainer"]["max_sequence_tokens"]))
        head = subprocess.run(
            ["git", "-C", str(root.parent), "rev-parse", "HEAD"],
            check=True, text=True, capture_output=True).stdout.strip()
        receipt = {
            "schema": "VERA_DOLPHIN_24B_EXACT_TOKENIZER_PREFLIGHT_V1",
            "status": "PASS_TOKENIZER_ONLY",
            "source_head": head,
            "donor_revision": pins["exact_revision"],
            "tokenizer_assets_verified": count,
            "tokenizer_sha256": pins["tokenizer_assets"]["tokenizer.json"]["sha256"],
            "corpus_manifest_sha256": recipe["data"]["manifest_sha256"],
            "corpus_sft_sha256": recipe["data"]["files_sha256"]["vera_sft.jsonl"],
            "trainer_recipe_sha256": digest(recipe_file),
            "checks": result,
            "training_executed": False,
            "optimizer_steps": 0,
            "adapter_created": False,
            "qualification": "UNVERIFIED",
        }
    except (RuntimeError, ValueError, OSError, KeyError,
            subprocess.SubprocessError) as err:
        print(json.dumps({"status": "HOLD", "error": str(err)}))
        return 2
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf8")
    print(json.dumps({"status": receipt["status"],
                      "checks": receipt["checks"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
