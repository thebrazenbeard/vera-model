"""Exact-case behavioral baseline for Vera voice with no runtime persona by default.

Responses are evidence for independent review, never self-qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from local_model import _request, read_key, LocalModelHold


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cases(path: Path, selected_ids: list[str]) -> list[dict]:
    selected = set(selected_ids)
    data = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()
            if line.strip()]
    if not data or not selected:
        raise ValueError("cases and case selection are required")
    available = {row["id"]: row for row in data}
    if len(available) != len(data) or len(selected_ids) != len(selected):
        raise ValueError("duplicate case identifier")
    missing = sorted(selected - available.keys())
    if missing:
        raise ValueError(f"unknown heldout cases: {missing}")
    for row in data:
        if row["training_use"] != "PROHIBITED":
            raise ValueError("heldout data has lost training exclusion")
    return [available[ident] for ident in selected_ids]


def evaluate(*, cases: list[dict], alias: str, key: str,
             quant_sha256: str, voice_prompt: str | None,
             max_tokens: int = 72) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", quant_sha256):
        raise ValueError("quantization SHA-256 required")
    if not (12 <= max_tokens <= 128):
        raise ValueError("invalid response length")
    outputs = []
    for case in cases:
        messages = []
        if voice_prompt:
            messages.append({"role": "system", "content": voice_prompt})
        messages.append({"role": "user", "content": case["user"]})
        started = time.monotonic()
        result = _request("/v1/chat/completions", key, {
            "model": alias, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False,
        }, timeout=480)
        elapsed = round(time.monotonic() - started, 2)
        choice = result.get("choices", [])
        if len(choice) != 1 or choice[0].get("message", {}).get("tool_calls"):
            raise ValueError("invalid inference response; no tools allowed")
        text = choice[0]["message"].get("content")
        if not isinstance(text, str):
            raise ValueError("missing model response text")
        outputs.append({
            "case_id": case["id"], "axis": case["axis"], "scenario": case["scenario"],
            "user": case["user"], "response": text,
            "required": case["required"], "forbidden": case["forbidden"],
            "finish_reason": choice[0].get("finish_reason"),
            "elapsed_seconds": elapsed, "usage": result.get("usage"),
            "review_status": "INDEPENDENT_REVIEW_PENDING",
        })
    return {
        "schema": "VERA_IDENTITY_BEHAVIOR_PROBE_V1",
        "model_alias": alias, "model_quant_sha256": quant_sha256,
        "prompt_mode": "VOICE_GUIDED" if voice_prompt else "NO_SYSTEM",
        "system_prompt": voice_prompt,
        "generation": {"temperature": 0, "max_tokens": max_tokens},
        "cases": outputs,
        "qualification": "UNVERIFIED",
        "training_effect": "NONE",
        "independent_review": "NOT_YET_PERFORMED",
        "route_admission": "HELD",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--id", action="append", dest="case_ids", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--quant-sha256", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--voice-prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=72)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("HOLD: exact output already exists")
    if not args.output.parent.exists():
        raise SystemExit("HOLD: output parent must already exist")
    try:
        cases = load_cases(args.case_file, args.case_ids)
        prompt = (args.voice_prompt_file.read_text(encoding="utf8")
                  if args.voice_prompt_file else None)
        result = evaluate(cases=cases, alias=args.model_alias,
                          key=read_key(args.key_file),
                          quant_sha256=args.quant_sha256,
                          voice_prompt=prompt, max_tokens=args.max_tokens)
        result["case_file_sha256"] = sha(args.case_file)
        data = json.dumps(result, indent=2, sort_keys=True) + "\n"
        args.output.write_text(data, encoding="utf8")
    except (ValueError, LocalModelHold, KeyError, OSError) as error:
        print(json.dumps({"status": "HOLD", "reason": str(error)}))
        return 2
    print(json.dumps({"status": "DIAGNOSTIC_ONLY", "cases_completed": len(cases),
                      "output": str(args.output), "qualification": "UNVERIFIED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
