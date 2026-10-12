"""Bounded manual smoke check; does not select a P.O.R.T.A.L. route."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_model import (_request, advise, probe_model, read_key,
                         MODEL_ALIAS, EXPECTED_SHA256, LocalModelHold)


def smoke(key_path: Path) -> dict:
    key = read_key(key_path)
    probe = probe_model(key)
    started = time.perf_counter()
    result = _request("/v1/chat/completions", key, {
        "model": MODEL_ALIAS, "stream": False,
        "max_tokens": 28, "temperature": 0,
        "messages": [{"role": "user", "content":
            "In one concise sentence, explain the difference between "
            "a read-only suggestion and permission to execute a change."}],
    }, timeout=600)
    elapsed = round(time.perf_counter() - started, 3)
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise LocalModelHold("smoke response has invalid choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls"):
        raise LocalModelHold("unexpected tool call in model smoke")
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise LocalModelHold("no generated content in smoke")
    return {
        "schema": "VERA_GLM24B_LOCAL_SMOKE_V1",
        "status": "BEHAVIOR_OBSERVED_DIAGNOSTIC_ONLY",
        "model_alias": MODEL_ALIAS,
        "expected_gguf_sha256": EXPECTED_SHA256,
        "endpoint_probe": probe,
        "elapsed_seconds": elapsed,
        "output": text.strip(),
        "tokens_used": result.get("usage"),
        "qualification": "UNVERIFIED",
        "portal_route_selected": False,
        "effect_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("smoke output already exists; no overwrite")
    try:
        result = smoke(args.key_file)
    except LocalModelHold as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc)}))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
