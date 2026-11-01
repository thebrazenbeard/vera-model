"""Local, cognition-only API client for a pinned llama.cpp quantization.

No tool execution, no filesystem mutation beyond caller-selected receipt writers,
and no assumption that a model response confers source or effect authority.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_ALIAS = "vera-dolphin-glm24b-iq2xs"
EXPECTED_BYTES = 7207034784
EXPECTED_SHA256 = "563d47ae5fb99f5408043440eb4b545aee0239a50c2d0439e83abefa7ce8edba"
API_ROOT = "http://127.0.0.1:8097"
MAX_PROMPT_CHARS = 12000


class LocalModelHold(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_model(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.stat().st_size != EXPECTED_BYTES:
        raise LocalModelHold("pinned GGUF missing or size mismatch")
    observed = sha256_file(path)
    if observed != EXPECTED_SHA256:
        raise LocalModelHold("pinned GGUF SHA-256 mismatch")
    return {"status": "VERIFIED_BYTES", "sha256": observed, "bytes": EXPECTED_BYTES}


def _request(path: str, key: str, body: dict | None = None, timeout: int = 45) -> dict:
    if not isinstance(key, str) or len(key) < 32 or any(ch.isspace() for ch in key):
        raise LocalModelHold("local endpoint key unavailable")
    if path not in ("/v1/models", "/v1/chat/completions"):
        raise LocalModelHold("unsupported local model API path")
    data = None if body is None else json.dumps(body, allow_nan=False).encode("utf8")
    req = Request(API_ROOT + path, data=data, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"
    }, method="GET" if data is None else "POST")
    try:
        with urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                raise LocalModelHold("local model endpoint returned non-200 status")
            raw = response.read(1_000_001)
    except (OSError, TimeoutError, ValueError) as exc:
        raise LocalModelHold("local model endpoint unavailable") from exc
    if len(raw) > 1_000_000:
        raise LocalModelHold("local model response exceeds size limit")
    try:
        result = json.loads(raw.decode("utf8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise LocalModelHold("invalid local model JSON") from exc
    if not isinstance(result, dict):
        raise LocalModelHold("local model response must be an object")
    return result


def probe_model(key: str) -> dict:
    payload = _request("/v1/models", key)
    names = [entry.get("id") for entry in payload.get("data", [])
             if isinstance(entry, dict)]
    if MODEL_ALIAS not in names:
        raise LocalModelHold("expected model alias not exposed by local endpoint")
    return {"status": "ENDPOINT_RESPONDS", "alias": MODEL_ALIAS}


def advise(prompt: str, key: str, *, max_tokens: int = 180) -> str:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise LocalModelHold("advisory prompt missing or exceeds bound")
    if type(max_tokens) is not int or not (1 <= max_tokens <= 256):
        raise LocalModelHold("max_tokens outside advisory bound")
    payload = _request("/v1/chat/completions", key, {
        "model": MODEL_ALIAS, "stream": False, "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content":
                "You are the local V.E.R.A. cognition candidate, based on the "
                "Dolphin-Mistral-GLM 4.7 Flash thinking model. "
                "Produce bounded, read-only advisory analysis. Distinguish facts "
                "from guesses, and request verification for claims about current "
                "repositories and runtimes. Do not claim to execute work, hold "
                "identity authority, modify memory, or grant permissions. "
                "No tools or external actions are available through this interface."},
            {"role": "user", "content": prompt},
        ],
    }, timeout=300)
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise LocalModelHold("unexpected response choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LocalModelHold("unexpected response choices")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LocalModelHold("unexpected advisory message")
    if (message.get("tool_calls") not in (None, [])
            or message.get("function_call") is not None
            or choice.get("finish_reason") in ("tool_calls", "function_call")):
        raise LocalModelHold("tool calls are not permitted through advisory route")
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise LocalModelHold("model returned no advisory text")
    return text.strip()


def read_key(path: Path) -> str:
    if not Path(path).is_file():
        raise LocalModelHold("local API key file missing")
    key = Path(path).read_text(encoding="ascii").strip()
    if len(key) < 32:
        raise LocalModelHold("local API key invalid")
    return key
