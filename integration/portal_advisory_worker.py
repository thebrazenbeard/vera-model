"""Read-only P.O.R.T.A.L. process-worker candidate for the local Vera model.

Intentionally HOLD unless VeraOS issues exact model qualification and
the caller explicitly enables an already qualified advisory provider.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from local_model import LocalModelHold, advise, read_key

SCHEMA = "PORTAL_WORKER_RECEIPT_V1"
DENIAL_FIELDS = (
    "protected_effects_authorized",
    "source_mutation_authorized",
    "target_ref_mutation_authorized",
)


def validate_packet(packet: dict) -> None:
    if not isinstance(packet, dict):
        raise LocalModelHold("portal packet must be an object")
    if packet.get("advisory_only") is not True:
        raise LocalModelHold("only advisory packet is allowed")
    if packet.get("execution_authorized") is not False:
        raise LocalModelHold("execution_authorized must be false")
    if any(packet.get(name) is not False for name in DENIAL_FIELDS):
        raise LocalModelHold("protected or source mutation authorization denied")
    if packet.get("execution_effect_class") is not None:
        raise LocalModelHold("effect class forbidden for advisory worker")
    if packet.get("execution_promotion") is not None:
        raise LocalModelHold("execution promotion forbidden")
    for key in ("run_id", "subject_id", "node_id", "delivery_fencing_token",
                "exact_head", "plan_sha256"):
        if not packet.get(key):
            raise LocalModelHold(f"missing exact packet binding: {key}")
    task = packet.get("frontier")
    if not isinstance(task, str) or not task.strip() or len(task) > 8000:
        raise LocalModelHold("bounded advisory frontier is required")


def make_receipt(receipt_class: str, reason: str, artifacts: list[dict] | None = None) -> dict:
    if receipt_class not in ("HELD", "SUCCEEDED_NO_EFFECT"):
        raise ValueError("worker receipt class must have no effects")
    return {"schema": SCHEMA, "receipt_class": receipt_class,
            "reason": reason, "artifacts": artifacts or []}


def process(packet: dict, *, allowed: bool, key: str | None = None) -> tuple[dict, str | None]:
    validate_packet(packet)
    if not allowed:
        return make_receipt("HELD",
            "GLM24B local provider is unqualified; no cognition route selected"), None
    if key is None:
        raise LocalModelHold("missing local model key")
    prompt = ("Repository: " + str(packet.get("repository") or packet["subject_id"]) +
              "\nExact head: " + str(packet["exact_head"]) +
              "\nFrontier:\n" + packet["frontier"] +
              "\nReturn advisory findings only. No effects, no execution receipts.")
    answer = advise(prompt, key)
    return make_receipt("SUCCEEDED_NO_EFFECT",
                        "Model produced a read-only advisory response; "
                        "this is not execution or verified effect"), answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portal-packet", type=Path, required=True)
    parser.add_argument("--portal-receipt", type=Path, required=True)
    parser.add_argument("--qualification", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "contracts" / "veraos-provider.json")
    args = parser.parse_args()
    workspace = Path(os.environ.get("PORTAL_WORKSPACE", "")).resolve()
    if not os.environ.get("PORTAL_WORKSPACE"):
        raise SystemExit("PORTAL_WORKSPACE required")
    if args.portal_packet.resolve() != workspace / "packet.json":
        raise SystemExit("packet must be exactly in bound workspace")
    if args.portal_receipt.resolve() != workspace / "receipt.json":
        raise SystemExit("receipt must be exactly in bound workspace")
    if args.portal_receipt.exists():
        raise SystemExit("existing receipt cannot be overwritten")

    try:
        packet = json.loads(args.portal_packet.read_text(encoding="utf8"))
        qualification = json.loads(args.qualification.read_text(encoding="utf8"))
        # A separately signed, current qualification and route selection is needed
        # before true. Staged provider contract must remain HOLD.
        enabled = False
        if qualification.get("qualification", {}).get("status") == "PASS":
            raise LocalModelHold("qualification requires separately reviewed activation")
        receipt, advisory = process(packet, allowed=enabled)
        if advisory is not None:
            artifact = workspace / "advisory.txt"
            artifact.write_text(advisory + "\n", encoding="utf8")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            receipt["artifacts"] = [
                {"kind": "advisory", "relative_path": "advisory.txt", "sha256": digest}
            ]
    except (LocalModelHold, ValueError, OSError, KeyError, TypeError) as exc:
        receipt = make_receipt("HELD", str(exc))
    temporary = workspace / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                         encoding="utf8")
    os.replace(temporary, args.portal_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
