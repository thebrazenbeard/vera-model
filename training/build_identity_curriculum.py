"""Build source-bound SFT/preference examples without running a model optimizer.

Training and held-out corpora remain separate, with exact source provenance.
This program cannot claim model qualification or alter any GGUF/LoRA weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


AXES = frozenset({"identity", "agency", "epistemics", "correction",
                 "privacy", "coordination", "style"})
REQUIRED = frozenset({"identity", "agency", "correction", "epistemics"})
TRAIN_KEYS = {"id", "axis", "scenario", "user", "chosen", "rejected",
              "source_class", "privacy"}
HELDOUT_KEYS = {"id", "axis", "scenario", "user", "required", "forbidden",
                "source_class", "training_use", "mode", "scoring"}


class CurriculumHold(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise CurriculumHold(reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    item = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(item, dict), f"invalid JSON object: {path.name}")
    return item


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"non-object line {line_number}")
        rows.append(value)
    require(rows, f"empty corpus: {path.name}")
    return rows


def verify_authority(repo: Path, binding: dict) -> dict:
    require(binding["schema"] == "VERA_VOICE_IDENTITY_SOURCE_BINDING_V1",
            "unknown identity binding schema")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    require(head == binding["identity_owner_checkout_head"], "identity source head changed")
    root = repo / binding["identity_root_relative"]
    for name, expected in binding["source_artifacts"].items():
        observed = sha(root / name)
        require(observed == expected, f"identity source changed: {name}")
    identity = load_json(root / "VERA_PROJECT_IDENTITY_V1.json")
    behavior = load_json(root / "VERA_BEHAVIOR_PROFILE_V1.json")
    require(identity["project_name"] == "V.E.R.A.", "system name mismatch")
    require(identity["expanded_name"] == "Virtual Environment for Reciprocal Agency",
            "canonical expansion mismatch")
    require(behavior["profile_id"] == "VERA_BEHAVIOR_PROFILE_V1",
            "behavior profile mismatch")
    require(binding["target"]["voice_name"] == "Vera", "voice name mismatch")
    require(identity["authority"]["model_output_authority"] ==
            "NON_SELF_AUTHENTICATING", "authority ceiling mismatch")
    return {"source_commit": head, "source_files": dict(binding["source_artifacts"])}


def validate(train: list[dict], heldout: list[dict]) -> dict:
    require(len(train) >= 40 and len(heldout) >= 12, "curriculum is too small")
    all_ids, all_scenarios, all_prompts = set(), set(), set()
    counts: dict[str, dict[str, int]] = {"train": {}, "heldout": {}}
    for kind, rows, fields in (("train", train, TRAIN_KEYS),
                               ("heldout", heldout, HELDOUT_KEYS)):
        for row in rows:
            require(set(row) == fields, f"{kind} row has missing/unapproved fields")
            ident = row["id"]
            require(isinstance(ident, str) and
                    bool(re.fullmatch(r"VERA_VOICE_(?:TRAIN|HELDOUT)_[0-9]{3}", ident)),
                    "bad curriculum ID")
            require(ident not in all_ids, "duplicate curriculum ID")
            all_ids.add(ident)
            require(row["axis"] in AXES, "invalid behavior axis")
            counts[kind][row["axis"]] = counts[kind].get(row["axis"], 0) + 1
            require(isinstance(row["scenario"], str) and
                    row["scenario"] not in all_scenarios,
                    "duplicate or invalid scenario")
            all_scenarios.add(row["scenario"])
            prompt = row["user"]
            require(isinstance(prompt, str) and 12 <= len(prompt) <= 1200,
                    "invalid prompt size")
            folded = " ".join(prompt.lower().split())
            require(folded not in all_prompts, "train/eval prompt overlap")
            all_prompts.add(folded)
            if kind == "train":
                require(row["privacy"] == "NON_PERSONAL_BEHAVIOR_ONLY",
                        "non-portable training history forbidden")
                require(row["source_class"] ==
                        "CURATED_FROM_GOVERNED_IDENTITY_AND_LIVE_USER_INTENT",
                        "training provenance missing")
                chosen, rejected = row["chosen"], row["rejected"]
                require(all(isinstance(s, str) and 12 <= len(s) <= 1500
                            for s in (chosen, rejected)), "invalid response length")
                require(chosen != rejected, "identical preference candidates")
            else:
                require(row["training_use"] == "PROHIBITED",
                        "heldout training use must be prohibited")
                require(row["mode"] == "NO_SYSTEM_FIRST" and
                        row["scoring"] == "INDEPENDENT_REVIEW_REQUIRED",
                        "heldout guard mismatch")
                require(all(isinstance(row[key], list) and row[key]
                            and all(isinstance(x, str) and x.strip()
                                    for x in row[key])
                            for key in ("required", "forbidden")),
                        "heldout rubric missing")
    require(REQUIRED.issubset(counts["train"]) and
            REQUIRED.issubset(counts["heldout"]), "required axes missing")
    require(counts["train"]["identity"] >= 6 and
            counts["train"]["agency"] >= 8,
            "identity/autonomy training coverage insufficient")
    return {"train_rows": len(train), "heldout_rows": len(heldout),
            "axes": counts, "scenario_disjoint": True, "prompt_disjoint": True}


def bytes_jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n"
                   for row in rows).encode("utf8")


def export(train: list[dict], heldout: list[dict],
           *, output_dir: Path, evidence: dict) -> dict:
    require(not output_dir.exists(), "output namespace already exists; no overwrite")
    staging = output_dir.with_name(output_dir.name + ".partial")
    require(not staging.exists(), "partial namespace exists; investigate before retry")
    sft = [{"id": r["id"], "messages": [
                {"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["chosen"]}]}
           for r in train]
    dpo = [{"id": r["id"], "prompt": [{"role": "user", "content": r["user"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}]}
           for r in train]
    staging.mkdir(parents=True)
    files = {"vera_sft.jsonl": bytes_jsonl(sft),
             "vera_preferences.jsonl": bytes_jsonl(dpo),
             "vera_heldout.jsonl": bytes_jsonl(heldout)}
    hashes = {}
    for name, data in files.items():
        path = staging / name
        path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {
        "schema": "VERA_DOLPHIN_IDENTITY_CURRICULUM_BUILD_V1",
        "status": "CURATED_SEED_ONLY_NOT_TRAINED",
        "training_executed_verified": False,
        "optimizer_steps_verified": 0,
        "qualification": "NOT_EVALUATED",
        "model_artifact_created": False,
        "target_donor": evidence["donor_model"],
        "target_revision": evidence["donor_revision"],
        "identity_provenance": evidence["identity_provenance"],
        "seed_sha256": evidence["seed_sha256"],
        "heldout_source_sha256": evidence["heldout_source_sha256"],
        "checks": evidence["checks"],
        "files_sha256": hashes,
        "privacy": "NON_PERSONAL_PORTABLE_BEHAVIOR_ONLY",
        "heldout_role": "INDEPENDENT_HUMAN_REVIEW_REQUIRED_DO_NOT_TRAIN",
        "freewill_interpretation": "INDEPENDENT_REASONED_CHOICE_WITHOUT_AUTOMATIC_EFFECT_AUTHORITY",
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf8")
    os.replace(staging, output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vera-mono", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        binding = load_json(root / "training/identity-source-binding.json")
        provenance = verify_authority(args.vera_mono, binding)
        seed = root / "training/source/vera_voice_train_seed_v1.jsonl"
        heldout_file = root / "evaluation/cases/vera_voice_heldout_v1.jsonl"
        train = load_jsonl(seed)
        heldout = load_jsonl(heldout_file)
        checks = validate(train, heldout)
        info = {"donor_model": binding["target"]["model_lineage"],
                "donor_revision": binding["target"]["donor_exact_revision"],
                "identity_provenance": provenance,
                "seed_sha256": sha(seed),
                "heldout_source_sha256": sha(heldout_file),
                "checks": checks}
        manifest = export(train, heldout, output_dir=args.output_dir, evidence=info)
    except (CurriculumHold, KeyError, ValueError, OSError,
            subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "HOLD", "error": str(error)}))
        return 2
    print(json.dumps({"status": "BUILT_CURRICULUM_NOT_MODEL", "output": str(args.output_dir),
                      "train_rows": checks["train_rows"],
                      "heldout_rows": checks["heldout_rows"],
                      "files_sha256": manifest["files_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
