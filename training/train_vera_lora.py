"""Bounded pinned-donor 24B QLoRA identity pilot (not a deployment system).

Default is read-only preflight. The --execute path requires verified original
weight shards, sufficient GPU capacity and a previously built identity corpus.
GGUF quantizations cannot be substituted for the HF training checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from preflight_24b import classify, windows_ram, gpu_memory, DONOR_REVISION


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_donor(snapshot: Path, pins: dict) -> dict:
    require(pins["exact_revision"] == DONOR_REVISION,
            "donor manifest revision mismatch")
    require(pins["architecture"] == "MistralForCausalLM",
            "donor architecture mismatch")
    if not snapshot.exists() or not snapshot.is_dir():
        return {"complete": False, "verified_files": 0}
    files = pins["files"]
    expected = {f"model-{n:05d}-of-00010.safetensors" for n in range(1, 11)}
    require(set(files) == expected, "donor shard set incomplete")
    present = sum((snapshot / name).is_file()
                  and (snapshot / name).stat().st_size == row["bytes"]
                  for name, row in files.items())
    if (present != len(files) or not (snapshot / "config.json").is_file()
            or not (snapshot / "tokenizer.json").is_file()
            or not (snapshot / "tokenizer_config.json").is_file()):
        return {"complete": False, "verified_files": 0}
    for name, row in files.items():
        require(digest(snapshot / name) == row["sha256"],
                "donor SHA-256 mismatch: " + name)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf8"))
    require(config.get("model_type") == "mistral",
            "local donor has unexpected model architecture")
    return {"complete": True, "verified_files": len(files)}


def verify_corpus(root: Path, recipe: dict) -> tuple[list[dict], dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    require(manifest["schema"] == "VERA_DOLPHIN_IDENTITY_CURRICULUM_BUILD_V1",
            "curriculum manifest schema mismatch")
    require(manifest["status"] == "CURATED_SEED_ONLY_NOT_TRAINED" and
            manifest["training_executed_verified"] is False,
            "unexpected corpus lifecycle")
    path = root / "vera_sft.jsonl"
    require(digest(path) == manifest["files_sha256"]["vera_sft.jsonl"],
            "source training bytes differ from pinned build")
    require(digest(root / "vera_preferences.jsonl") ==
            manifest["files_sha256"]["vera_preferences.jsonl"],
            "preference data bytes differ")
    require(digest(root / "vera_heldout.jsonl") ==
            manifest["files_sha256"]["vera_heldout.jsonl"],
            "heldout bytes differ")
    require(manifest["target_revision"] == DONOR_REVISION,
            "corpus targets a different donor revision")
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()
            if line.strip()]
    expected = int(recipe["data"]["initial_examples"])
    require(len(rows) == expected and expected == 68,
            "training row count differs from approved seed")
    require(len({r["id"] for r in rows}) == len(rows),
            "duplicate training identity")
    require(all(len(r["messages"]) == 2 and
                r["messages"][0]["role"] == "user" and
                r["messages"][1]["role"] == "assistant"
                for r in rows), "unexpected training conversation")
    heldout = [json.loads(line) for line in
               (root / "vera_heldout.jsonl").read_text(encoding="utf8").splitlines()
               if line.strip()]
    require(all(r["training_use"] == "PROHIBITED" for r in heldout),
            "heldout training prohibited")
    require(not {r["messages"][0]["content"] for r in rows} &
            {r["user"] for r in heldout}, "training contains heldout prompt")
    return rows, manifest


def assistant_only_tokens(tokenizer, messages: list[dict],
                          limit: int) -> tuple[list[int], list[int]]:
    require(len(messages) == 2, "expected one user and one assistant message")
    user_text = tokenizer.apply_chat_template(messages[:1], tokenize=False,
                                               add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False)
    require(full_text.startswith(user_text),
            "tokenizer chat template lacks a reliable assistant prefix")
    prefix_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    require(full_ids[:len(prefix_ids)] == prefix_ids,
            "token boundary differs; cannot reliably mask user labels")
    require(len(prefix_ids) < len(full_ids) <= limit,
            "example out of supported token limit")
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    require(any(x != -100 for x in labels), "assistant labels empty")
    return full_ids, labels


def write_receipt(out: Path, *, recipe: dict, pins: dict,
                  manifest: dict, steps: int, loss: float) -> None:
    adapter = out / "adapter"
    require(adapter.is_dir(), "adapter not saved")
    safetensors = sorted(adapter.glob("*.safetensors"))
    require(len(safetensors) == 1 and
            safetensors[0].stat().st_size > 10000,
            "adapter safetensors missing")
    hashes = {p.name: digest(p) for p in adapter.iterdir() if p.is_file()}
    receipt = {
        "schema": "VERA_DOLPHIN24B_VOICE_QLORA_RECEIPT_V1",
        "status": "LOCAL_DEVELOPMENT_OPTIMIZER_STEPS_COMPLETE",
        "donor_revision": pins["exact_revision"],
        "donor_repository": pins["source_repository"],
        "dataset_sha256": manifest["files_sha256"]["vera_sft.jsonl"],
        "identity_source_head":
            manifest["identity_provenance"]["source_commit"],
        "optimizer_steps": steps, "observed_train_loss": loss,
        "adapter_files_sha256": hashes,
        "training_executed_reported": True,
        "training_executed_verified": False,
        "verification_status": "PRODUCER_SELF_REPORT_UNVERIFIED",
        "qualification": "UNVERIFIED", "deployment": "NOT_DEPLOYED",
        "model_route_eligible": False, "model_selected_by_portal": False,
        "full_weights_merged": False,
        "seed_pilot_only": True,
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["receipt_sha256"] = hashlib.sha256(canonical.encode("utf8")).hexdigest()
    (out / "DEV_TRAINING_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf8")


def train(snapshot: Path, rows: list[dict], recipe: dict,
          out: Path, pins: dict, manifest: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(recipe["trainer"]["seed"])
    torch.cuda.manual_seed_all(recipe["trainer"]["seed"])
    compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=compute)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        quantization_config=quant, device_map={"": 0}, dtype=compute)
    require(model.config.model_type == "mistral", "loaded wrong donor architecture")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    settings = recipe["trainer"]
    adapter_cfg = LoraConfig(
        r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        target_modules=settings["target_modules"],
        task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, adapter_cfg)
    model.train()
    pairs = [assistant_only_tokens(tokenizer, r["messages"],
                                   settings["max_sequence_tokens"])
             for r in rows]
    order = list(range(len(pairs)))
    random.Random(settings["seed"]).shuffle(order)
    accum = settings["gradient_accumulation_steps"]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=settings["learning_rate"])
    opt.zero_grad(set_to_none=True)
    total_steps = 0
    loss_total = 0.0
    for idx, offset in enumerate(order, start=1):
        tokens, labels = pairs[offset]
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
        targets = torch.tensor([labels], dtype=torch.long, device="cuda:0")
        attention = torch.ones_like(input_ids)
        loss = model(input_ids=input_ids, labels=targets,
                     attention_mask=attention).loss
        require(bool(torch.isfinite(loss).item()), "non-finite loss: HOLD")
        loss_total += float(loss.detach().cpu())
        window_start = ((idx - 1) // accum) * accum
        current_window = min(accum, len(order) - window_start)
        (loss / current_window).backward()
        if idx % accum == 0 or idx == len(order):
            opt.step()
            opt.zero_grad(set_to_none=True)
            total_steps += 1
    require(total_steps == math.ceil(len(order) / accum),
            "unexpected optimizer step count")
    (out / "adapter").mkdir(parents=True)
    model.save_pretrained(out / "adapter", safe_serialization=True)
    write_receipt(out, recipe=recipe, pins=pins, manifest=manifest,
                  steps=total_steps, loss=loss_total / len(order))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    recipe = json.loads((root / "qlora-recipe-v1.json").read_text(encoding="utf8"))
    pins = json.loads((root / "donor-checkpoint-manifest.json").read_text(encoding="utf8"))
    require(recipe["schema"] == "VERA_DOLPHIN24B_VOICE_QLORA_RECIPE_V1",
            "unknown training recipe")
    require(not args.output_dir.exists(),
            "prior training output exists: do not retry or overwrite")
    rows, corpus = verify_corpus(args.corpus, recipe)
    host_total, host_free = windows_ram()
    gpu_total, gpu_free = gpu_memory()
    # Check availability before expensive full-shard SHA-256 validation.
    capacity = classify(gpu_total_mib=gpu_total, gpu_free_mib=gpu_free,
                        host_total_gib=host_total, host_free_gib=host_free,
                        donor_checkpoint_complete=args.checkpoint.is_dir())
    if capacity["status"] != "READY_FOR_RECIPE_ONLY":
        print(json.dumps({"status": "HOLD", "capacity": capacity}, indent=2))
        return 2
    donor = verify_donor(args.checkpoint, pins)
    if not donor["complete"]:
        print(json.dumps({"status": "HOLD", "reason": "pinned donor shards incomplete"}))
        return 2
    if not args.execute:
        print(json.dumps({"status": "READY_FOR_EXPLICIT_EXECUTE",
                          "training_executed_verified": False,
                          "dataset_rows": len(rows)}, indent=2))
        return 0
    # Persist effect attempt boundary: an interrupted run is NOT safely retryable.
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "TRAINING_ATTEMPT.json").write_text(
        json.dumps({"schema": "VERA_DOLPHIN24B_ATTEMPT_V1",
                    "state": "ATTEMPTED_UNVERIFIED",
                    "donor_revision": DONOR_REVISION,
                    "dataset_sha256": corpus["files_sha256"]["vera_sft.jsonl"],
                    "protected_effect": "LOCAL_MODEL_WEIGHT_UPDATE"},
                   indent=2) + "\n", encoding="utf8")
    train(args.checkpoint, rows, recipe, args.output_dir, pins, corpus)
    print(json.dumps({"status": "ADAPTER_SAVED_EVALUATION_REQUIRED",
                      "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError,
            subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "HOLD", "reason": str(exc),
                          "training_completion": "UNVERIFIED"}))
        raise SystemExit(2)
