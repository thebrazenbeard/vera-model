# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.7,<3",
#   "transformers==5.17.0",
#   "peft==0.21.0",
#   "accelerate>=1.7,<2",
#   "bitsandbytes==0.50.2",
#   "huggingface_hub>=1.20,<3",
#   "safetensors>=0.5",
#   "tokenizers>=0.21",
# ]
# ///
"""Cloud pilot for exact Dolphin 24B Vera LoRA; never runs without durable storage.

This script is local-source input to 'hf jobs uv run' with an explicitly mounted
PRIVATE Hugging Face bucket at /outputs. The HF job must have a platform-enforced
timeout <= 8400 seconds and no retries. The cloud's ephemeral disk is NOT evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


HARD_GPU_TIMEOUT_SECONDS = 8400
EXPECTED_MODEL_REVISION = "f658f1724210316e17463d06b752ac243e5c85b0"


def require(condition: bool, reason: str):
    if not condition:
        raise RuntimeError(reason)


def canonical_sha(data: dict) -> str:
    raw = json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf8")).hexdigest()


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as inp:
        for blob in iter(lambda: inp.read(1048576), b""):
            h.update(blob)
    return h.hexdigest()


def validate_budget(config: dict) -> float:
    require(config["schema"] == "VERA_DOLPHIN24B_HF_JOB_SPEND_AND_STORAGE_GATE_V1",
            "budget schema mismatch")
    require(config["status"] == "READY_FOR_LAUNCH",
            "cloud pilot blocked pending authorized writable private storage")
    require(config["authorized_gpu_cost_usd"] == 5,
            "user authorization changed: HOLD")
    require(config["flavor"] == "l40sx1" and
            config["attempts"] == 1,
            "GPU flavor or retries would change spending")
    require(type(config["timeout_seconds"]) is int and
            0 < config["timeout_seconds"] <= HARD_GPU_TIMEOUT_SECONDS,
            "GPU timeout exceeds bound")
    hourly=float(config["expected_gpu_price_usd_per_hour"])
    worst=round(hourly * config["timeout_seconds"] / 3600, 4)
    require(worst <= config["authorized_gpu_cost_usd"] -
            config["buffer_usd"], "nominal GPU budget not bounded")
    require(config["artifacts"]["requires_write_readback"] is True and
            config["artifacts"]["never_publish_to_public_repo"] is True,
            "persistent private storage contract missing")
    require(config["donor_revision"] == EXPECTED_MODEL_REVISION,
            "donor pin changed")
    return worst


def mounted_output_verified(*, mountinfo: str, target: Path) -> bool:
    """Prove a distinct mounted filesystem is attached at /outputs.

    Merely creating /outputs in the ephemeral container must never be
    mistaken for the private persistent Hugging Face bucket volume.
    """
    if target != Path("/outputs"):
        return False
    for raw in mountinfo.splitlines():
        fields = raw.split()
        if len(fields) > 5 and fields[4] == "/outputs" and " - " in raw:
            return True
    return False


def fresh_output(run_root: Path, run_id: str) -> Path:
    require(bool(re.fullmatch(r"vera24b-[0-9]{8}-[a-z0-9]{5,24}", run_id)),
            "run ID is not scoped")
    require(run_root == Path("/outputs"),
            "output must be the private bucket mount /outputs")
    require(sys.platform == "linux" and
            mounted_output_verified(mountinfo=Path("/proc/self/mountinfo").read_text(),
                                    target=run_root),
            "private output bucket mount was not verified")
    require(run_root.is_dir() and os.access(run_root, os.W_OK),
            "private output bucket not mounted or not writable")
    dest = run_root / run_id
    require(not dest.exists(), "run ID already exists; no overwrite")
    dest.mkdir(mode=0o700)
    payload={"schema":"VERA24B_PERSISTENCE_PROBE_V1",
             "status":"MOUNT_READ_WRITE_UNVERIFIED",
             "run_id":run_id}
    raw=json.dumps(payload,sort_keys=True).encode("utf8")
    probe=dest/"PERSISTENCE_PROBE.json"
    with probe.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(probe.read_bytes()==raw,"private bucket readback failed")
    return dest


def write_evidence(folder: Path, name: str, item: dict) -> None:
    require(re.fullmatch(r"[A-Z_]+\.json",name) is not None,
            "unapproved evidence filename")
    dest=folder/name
    require(not dest.exists(), "evidence file already exists; no overwrite")
    unsigned=dict(item)
    unsigned.pop("receipt_sha256",None)
    item=dict(unsigned,receipt_sha256=canonical_sha(unsigned))
    raw=(json.dumps(item,indent=2,sort_keys=True)+"\n").encode("utf8")
    with dest.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(dest.read_bytes()==raw,"durable receipt readback failed")


def run(*, source: Path, corpus: Path, output: Path,
        run_id: str, budget_path: Path) -> int:
    budget=json.loads(budget_path.read_text(encoding="utf8"))
    cost=validate_budget(budget)
    require(source.is_dir() and corpus.is_dir(),
            "training source/corpus not present")
    required=("train_vera_lora.py","donor-checkpoint-manifest.json",
              "qlora-recipe-v1.json","preflight_24b.py")
    require(all((source/name).is_file() for name in required),
            "exact source files unavailable")
    from importlib.util import spec_from_file_location, module_from_spec
    spec=spec_from_file_location("vera_cloud_trainer",source/"train_vera_lora.py")
    # Support trainer's sibling import without treating package discovery as authority.
    sys.path.insert(0,str(source))
    module=module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe=json.loads((source/"qlora-recipe-v1.json").read_text(encoding="utf8"))
    donor=json.loads((source/"donor-checkpoint-manifest.json").read_text(encoding="utf8"))
    require(donor["exact_revision"]==EXPECTED_MODEL_REVISION,
            "original donor revision changed")
    rows,corpus_manifest=module.verify_corpus(corpus,recipe)
    require(len(rows)==68,"expected exact SFT pilot only")
    for name, digest in budget["curriculum_sha256"].items():
        require(sha(corpus/name)==digest,
                "curriculum digest mismatch: "+name)
    # Confirm host and GPU memory BEFORE allocating storage or starting a download.
    from preflight_24b import host_ram,gpu_memory,classify
    host_total,host_free=host_ram()
    gpu_total,gpu_free=gpu_memory()
    capacities=classify(gpu_total_mib=gpu_total,gpu_free_mib=gpu_free,
                        host_total_gib=host_total,host_free_gib=host_free,
                        donor_checkpoint_complete=True)
    require(capacities["status"]=="READY_FOR_RECIPE_ONLY",
            "insufficient remote 24B QLoRA resources")
    job_dir=fresh_output(output,run_id)
    write_evidence(job_dir,"JOB_ATTEMPT.json",{
        "schema":"VERA_DOLPHIN24B_HF_ATTEMPT_V1",
        "status":"DOWNLOAD_PREPARING",
        "donor_revision":donor["exact_revision"],
        "train_sha256":budget["curriculum_sha256"]["vera_sft.jsonl"],
        "max_gpu_price_usd":cost,"platform_timeout_seconds":budget["timeout_seconds"],
        "optimizer_steps":0,"model_qualified":False,"route_eligible":False})
    from huggingface_hub import snapshot_download
    work=Path("/tmp")/run_id
    require(not work.exists(),"ephemeral scratch path already used")
    work.mkdir()
    snapshot=work/"donor"
    print("VERA_PILOT_PREPARE_DONOR",donor["exact_revision"],flush=True)
    snapshot_download(
        repo_id=donor["source_repository"],revision=donor["exact_revision"],
        local_dir=str(snapshot),max_workers=4,
        allow_patterns=["*.safetensors","*.json","*.model"],
    )
    observed=module.verify_donor(snapshot,donor)
    require(observed.get("complete") is True,
            "donor bytes/source cannot be independently verified")
    training=work/"pilot"
    cmd=[sys.executable,str(source/"train_vera_lora.py"),
         "--checkpoint",str(snapshot),"--corpus",str(corpus),
         "--output-dir",str(training),"--execute"]
    stdout=job_dir/"TRAINER_STDOUT.log"
    stderr=job_dir/"TRAINER_STDERR.log"
    print("VERA_PILOT_OPTIMIZER_START",flush=True)
    with stdout.open("xb") as sout,stderr.open("xb") as serr:
        started=time.monotonic()
        proc=subprocess.run(
            cmd,stdout=sout,stderr=serr,
            timeout=max(1,budget["timeout_seconds"]-1800),
            check=False,
        )
        sout.flush();os.fsync(sout.fileno())
        serr.flush();os.fsync(serr.fileno())
    require(proc.returncode==0,"trainer failed; artifacts remain unqualified")
    require((training/"DEV_TRAINING_COMPLETE.json").is_file(),
            "trainer missing completion receipt")
    manifest=json.loads((training/"DEV_TRAINING_COMPLETE.json").read_text())
    require(manifest.get("status")=="LOCAL_DEVELOPMENT_OPTIMIZER_STEPS_COMPLETE",
            "trainer receipt reports incomplete execution")
    require(manifest["qualification"]=="UNVERIFIED" and
            manifest["deployment"]=="NOT_DEPLOYED",
            "trainer overclaims deployment/qualification")
    archive=job_dir/"adapter"
    require(not archive.exists(),"adapter destination exists")
    shutil.copytree(training/"adapter",archive)
    verify={}
    for name,expected in manifest["adapter_files_sha256"].items():
        p=archive/name
        require(p.is_file() and sha(p)==expected,
                "private bucket adapter readback mismatch")
        verify[name]=expected
    shutil.copy2(training/"DEV_TRAINING_COMPLETE.json",
                 job_dir/"TRAINER_COMPLETION.json")
    require(sha(job_dir/"TRAINER_COMPLETION.json")==
            sha(training/"DEV_TRAINING_COMPLETE.json"),
            "private bucket receipt copy mismatch")
    write_evidence(job_dir,"JOB_COMPLETION.json",{
        "schema":"VERA_DOLPHIN24B_HF_JOB_RECEIPT_V1",
        "status":"TRAINED_ADAPTER_PERSISTED_UNQUALIFIED",
        "donor_revision":donor["exact_revision"],
        "train_sha256":budget["curriculum_sha256"]["vera_sft.jsonl"],
        "optimizer_steps":manifest["optimizer_steps"],
        "adapter_files_sha256":verify,
        "training_elapsed_seconds":round(time.monotonic()-started,2),
        "model_qualified":False,"route_eligible":False,"merged":False})
    print("VERA_PILOT_DURABLE_ADAPTER_READY_FOR_INDEPENDENT_VERIFICATION",run_id,
          flush=True)
    return 0


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("/train-src"))
    p.add_argument("--corpus",type=Path,default=Path("/corpus"))
    p.add_argument("--output",type=Path,default=Path("/outputs"))
    p.add_argument("--run-id",required=True)
    p.add_argument("--budget",type=Path,default=Path("/train-src/hf_spend_and_storage_gate_v1.json"))
    args=p.parse_args()
    return run(source=args.source,corpus=args.corpus,output=args.output,
               run_id=args.run_id,budget_path=args.budget)


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("HOLD_CLOUD_PILOT_UNQUALIFIED",type(exc).__name__,
              str(exc)[:350],flush=True)
        raise SystemExit(2)
