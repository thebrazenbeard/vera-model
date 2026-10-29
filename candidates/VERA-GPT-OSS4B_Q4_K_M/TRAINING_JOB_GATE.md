# VERA-GPT-OSS4B Q4_K_M — Hugging Face Jobs persistence and spend gate

**Status: BLOCKED BEFORE PAID GPU TRAINING.** No model optimizer step or cloud adapter exists.

## Verified source
Trainable BF16 donor: AmanPriyanshu/gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts at commit d2fbf35dfdc9de041e71377c7747508fff794cbe (five safetensors shards, approximately 8.37 GB). The separate 3.23 GB Q4_K_M GGUF is for deployment, not the normal PEFT input.

Public pilot entrypoint: https://github.com/thebrazenbeard/labyrinth/blob/713cec86cb5b7643f9f64435ffca0ac3ca56d67a/scripts/train_vera_gptoss4b_pilot.py

GitHub source readback, actual Lappy Python compilation and no-training preflight passed. The script uses public-only identity/provenance/honesty examples, native chat templates, attention-projection rank-8 LoRA, BF16, <=192-token sequence, <=32 steps, and a changed-tensor digest. It cannot establish semantic identity or general reasoning by itself.

## Storage — mandatory BEFORE GPU spend

Hugging Face Jobs deletes ephemeral disk contents when the job ends. HF official documentation recommends a writable Storage Bucket mounted into Jobs, or a write-permitted Hub model repository to save results.

Current connected OAuth account has Jobs and read-repos but not a verified repository/bucket write grant. CPU Job https://huggingface.co/jobs/thebrazenbeard/6ae365de5ddcdc0830261828 failed 403 Forbidden when creating the proposed bucket. Its runtime was ~14 seconds. **No bucket was created and no GPU training ran.**

Required next gate: user reauthorizes Hugging Face with bucket/repository write/create permissions, or creates an appropriately writable bucket in the Hugging Face UI. Then:
1. Run CPU Job A mounting hf://buckets/thebrazenbeard/vera-gptoss4b-artifacts-20261029:/output:rw and write an isolated marker, then EXIT.
2. Run a distinct CPU Job B mounting that bucket :ro and verify exact marker contents and SHA-256 after Job A exited.
3. Confirm the output object is visible in the authenticated bucket listing. Only then run a paid GPU pilot.
4. After training, perform the same separate-job SHA-256 readback for adapter_model.safetensors and training_receipt.json, then reload the LoRA with the exact base checkpoint in a new process.

The separately reviewed two-job marker script is in this candidate's storage_gate.py. Do not substitute an in-process file read for cross-job persistence proof.


## Historical Hugging Face training precedent (verified Jobs history)

On 2026-09-25, Hugging Face Jobs completed actual **L4 GPU** jobs for the
separate Vera Qwen3.5 4B behavior adapter:

- GPU smoke job `6ab655326b030d633f690f81` — COMPLETED.
- GPU training job `6ab6551e52d0dbd7f1d8f67d` — COMPLETED.
- CPU artifact-custody check `6ab678fb6b030d633f6914d3` — COMPLETED.

The custody check reconstructed an archived adapter from 18 GitHub-stored
parts, checked 12,854,865 bytes and SHA-256
`1645cbe359cfdf4b3c9acd80471f71d2d6dfbce3c2a1a0fe6be24fc0513d1e69`.
**That is a proven historical alternative to a writable HF bucket for a
*separate* model; it does not establish that a GPT-OSS4B job currently has a
qualified adapter-export path.** The trainable model, scripts, available
permissions, and confidentiality must be checked afresh. Do not put private
model data in public GitHub merely to bypass a bucket permission failure.

The October 13 Dolphin24B cloud training attempt also failed the bucket-write
gate, so its failure must not be conflated with the September successful
Qwen3.5 Jobs.

## Compute efficiency and budget

HF published hourly prices: L4 GPU 24 GB $0.80/h; T4 GPU 16 GB $0.40/h. L4 is preferable for BF16 and greater headroom; T4/FP16 fallback is not presumed safe for GPT-OSS training. The first 20-minute L4 smoke has an upper hardware estimate of ~$0.27. A 60-minute L4 pilot would add ~$0.80. These are ceilings based on published compute rates, not invoice readings; observe actual charges and enforce Patrick's total $6 cap, including any other jobs. No recurring/scheduled jobs or automatic retries.

Use a pinned compatible GPU image once a smoke validates it. Fetch the public Labyrinth entrypoint at the exact Git commit, not the private vera-model repository, so GitHub credentials never enter the training container. Stop after a loader failure, nonfinite loss, label-template mismatch, absent trainable modules, OOM, or failed output write. Preserve existing vera-local:latest and the GLM24B candidate.

Successful optimizer updates do not imply real semantic understanding. Require separate pretrained/frozen, trained, sham and corrupted-label controls, held-out semantic testing and independent reload. This is a **source-only HOLD candidate** until every gate passes.

Official references:
- https://huggingface.co/docs/hub/jobs-training
- https://huggingface.co/docs/hub/jobs-configuration
- https://huggingface.co/docs/hub/jobs-manage
