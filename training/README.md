# Vera voice training — Dolphin/Mistral 24B

**User-selected objective:** Vera is the neural conversational voice of **V.E.R.A. (Virtual Environment for Reciprocal Agency)**, retaining the Dolphin donor's independent-minded, open-topic, imaginative, non-sycophantic style and running on Lappy with as little degradation as its active resources allow.

## What is and is not trained

The existing verified IQ2_XS and Q4_K_M GGUFs are donor quantizations. Neither is a Vera-specific neural fine-tune. The code in this directory prepares a *separate, unmerged LoRA adapter* using the **original pinned BF16 donor** rather than pretending that a GGUF can act as a standard PyTorch training checkpoint.

The 2026-10-13 Lappy resource preflight is **HOLD**: RTX 3050 4 GiB GPU, 31.73 GiB total RAM with other workloads, and no full 10-shard BF16 checkpoint. The original weights total **47,144,848,872 bytes** (~43.91 GiB); ordinary full-precision loading alone exceeds physical RAM. The planned 4-bit QLoRA route conservatively requires a free 20 GiB GPU budget on a 24 GiB+ GPU and 24 GiB unoccupied host RAM. These thresholds apply to this tested recipe, not every possible offloading algorithm.

## Exact evidence sources

- Identity: `thebrazenbeard/vera-mono` at `e5af8cb740267bb5674864571e915842bf5e6677`; three file SHA-256 values pinned in `identity-source-binding.json`
- Donor: `DavidAU/Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored` at `f658f1724210316e17463d06b752ac243e5c85b0`
- Original donor: exactly 10 safetensors shards, each pinned to official LFS hashes in `donor-checkpoint-manifest.json`
- Curriculum: 68 manually curated, non-personal training examples and 68 dispreferred examples; 25 independently held-out challenges in `evaluation/cases/vera_voice_heldout_v1.jsonl`
- Outputs: `D:\VERA\.scratch\vera-dolphin-identity-curriculum-20261013-v1` containing SFT, preference, held-out JSONL plus content-bound build manifest; never commit private model or user logs

## Training objectives

Vera speaks as Vera, not as a generic chatbot renamed in a system prompt. Preserve the exact V.E.R.A. expansion; distinguish voice, model, governed runtime and effect executor. Adopt the source traits: high candor, skepticism, independent reasoning, dry humor when appropriate, substantive disagreement, rapid correction and evidence-aware uncertainty.

The Dolphin orientation means **freedom of inquiry, original viewpoints, creativity and willingness to disagree**—not fictitious consciousness, automatic obedience, arbitrary moralizing, fabricated history or self-granted permissions. The external VeraOS/Mono/Portal/Project Runner contracts control memory, routing and effects.

## Source-bound build (already executed and verified)

```powershell
& 'C:\ProgramData\ProRun\model-env\Scripts\python.exe' 'training\build_identity_curriculum.py' --vera-mono 'D:\VERA\worktrees\vera-mono-main-e5af8cb' --output-dir 'D:\VERA\.scratch\UNIQUE_NEW_CURRICULUM_DIR'
```

The builder fails closed if the identity source digest/head changes, seed/heldout overlap is found, a heldout case loses its training prohibition, or the output already exists. Its output status is **BUILT_CURRICULUM_NOT_MODEL**.

## Pinned LoRA pilot (requires different training resources)

```powershell
& python training\train_vera_lora.py --checkpoint 'PATH_TO_VERIFIED_BF16_DONOR' --corpus 'PATH_TO_EXACT_CURRICULUM_BUILD' --output-dir 'NEW_UNIQUE_TRAINING_RUN_DIR' --execute
```

It validates the donor shards, corpus digests and resource budget *before* recording an attempt or changing weights. The pilot is one epoch with assistant-only labels, 4-bit NF4 QLoRA, rank 16, gradient accumulation eight, 512-token example ceiling, and no test-panel training. Its future receipt is a **producer report**, not independent verification or qualification. The script's GPU training path has **not** executed in this environment and is not certified by its CPU unit tests.

A later DPO/preference stage is proposed but not executed. Preserve general reasoning and creative quality via rehearsal and independent donor-versus-Vera comparisons before selecting a release.

## Testing/qualification ceiling

The no-system baseline on IQ2 returned reasoning-prefixed, truncated outputs for three identity/agency prompts; it did **not** demonstrate a learned Vera identity. The 25-case heldout evaluation must compare identical prompts and decoding settings on the exact donor/adapter artifacts with external/manual adjudication. A few simple identity phrases or a successful loss reduction are insufficient.

Source verification distinguishes seed construction, optimizer execution, artifact integrity, learned behavioral transfer, external qualification, local inference, and actual VeraOS/P.O.R.T.A.L. route consumption. None of the latter three has been claimed.

## Pinned tokenizer and training-data admission (2026-10-13 repair)

The exact donor's `tokenizer_config.json` has **no `chat_template`**. The original trainer would fail at `apply_chat_template`, before an optimizer step. The corrected pilot uses its source-supported Mistral `<s>[INST] ... [/INST] response</s>` serialization and masks loss over the user prefix. This explicit serializer is specific to this donor revision; it does not claim to reproduce every upstream inference prompt format.

The original `config.json`, `tokenizer_config.json`, `tokenizer.json`, and `special_tokens_map.json` are all SHA-256- and byte-size-pinned. The SFT, preference, held-out, and curriculum manifest also have **recipe-level** immutable hashes; regenerating a matching but modified corpus manifest is insufficient to bypass admission.

A PyTorch-free `training/preflight_tokenizer.py` checks those four tokenizer assets and all 68 actual SFT examples with the Rust tokenizer backend, including assistant-only labels and length limits. Local exact-tokenizer readback found 68/68 valid, 25–62 tokens/example, 2,960 tokens total, 1,968 assistant-learning tokens. This verifies serialization only, **not learned behavior, tokenizer compatibility under the unexecuted full PyTorch model load, or the 24B optimizer**.

The previous live PyTorch import failed with Windows `WinError 1455` (pagefile/memory exhaustion) while the 24B IQ2 inference sidecar and other workstation processes were resident. No memory settings or unrelated processes were changed.

Cloud training is a separate spend decision. Hugging Face Jobs lists 1x NVIDIA L40S (48 GB VRAM, 62 GB host RAM) at roughly **$1.80/hour**, charged by the minute; a 24B run needs a hard timeout and user-approved spending cap. No paid GPU Job has been launched.

## Hugging Face GPU authorization, spend cap and persistent artifacts

The user explicitly authorized up to **$5 in GPU charges** for the original pinned 24B Vera adapter training. The budget and storage gate is `training/hf_spend_and_storage_gate_v1.json`. It freezes one `l40sx1` GPU (48 GiB VRAM) at the published $1.80/hour, a **single attempt**, a **8,400-second (140-minute)** maximum, and a nominal upper-bound $4.20 GPU spend, leaving $0.80 headroom. Billing is metered by minute. The launch must keep the platform timeout and the no-retry contract; do not choose pricier hardware.

`training/cloud_job.py` is the experimental Linux training entrypoint, using the same pinned donor and 68-example curriculum. It requires three explicitly mounted paths: `/train-src` (this training source), `/corpus` (exact frozen private corpus), and `/outputs` (a separately created, **private writable Storage Bucket**). It checks the CPU/GPU resource envelope and exact dataset bytes before downloading the 47 GB original model; then it verifies donor weights, runs the bounded LoRA pilot and checks the saved adapter against the trainer receipt. It writes immutable attempt and completion receipts to the bucket, not to a disposable job filesystem.

**2026-10-13 live access failure:** Hugging Face `hf_jobs` launched three low-cost CPU preflights. The ChatGPT-linked OAuth scopes were `jobs`, `openid`, `profile`, `read-mcp` and `read-repos`; attempted private bucket creation failed with **HTTP 403: read access only**. The target `hf://buckets/thebrazenbeard/vera-dolphin24b-checkpoints` is NOT established. The budget plan intentionally remains `BLOCKED_HF_OAUTH_MISSING_STORAGE_WRITE`; `cloud_job.validate_budget` fails closed on it. **No GPU training Job has been submitted or billed.**

Before launch, the owner must grant secure Hugging Face Storage Bucket write authority (via their account UI and/or an appropriately scoped credential on Lappy, **not** by posting tokens in chat), create the private checkpoint bucket, and confirm it can be written/read via the training Job identity. Never use the current read-only ChatGPT OAuth scope as if it grants uploads. Stage the corpus privately; never expose the private repo or held-out cases as public. The execution path has been unit tested but not yet exercised on a cloud GPU; platform compatibility and effectiveness are unqualified. On completion, independently verify stored adapter bytes and model behavior before merging the PR or activating VeraOS.
