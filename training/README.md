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
