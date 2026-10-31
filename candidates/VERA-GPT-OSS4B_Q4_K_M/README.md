# VERA-GPT-OSS4B Q4_K_M — independent Vera candidate

**Status: LAPPY GGUF FILE VERIFIED / NEURAL TRAINING NOT VERIFIED / DEPLOYMENT HOLD.**

This is a second candidate in `vera-model`. The existing Dolphin/GLM24B
advisory model and its provider contract remain unchanged. This document is
a source and training handoff, not a model file or release. The model file is
stored separately on Lappy; a hash-verified local file is not evidence of
good inference, usable reasoning, neural training or deployment.

## Lappy acquisition receipt (2026-10-29)

- **Host**: `LAPPY`; **source class**: `REAL_SYSTEM_PROBE`.
- **Path**: `D:\VERA\models\gguf\vera-gpt-oss4b-q4-k-m\gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-q4_k_m.gguf`.
- **Length**: `3,233,019,552` bytes; **SHA-256**: `38b0279e4294beb54f97ac23051674174b34b2acd6c5e6532860c72298789586`.
- **Verified**: `2026-10-29T17:45:26Z` using PowerShell `Get-Item` and `Get-FileHash`, with no file modification.
- **Limits**: Historical point-in-time custody, not live file monitoring, BF16
  donor availability, basic reasoning competence, LoRA training, or provider
  route admission. The candidate remains `HOLD`.

## Exact provenance

- Canonical identity: **VERA-GPT-OSS4B Q4_K_M**.
- Trainable original: `AmanPriyanshu/gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts` at immutable Git commit `d2fbf35dfdc9de041e71377c7747508fff794cbe` (public Hugging Face API observed 2026-10-29).
- Existing inference GGUF: `cyberjunkee/gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-Q4_K_M-GGUF` at `d3825cfdc1c250b75a2485e9054162bf237fdd01`.
- GGUF filename: `gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-q4_k_m.gguf`.
- Exact expected GGUF bytes: `3233019552`; SHA-256 `38b0279e4294beb54f97ac23051674174b34b2acd6c5e6532860c72298789586`.
- Acquisition-only code and offline tests: [Labyrinth PR #57](https://github.com/thebrazenbeard/labyrinth/pull/57).
- The original GPT-OSS 20B had experts pruned to approximately 4.2B parameters **without retraining**. No competent baseline or improved reasoning has been established.

## Distinct responsibilities

`vera-model` registers the model lineage, training artifacts and gated candidate;
`labyrinth` measures neural learning, controls, regression and qualification;
`model-training` provides optional identity/curriculum exercises and is not itself a
neural optimizer; `vera-adapter` supplies runtime context/memory/tool interfaces.
A weights update cannot fabricate a memory backend or tool connectivity.

The experiment should first distinguish operational identity ('I'm Vera') from
underlying technical provenance, and demand truthful uncertainty about actual
memory access, tools, consciousness and completed actions. Operational identity
is not evidence of an experience or independently validated self-model.

## Training path and budget

1. Validate the exact trainable revision, model config, tokenizer and permitted
   LoRA targets. Do not use the compressed GGUF as the trainable PEFT input.
2. Keep any personal/private text out of publicly exposed training data.
   Use the generic public training curriculum in this branch for the first pilot.
3. On a compatible GPU, run the optional training pilot, save a *new* LoRA
   adapter, prove changed trainable tensors, and independently reload it.
4. Compare matched frozen, trained, zero-update and corrupted-label arms on
   development data. Preserve separate sealed tests for independent release.
5. Do not merge/quantize/install the resulting model until the identity,
   provenance, honesty, general reasoning and regression gates pass.

Host capacity cannot be inferred from the model's parameter count alone.
Before choosing a training runtime, measure available RAM, VRAM, storage,
software compatibility and concurrent workload occupancy. A suitable inference
host is not automatically a suitable optimizer host. Training must not interrupt
unrelated workloads.

For paid or ephemeral GPU jobs, establish an explicit cost ceiling, hard
timeout, and independently verified artifact destination **before** starting.
Require a successful test write and post-run readback from a bucket or a
write-permitted model repository; otherwise a completed job can still lose its
adapter on exit. This repository does not establish a paid-training execution
or persisted adapter artifact.

## Release and runtime holds

- Any pre-existing local runtime is outside this candidate's deployment scope.
- The GLM24B route remains `HOLD`.
- New GPT-OSS4B candidate remains `HOLD` with `COGNITION_ONLY` scope.
- No automatic provider selection, protected effects, model promotion or
  self-authorized memory writes follow from the presence of this document.
- Training proof classes: SOURCE / optimizer step / adapter tensor digest /
  reload / actual inference / independent evaluation / deployment must be separate.
