# VERA-GPT-OSS4B Q4_K_M — independent Vera candidate

**Status: SOURCE PREPARED / NEURAL TRAINING NOT VERIFIED / DEPLOYMENT HOLD.**

This is a second candidate in `vera-model`. The existing Dolphin/GLM24B
advisory model and its provider contract remain unchanged. This document is
a source and training handoff, not a model file or release.

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

Lappy has 32 GB physical RAM and 4 GB VRAM, but current live free VRAM was
only 828 MB and free RAM around 8 GB. It is not a qualified 4B training host in
that state. The user authorized at most **USD 6 total paid Hugging Face compute**
and no destructive interruption of active workloads.

A Hugging Face L4 Jobs instance is listed at USD 0.80/hour. Bound every job
by a hard timeout and account for all charges. **The initial storage mount
`hf://buckets/thebrazenbeard/jobs-artifacts` returned 404.** Jobs delete
their ephemeral files on exit, so do not start a GPU training job without an
accessible read/write bucket or proven write-permitted Hub repository and
post-run artifact readback. No paid training occurred at this point.

## Release and runtime holds

- Current `vera-local:latest` remains untouched.
- The GLM24B route remains `HOLD`.
- New GPT-OSS4B candidate remains `HOLD` with `COGNITION_ONLY` scope.
- No automatic provider selection, protected effects, model promotion or
  self-authorized memory writes follow from the presence of this document.
- Training proof classes: SOURCE / optimizer step / adapter tensor digest /
  reload / actual inference / independent evaluation / deployment must be separate.
