# Qwen3.5 Vera V10R3R9 — primary development lineage

**Status: PRIMARY DEVELOPMENT / HISTORICAL R9 LORA REPORTED / R10 HELD / RUNTIME AND BEHAVIORAL QUALIFICATION HOLD.**

This is the primary Vera model direction. It does not rename the stock Qwen base into a trained Vera release. The exact model, tokenizer, adapter and runtime must remain separately bound.

## Evidence already recorded

- [Vera's October 13 supersession decision](https://github.com/thebrazenbeard/vera-model/pull/1) selected the previously trained **Qwen3.5 Vera V10R3R9 LoRA** on Lappy over a Dolphin 24B optimizer. That record reports an R9 adapter SHA-256 `dc2a67a66c16c28a3b8cd5295a3e7a6ba243fb12ffb181540a0066b32047e495`.
- A [separate September Hugging Face Jobs precedent](../VERA-GPT-OSS4B_Q4_K_M/TRAINING_JOB_GATE.md) records a completed Qwen3.5 behavior-adapter GPU job and a reconstructed archived adapter SHA-256 `1645cbe359cfdf4b3c9acd80471f71d2d6dfbce3c2a1a0fe6be24fc0513d1e69`. This is a **different receipt**; it must not be silently identified with the October R9 artifact.
- The proposed R10 continuation used 32 identity/independent-judgment and 32 general-skill rehearsal examples over eight proposed updates. Its first launch was **held before training** by observed memory/pagefile pressure. No completed R10 adapter is verified.
- The current [TrainingDay source-only Draft PR #178](https://github.com/thebrazenbeard/model-training/pull/178) strengthens checkpoint custody and evidence gates; it does not prove a new Qwen model update, actual loader invocation or sealed evaluation.

## What still needs independent proof

1. Exact R9 adapter bytes and metadata, including `adapter_config.json`, model/base revision, architecture and tokenizer binding, verified against a fresh real artifact readback.
2. Matching verified Qwen base weights (Hugging Face/Transformers for PEFT, or exactly compatible GGUF plus converted GGUF LoRA for llama.cpp). An arbitrary Qwen3.5 GGUF is not automatically compatible with this adapter.
3. Fresh-process adapter load with a measurable changed-tensor or output delta against the same frozen base and input; no silently ignored adapter.
4. Same-source control comparisons: frozen base, correct-trained, zero-update and corrupted-label, plus sealed independent evaluation and general-skill retention.
5. Explicit governed provider admission, not inferred from a passed source test or an identity claim.

No weight files, private conversations, adapter bytes, activation keys or local host paths are included here. No current Qwen runtime route is authorized by this document.

See [adapter application and GGUF distinctions](ADAPTER_APPLICATION.md) and the [machine-readable primary record](../../contracts/primary-model.json).
