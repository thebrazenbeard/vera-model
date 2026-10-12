# Vera Model: Dolphin-Mistral-GLM-4.7-Flash-24B (Lappy candidate)

This **separate** local-development candidate corresponds to the actual description of the private `thebrazenbeard/vera-model` repository, not the Qwen R9 lineage.

## Donor and quantization
- Fine-tuned source: `DavidAU/Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored`.
- GGUF quantization: `mradermacher/Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored-i1-GGUF`.
- Pinned GGUF Git revision `37bdf0317324e9e5a9ad43065e573cf37fd6653f`.
- Type `i1-IQ2_XS`, byte size `7207034784`, expected SHA-256 `563d47ae5fb99f5408043440eb4b545aee0239a50c2d0439e83abefa7ce8edba`.
- Lappy: i5-11400H 6 cores/12 threads, ~32 GiB physical RAM, RTX 3050 4 GiB, Windows, local llama.cpp Vulkan runtime.
- The very low-bit quantization may substantially impair instruction following and reasoning. Benchmark against alternatives before treating it as a coordination engine.

## Runtime
`runtime/start-local.ps1` verifies the complete GGUF bytes and starts a separate local llama.cpp service at `http://127.0.0.1:8097/v1` with explicit NVIDIA Vulkan offload and an automatically generated local-only API key. It does **not** replace existing Vera, Ollama, or P.O.R.T.A.L. services. It is launched manually; never as Windows autostart by this package.

The pure-stdlib `runtime/local_model.py` validates the pinned bytes, verifies the exact model alias, and makes a bounded chat completion request without tool-call access. It is **cognition only**.

## Integration and control ownership
- VeraOS and Vera Mono: governed runtime and identity.
- Rezon: reasoning architecture.
- P.O.R.T.A.L.: provider selection, portfolio planning, admission and scheduling.
- Project Runner: fenced execution, receipts, reconciliation and verification.
- GLM24B: optional advisory cognition, **not** ambient protected-effect authority.

`contracts/veraos-provider.json` is a **HOLD** candidate. The `integration/portal_advisory_worker.py` implements the existing `PORTAL_PROCESS_WORKER_V1` receipt form but refuses route activation until a separately reviewed exact-subject qualification is admitted. It must not be selected merely because its HTTP service is reachable.

## Safety and evidence
- No repo mutations, shell commands, browser access, memory writes, or project actions are exposed to model output.
- No transfer of credentials or private training corpora.
- All generated advisory content needs source and currentness checks before any plan or effect.
- Model download, checksum, model load, inference, behavioral evaluation, qualification, route selection, and sustained usability are **separate** proof stages.
- Repo Git push, deployments, and live provider cutover have **not** been performed by this staging package.


## Lappy on-demand commands (tested with Windows PowerShell)

Launch the separately pinned model:

```powershell
& 'D:\VERA\portal-task-state\vera-model-glm24b-lappy-20261012\runtime\start-local.ps1'
```

Ask for local read-only advice (model service must be running):

```powershell
& 'C:\ProgramData\ProRun\model-env\Scripts\python.exe' 'D:\VERA\portal-task-state\vera-model-glm24b-lappy-20261012\runtime\ask_vera.py' 'Review a portfolio decision without performing any changes.'
```

Stop only this model instance to reclaim RAM:

```powershell
& 'D:\VERA\portal-task-state\vera-model-glm24b-lappy-20261012\runtime\stop-local.ps1'
```

The 2026-10-12 benchmark with 14 GPU-offloaded layers produced 20 tokens in 15.25 seconds end-to-end, about 2.39 generated tokens/s. This is a single test, not a throughput guarantee. Model selection in VeraOS/P.O.R.T.A.L. is **still HOLD** until separately reviewed qualification and explicit route binding. The running local sidecar is not proof of production deployment.
