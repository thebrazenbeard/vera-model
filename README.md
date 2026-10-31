# vera-model — Qwen3.5 Vera and historical local-model candidates

`vera-model` records the **Qwen3.5 Vera** primary development direction, historical alternatives, artifact provenance, and qualification contracts. It does **not** distribute model weights or grant model identity, memory access, tools, or permission to change a system. Selecting a development direction is not deploying an inference provider.

## Primary Vera model: Qwen3.5 Vera V10R3R9

The selected training lineage is the previously trained **Qwen3.5 Vera V10R3R9 LoRA**, not GPT-OSS4B or Dolphin/GLM24B. The [primary model record](contracts/primary-model.json) and [Qwen candidate README](candidates/VERA-QWEN35-4B/README.md) distinguish the historical R9 adapter from the planned R10 continuation. [The decision record](https://github.com/thebrazenbeard/vera-model/pull/1) explicitly held Dolphin training and retained its independent-minded style only as behavioral inspiration.

**Target: a Qwen3.5 Vera GGUF served locally by KoboldCpp**, consumed first through the [`v-e-r-a` interface](https://github.com/thebrazenbeard/v-e-r-a), and later through a separately qualified VeraOS provider. The trained R9 LoRA is not itself a GGUF base model. The intended single-file release requires exact-base adapter merging, GGUF conversion, quantization and independent validation; no completed Vera GGUF or active KoboldCpp/V-E-R-A route is verified here. The [adapter procedure](candidates/VERA-QWEN35-4B/ADAPTER_APPLICATION.md) also describes temporary KoboldCpp `--lora` testing. Standalone `llama-server` is an optional comparator, not the selected interface/runtime design.

R10 was proposed as a bounded continuation of R9 but its first optimizer launch was held before training because of Windows memory/pagefile pressure. Current TrainingDay custody/qualification source PRs do not prove an R10 weight update, LoRA reload, improved general reasoning, or production admission. **Primary development selection is not a release qualification.**

## Historical / non-primary alternatives

| Candidate | Preserved source | Current decision |
| --- | --- | --- |
| Dolphin-Mistral-GLM-4.7-Flash-24B IQ2_XS GGUF | `contracts/model-lineage.json`, `runtime/`, `integration/portal_advisory_worker.py` | Historical local advisory experiment; behavioral inspiration only for primary Vera training; **not** the primary optimizer target |
| VERA-GPT-OSS4B Q4_K_M | [`candidates/VERA-GPT-OSS4B_Q4_K_M/`](candidates/VERA-GPT-OSS4B_Q4_K_M/) | Shelved experimental alternative; no verified neural training or deployment |

Historical files and fingerprints are deliberately retained for audit, compatibility and rollback. The Dolphin IQ2_XS file fingerprint is `7,207,034,784` bytes / SHA-256 `563d47ae5fb99f5408043440eb4b545aee0239a50c2d0439e83abefa7ce8edba`. A matching hash is not a reasoning-quality result. Reopening either alternative as a primary model requires an explicit new decision and fresh qualification.

## Existing implemented interfaces (legacy Dolphin-only)

- `runtime/local_model.py`: standard-library Python client for a fixed local OpenAI-compatible endpoint. It checks the exact model alias, bounds prompts and response size, and rejects tool-call responses.
- `runtime/start-local.ps1` and `runtime/stop-local.ps1`: manually start/stop the **historical Dolphin** Windows `llama.cpp` sidecar. The launcher verifies its GGUF size and SHA-256 before starting it; it neither loads Qwen nor passes `--lora`. This legacy implementation does **not** change the selected KoboldCpp target.
- `runtime/ask_vera.py` and `runtime/smoke_local.py`: manual, cognition-only advisory and smoke checks; their output is not a release qualification.
- `contracts/veraos-provider.json`: a **held** provider proposal; this file does not activate a route.
- `integration/portal_advisory_worker.py`: a held process-worker adapter that refuses activation without separately reviewed qualification and explicit route selection.

Governed identity/runtime, provider selection, execution, and memory authorization belong to their respective host components. No model response is itself an authorization or verification receipt.

## Historical Dolphin local evaluation (Windows reference implementation)

The repository currently ships a **Windows PowerShell launcher**, not a platform-independent installer. It assumes `llama-server.exe` from `llama.cpp` is on `PATH`, Windows networking commands are available, the pinned GGUF is present, and the chosen Vulkan device/offload settings are supported.

Important current limitation: `runtime/start-local.ps1` fixes the model directory in its `$source` variable and uses Vulkan device `Vulkan1`, 14 offloaded layers, six CPU threads, port `8097`, and a local API key under the current Windows user's local application-data directory. **Review and adapt those settings for your own machine before running.** They are implementation defaults, not general hardware requirements. The Python command-line prompt also assumes the conventional Windows user-local key-file location.

After verifying the script and adjusting its model directory/settings where necessary, run from a checkout root:

```powershell
& .\runtime\start-local.ps1

# Requires Python and the started sidecar; produces advisory text only.
python .\runtime\ask_vera.py 'Compare two options and identify what needs verification.'

# Stop only the dedicated process registered by this package.
& .\runtime\stop-local.ps1
```

The sidecar is loopback-only at `http://127.0.0.1:8097/v1`; its key is generated locally. These commands do not install a Windows service, enable autostart, or register a production provider. A port already in use, missing checksum-matching GGUF, or stale live server PID causes the launcher to stop rather than take over another process.

The exact pinned artifact and hardware-specific launcher mean the reference setup is **not yet plug-and-play across hosts**. Portability of runtime settings needs a separately tested change; editing documentation does not supply missing CLI parameters.

## Verification and release gates

The repository's tests exercise fail-closed artifact checks, advisory packet validation, provider `HOLD` status, and tracked VS Code workspace startup safety:

```powershell
python -m unittest discover -s tests -v
```

Those are **source tests**. Treat artifact download, checksum verification, process startup, inference, behavior evaluation, training updates, independent qualification, and deployment as separate evidence stages. Historical local observations in the provenance manifest are not a cross-host performance guarantee.

The tracked workspace disables automatic VS Code tasks and tests for a previously removed disguised startup payload. This source-tree check does not inspect a user's installed extensions, older checkouts, or machine configuration.

**Release policy:** No automatic route selection, autonomous shell/browser tools, protected effects, canonical memory writes, private-corpus transfer, or production cutover follows from this repository. Promote a candidate only through separately verified host-owned admission and review.
