# vera-model — local model candidates and qualification contracts

`vera-model` is a source repository for **optional local-language-model candidates**. It records model provenance, checks pinned artifacts, provides a bounded read-only advisory interface, and keeps provider admission separate from model availability. It does **not** distribute model weights or grant a model identity, memory access, tools, or permission to change a system.

## What's included

| Candidate | Source artifacts | Current release boundary |
| --- | --- | --- |
| Dolphin-Mistral-GLM-4.7-Flash-24B, IQ2_XS GGUF | `contracts/model-lineage.json`, `runtime/`, `integration/portal_advisory_worker.py` | Pinned local inference candidate; behavioral qualification and provider route **HOLD** |
| VERA-GPT-OSS4B Q4_K_M | `candidates/VERA-GPT-OSS4B_Q4_K_M/` | Provenance and training handoff only; neural training and deployment **not verified** |

The first candidate's pinned GGUF is `7,207,034,784` bytes with SHA-256 `563d47ae5fb99f5408043440eb4b545aee0239a50c2d0439e83abefa7ce8edba`. Its donor and quantizer references are recorded in `contracts/model-lineage.json`. The very low-bit quantization may reduce model quality; a verified file hash is **not** a reasoning-quality result.

The second candidate is independent of the first. Its [candidate README](candidates/VERA-GPT-OSS4B_Q4_K_M/README.md) records distinct pinned sources, training prerequisites, and release holds. A quantized inference GGUF must not be mistaken for a trainable PEFT base.

## Interfaces and ownership

- `runtime/local_model.py`: standard-library Python client for a fixed local OpenAI-compatible endpoint. It checks the exact model alias, bounds prompts and response size, and rejects tool-call responses.
- `runtime/start-local.ps1` and `runtime/stop-local.ps1`: manually start/stop the dedicated Windows `llama.cpp` sidecar. The launcher verifies GGUF size and SHA-256 before starting it.
- `runtime/ask_vera.py` and `runtime/smoke_local.py`: manual, cognition-only advisory and smoke checks; their output is not a release qualification.
- `contracts/veraos-provider.json`: a **held** provider proposal; this file does not activate a route.
- `integration/portal_advisory_worker.py`: a held process-worker adapter that refuses activation without separately reviewed qualification and explicit route selection.

Governed identity/runtime, provider selection, execution, and memory authorization belong to their respective host components. No model response is itself an authorization or verification receipt.

## Local evaluation (Windows reference implementation)

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
