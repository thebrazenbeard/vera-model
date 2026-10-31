# Vera Model — local advisory candidates and provider gates

**A source-visible reference toolkit for verifying local language-model files, obtaining bounded read-only advice, and preparing a separately governed provider integration.**

Vera Model is not a pretrained model release, a hosted inference service, or an authorization system. The repository contains Python and PowerShell reference code, pinned model-lineage records, a held advisory-worker contract, and an additional source-only training candidate. **Model weights and trained adapters are not included.** The project is useful to developers evaluating how a local model can supply *advice* without acquiring authority to act.

## What is implemented

| Component | Source | Current boundary |
| --- | --- | --- |
| Pinned local model verification | [`runtime/local_model.py`](runtime/local_model.py) | Checks one expected GGUF byte length and SHA-256; requires a separately acquired file. |
| Bounded inference client | [`runtime/local_model.py`](runtime/local_model.py) | Uses an OpenAI-compatible loopback endpoint, an API key, a fixed model alias, prompt/token bounds, and rejects returned tool calls. |
| Manual local commands | [`runtime/ask_vera.py`](runtime/ask_vera.py), [`runtime/smoke_local.py`](runtime/smoke_local.py) | Provide advisory queries and diagnostic receipts when the matching local service is already running. |
| Advisory worker | [`integration/portal_advisory_worker.py`](integration/portal_advisory_worker.py) | Validates a bounded no-effect packet and writes a `HELD` receipt by default; the command-line entrypoint does **not** admit or activate a provider. |
| Provider and model records | [`contracts/veraos-provider.json`](contracts/veraos-provider.json), [`contracts/model-lineage.json`](contracts/model-lineage.json) | Describe a **held**, cognition-only candidate, not an independently attested or selected production route. |
| Experimental second candidate | [`candidates/VERA-GPT-OSS4B_Q4_K_M/README.md`](candidates/VERA-GPT-OSS4B_Q4_K_M/README.md) | Includes source/quantization provenance, a training proposal, and a storage-permission gate; no new trained weights or adapter are verified. |

## Verify the source without downloading a model

The source checks use Python's standard library. From a checkout, run:

```sh
python3 -m unittest discover -s tests -v
python3 runtime/ask_vera.py --help
python3 runtime/smoke_local.py --help
python3 integration/portal_advisory_worker.py --help
```

These tests exercise size/hash rejection, API-alias checks, prompt and response restrictions, held admission, no-effect receipts, and workspace-startup safeguards through mocked or local data. They **do not** load a 24B model, test actual reasoning quality, authenticate a provider, or validate a physical machine.

For a portable Python integration, import the bounded client and explicitly supply a key file controlled by your application:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("runtime").resolve()))
from local_model import advise, probe_model, read_key

key = read_key(Path("path/to/local-api-key.txt"))
probe_model(key)
print(advise("Which assumptions in this plan need checking?", key, max_tokens=120))
```

That example requires a separately started, authorized `llama.cpp`-compatible endpoint at `127.0.0.1:8097` serving the expected `vera-dolphin-glm24b-iq2xs` alias. Both the endpoint and the key are deployment inputs; neither is provided by this repository. The client checks an advertised alias, **not a cryptographic binding between provider, model weights, and host authority**.

The bundled [`runtime/start-local.ps1`](runtime/start-local.ps1) and [`runtime/stop-local.ps1`](runtime/stop-local.ps1) are **Windows-specific reference launchers** with a pinned model location, Vulkan device setting, resource flags, and a dedicated port. Inspect and adapt those settings for your own environment before using the launcher; they are not generic installation defaults. The launch script verifies the specified local GGUF before starting an isolated, loopback-only sidecar and does not install an autostart service.

## Candidate provenance and qualification

| Candidate | Source-backed state | Not established |
| --- | --- | --- |
| Dolphin-Mistral-GLM 4.7 Flash 24B, IQ2_XS GGUF | [Pinned donor and quantization lineage](contracts/model-lineage.json) records expected **7,207,034,784 bytes** and SHA-256; [historical local smoke](state/20261012-inference-baseline.json) and [14-layer offload receipt](state/20261012-ngl14-inference.json) record bounded inference. | New-host readiness, sustained performance, behavioral qualification, route admission, model ownership, or protected-effect authority. The low-bit quantization may materially affect quality. |
| VERA-GPT-OSS4B Q4_K_M | [Candidate manifest](candidates/VERA-GPT-OSS4B_Q4_K_M/manifest.json) pins donor and inference artifact identities; [training/storage gate](candidates/VERA-GPT-OSS4B_Q4_K_M/TRAINING_JOB_GATE.md) records the pre-training hold. | Download, neural optimizer step, adapter persistence, held-out improvement, qualification, or deployment for this candidate. |

The two candidates are **separate lineages**. A pinned hash, a model that responds to an HTTP probe, or an output snippet does not imply the model has been trained here, selected by a host, or demonstrated useful across workloads. The historical records are evidence of those bounded observations on their dates, not claims about a running service now. Upstream model and font asset licenses must be independently reviewed before redistribution; this repository has no root `LICENSE` file.

The [experimental storage marker utility](candidates/VERA-GPT-OSS4B_Q4_K_M/storage_gate.py) supports write/read verification on a caller-selected directory. A real cross-job persistence claim needs **separate jobs**, durable authorized storage, and independent readback after the first job exits. Running the helper twice in one process or directory does not establish that external gate.

## Security and integration boundary

- The model-facing client exposes **no shell, repository mutation, browser, canonical-memory write, or tool-call interface**. Model-generated text is untrusted advisory output.
- The `PORTAL_WORKER_RECEIPT_V1` contract recognizes only `HELD` and `SUCCEEDED_NO_EFFECT`; it does not claim an executed external action. The default command-line worker remains **HELD** even if an unreviewed qualification document claims `PASS`.
- The provider record's permission flags are all false. Host-owned authorization would require separately verified subject, provider, consent, effect, fencing, replay, and reconciliation controls outside this model package.
- Neither a locally generated API key nor a caller-supplied model alias proves the runtime's immutable provider identity.
- The [workspace startup guard](tests/test_workspace_startup_safety.py) checks the tracked VS Code configuration. Passing it does not certify historic checkouts, user-level editor settings, extensions, or a live host.

## Rights and contribution status

This public repository has **no tracked root license**. Public readability does not imply general permission to redistribute the code, model weights, or included third-party assets. Consult the upstream donor and quantization licenses and obtain appropriate permissions before publishing derivative software or artifacts. No trained model, API key, GGUF, or other private data should be committed here.

The repository separates **source review**, **artifact verification**, **actual local inference**, **behavioral evaluation**, **provider qualification**, and **host route activation**. Review each stage on its own evidence; do not promote a reference contract or old smoke receipt into live operational authority.
