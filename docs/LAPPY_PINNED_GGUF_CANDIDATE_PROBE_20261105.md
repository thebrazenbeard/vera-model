# Lappy pinned candidate GGUF probe — 2026-11-05

This is an **opt-in, read-only, local language-model comparison**. It does not
install or promote Vera, start services, grant tools or memory, write model
weights, or establish independent behavioral qualification.

The existing GLM24B sidecar and its `runtime/start-local.ps1` are unchanged.

## Exactly what is checked

`runtime/qualify_local_candidate.py` uses two byte-pinned files and
`llama-cli` from the user's PATH. The candidate must match both size and
SHA-256 before the script even considers executing inference. If a model is
missing or different, it returns a typed HOLD without launching the runner.

**Evidence limit:** A preflight hash cannot prove which bytes `llama-cli`
loads later by reopening the path. In `--execute` mode the tool rehashes
both model and resolved runner executable **after** the invocation and emits
`HOLD` if either readback differs or fails. Even matching pre/post hashes
cannot rule out a transient in-between swap. Consequently
`model_execution_evidence=UNVERIFIED_ON_EXEC`,
`model_bytes_loaded_verified=false`, and
`runner_bytes_executed_verified=false` always. Runner source/build identity,
dynamic libraries, and runtime behavior remain outside this contract.

- `gptoss-pruned4-q4`: 3,233,019,552 bytes, SHA-256
  `38b0279e4294beb54f97ac23051674174b34b2acd6c5e6532860c72298789586`.
  Upstream pruned GPT-OSS 20B to four experts per layer **without retraining**;
  this is not GPT-6.1 or an original ChatGPT checkpoint. Public source
  provenance: `candidates/VERA-GPT-OSS4B_Q4_K_M/README.md`.
- `qwen35-4b-q4-control`: 2,707,513,696 bytes, SHA-256
  `25082a7dd3776cc3c741c6347d3bd04523f05796607b3fbc32fa3a25dfa1418c`.
  This binds only the local Lappy artifact observed on 2026-11-05. It does not
  authenticate a source publisher, upstream commit, or training chain.

No private conversation text is used. Prompt fixtures inspect model provenance,
correction, and reasoning-limits behavior. The generated response is
**untrusted evidence**, never authoritative memory or a license to act.

## Verified Lappy observation (NOT a release)

Hardware at a read-only probe: 31.73 GiB RAM, about 13.7 GiB free;
RTX 3050 Laptop GPU 4096 MiB with 2613 MiB free per `nvidia-smi`.
`llama-cli` was version `0.4.1-dev`, build `11026`, commit `b49650adb`.
The runner enumerated Vulkan0 RTX3050 and Vulkan1 Intel iGPU; 8-layer Qwen
offload worked but was not faster than the short CPU probe.

Fresh controlled direct CLI tests, CPU `-ngl 0`, `-c 2048`,
`--single-turn`, `--reasoning off`, `--temp 0.2`:

- GPT-OSS pruned: binary loaded and generated ~11.2 tokens/second; on two
  bounded identity probes it repeated the prompt/meta text instead of giving
  a useful direct answer. **Conversational usability HOLD**.
- Qwen3.5 4B: binary loaded, ~8.0 tokens/second. Without an operational
  system prompt it answered its underlying identity as Qwen3.5 by Alibaba.
  With a generic Vera persona prompt it could reply in a casual conversational
  voice but did not demonstrate durable memory, original ChatGPT weights, or
  the richer multi-repository cognition route. **Vera qualification HOLD**.

The current source contains only the reproducible probing mechanism; it
does not serialize prior runs into authority, pick a production provider,
or declare any model ready for deployment.

## Usage on an already configured workstation

With a Python 3.10+ interpreter and existing `llama-cli` executable:

```powershell
python runtime/qualify_local_candidate.py --candidate gptoss-pruned4-q4 --model-path "D:\VERA\models\gguf\vera-gpt-oss4b-q4-k-m\gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-q4_k_m.gguf"

# Explicit --execute opts into ONE local inference invocation.
python runtime/qualify_local_candidate.py --candidate qwen35-4b-q4-control --model-path "D:\VERA\models\gguf\vera-qwen35-4b-q4-k-m-candidate-20261029\Qwen3.5-4B-Q4_K_M.gguf" --execute --prompt-index 1
```

Models are not downloaded by this tool. It defaults to CPU-only, bounded
generation, 120-second timeout, and a single request per invocation.
Live stdout/stderr are drained concurrently into bounded in-memory excerpts
(first 8,000 / last 2,000 raw bytes), discarding any excess rather than
buffering unbounded process output. Timeout terminates the subprocess.
Read/permission errors and path changes are typed HOLDs. Generated excerpts
may still contain sensitive text if you pass custom prompts; do not publish
receipts without inspecting them.

Before explicit execution the tool now checks live free system RAM against a
conservative floor (maximum of 4 GiB or twice file size plus 1 GiB). With
`--gpu-layers` set it also requires an NVIDIA free-VRAM snapshot from
`nvidia-smi` and a conservative floor (maximum of 1 GiB or one-quarter of
model size). Unavailable or insufficient resources produce a typed HOLD
without starting the model runner. These thresholds are guardrails, **not**
proof the actual backend, selected GPU, concurrent usage, or load will fit.
CPU-only remains the default.

On timeout, Windows attempts `taskkill /T /F` for the runner PID and then a
direct-process fallback; the receipt records the attempted cleanup but does
not guarantee all descendant processes terminated. Do not use this probe
as an isolation sandbox or run untrusted executables.
`--gpu-layers` is an explicit opt-in with maximum 16; do not assume GPU
exclusivity or capacity. The script writes JSON only to stdout; users choose
if/where to store a receipt. To avoid exporting private data, the built-in
prompts contain no personal names or history.

## Verification

```powershell
python -m unittest discover -s tests -p "test_qualify_local_candidate.py" -v
```

Original focused tests: **6/6 PASS** on Lappy Python 3.14 on 2026-11-05.
On 2026-11-06, additional hostile tests cover a synthetic 4-byte model swap
mid-run, runner path replacement, read/permission failure, and 120KB/150KB
process output. A second test wave covers low RAM/VRAM fail-closed behavior
and best-effort Windows child-tree cleanup. **13/13 focused tests PASS** on
Lappy Python 3.14. No real GGUF loading occurs during these unit tests. Both pinned
large files were freshly SHA-256 checked, and both `--execute`-absent probes
returned `READY_FOR_OPT_IN_LOCAL_PROBE`. That is a local binary and source
mechanism check, **not** an identity or cognitive quality benchmark.

Next experimental test should compare the same conversational fixtures
through (a) raw Qwen control and (b) current exact-head `vera-mono` plus
the already-tested BT2 eight-repo and Noema/Volition/Empathy cognition seams.
A blinded evaluator should judge correction, provenance, epistemic humility,
warmth, coherence, and novel problem-solving—not the presence of the name
"Vera" in the generated text. Hold production admission throughout.
