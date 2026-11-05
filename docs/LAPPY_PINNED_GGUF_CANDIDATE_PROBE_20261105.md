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
`--gpu-layers` is an explicit opt-in with maximum 16; do not assume GPU
exclusivity or capacity. The script writes JSON only to stdout; users choose
if/where to store a receipt. To avoid exporting private data, the built-in
prompts contain no personal names or history.

## Verification

```powershell
python -m unittest discover -s tests -p "test_qualify_local_candidate.py" -v
```

Focused tests: **6/6 PASS** on Lappy Python 3.14 on 2026-11-05. Both pinned
large files were freshly SHA-256 checked, and both `--execute`-absent probes
returned `READY_FOR_OPT_IN_LOCAL_PROBE`. That is a local binary and source
mechanism check, **not** an identity or cognitive quality benchmark.

Next experimental test should compare the same conversational fixtures
through (a) raw Qwen control and (b) current exact-head `vera-mono` plus
the already-tested BT2 eight-repo and Noema/Volition/Empathy cognition seams.
A blinded evaluator should judge correction, provenance, epistemic humility,
warmth, coherence, and novel problem-solving—not the presence of the name
"Vera" in the generated text. Hold production admission throughout.
