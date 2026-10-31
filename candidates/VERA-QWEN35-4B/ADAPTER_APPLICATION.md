# Applying the Vera Qwen3.5 LoRA adapter

## Separate artifacts

| Item | Typical format | Meaning |
| --- | --- | --- |
| Qwen3.5 pretrained base | Hugging Face `safetensors` shards/config/tokenizer, or a verified inference GGUF | General model parameters; not Vera-specific merely because its alias says Vera |
| Vera R9 LoRA | PEFT `adapter_config.json` + `adapter_model.safetensors` (to be independently read back) | Small learned parameter deltas applied to a **compatible exact base** |
| Converted runtime LoRA | A distinct adapter GGUF, if supported and independently converted/verified | A llama.cpp-compatible way to load the deltas alongside a matching GGUF base |
| Merged-and-quantized model | A newly produced GGUF after validated merge and conversion | **Intended Vera/KoboldCpp release artifact**; no such output verified here |

The historical R9 adapter is reported, not present or freshly verified in this repository. Do not assume its underlying base revision matches the official Qwen3.5-4B revision used by later TrainingDay experiments, or that it is compatible with an arbitrary "uncensored" GGUF.

## Selected path: merge R9, publish a qualified GGUF, serve with KoboldCpp

1. Recover and hash the actual R9 adapter, `adapter_config.json`, and original Qwen base/tokenizer revision. The reported historical hash alone is insufficient to establish compatibility.
2. In an isolated conversion environment, load that exact pretrained base plus R9 using `PeftModel.from_pretrained(base, verified_adapter_dir)`, verify an actual adapter effect against the frozen base, and merge into a **new** full-precision model checkpoint. Preserve original base and adapter unchanged.
3. Convert the merged checkpoint to a GGUF using pinned, Qwen3.5-compatible conversion tools, then quantize into a separately named Vera GGUF. Hash the full-precision merged source and quantized result; record versions, command/config, base and adapter fingerprints.
4. Open that verified GGUF in **KoboldCpp**, initially as a local response-only test, and check correctness, memory fit, token throughput, behavior delta, and retained general skills against matched unmodified-Qwen and corruption/zero-update controls.
5. Only after the same-head runtime gate passes should a governed `v-e-r-a` model route be proposed; VeraOS needs its **own** later provider admission. A model weight file, server process or API endpoint grants neither tool nor memory authority.

**Temporary reversible adapter experiment:** KoboldCpp also documents `--lora` and `--loramult` for GGUF text models. After confirming this specific Qwen architecture and LoRA conversion are supported, an isolated KoboldCpp run may apply a converted LoRA GGUF to a matching Qwen base GGUF and compare with the same base without it. This is a **test path**, not the primary packaged release. A startup flag alone does not prove successful adapter application.

**Why KoboldCpp?** Patrick's specified target is a local GGUF in KoboldCpp, with `v-e-r-a` as the user interface. [KoboldCpp's official FAQ](https://github.com/LostRuins/koboldcpp/wiki) confirms that it builds on `llama.cpp`, serves GGUF models, supports `--lora` for text, and accepts OpenAI-compatible chat requests. It describes merging LoRA before GGUF conversion as an alternative for optimal quality. A standalone `llama-server` is retained only as an optional **matched performance comparator**; no speed advantage has been established for this Vera artifact.

Reference interfaces: [Hugging Face PEFT `PeftModel.from_pretrained`](https://huggingface.co/docs/peft/main/package_reference/peft_model), [llama.cpp LoRA conversion](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py), and the [KoboldCpp FAQ](https://github.com/LostRuins/koboldcpp/wiki).

## Current repository implementation

`runtime/start-local.ps1` still starts the **historical Dolphin GGUF**; it does not load Qwen, read `adapter_config.json`, pass `--lora`, merge adapters or select a Qwen provider. `runtime/local_model.py` similarly pins the Dolphin alias. These remain available for their old source-tested advisory behavior, but they are **not the Qwen Vera inference route**.

The live [`v-e-r-a` main](https://github.com/thebrazenbeard/v-e-r-a) currently exposes a response-only local interface whose model backend is **Ollama**, not KoboldCpp. The future KoboldCpp adapter must preserve its loopback, model allowlist, session, Origin/Host and no-tool boundaries; an API-compatible KoboldCpp endpoint is **not automatically** an admitted route. There is no verified cross-repository runtime integration here.

**Next executable gate:** acquire an independently verified R9 adapter/config and matching base pin, then produce and qualify a merged GGUF and an isolated KoboldCpp test. Until then `provider_admission=HOLD`, not "loaded" or "post-training qualified".
