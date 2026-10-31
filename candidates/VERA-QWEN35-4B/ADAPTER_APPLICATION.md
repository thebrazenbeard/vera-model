# Applying the Vera Qwen3.5 LoRA adapter

## Separate artifacts

| Item | Typical format | Meaning |
| --- | --- | --- |
| Qwen3.5 pretrained base | Hugging Face `safetensors` shards/config/tokenizer, or a verified inference GGUF | General model parameters; not Vera-specific merely because its alias says Vera |
| Vera R9 LoRA | PEFT `adapter_config.json` + `adapter_model.safetensors` (to be independently read back) | Small learned parameter deltas applied to a **compatible exact base** |
| Converted runtime LoRA | A distinct adapter GGUF, if supported and independently converted/verified | A llama.cpp-compatible way to load the deltas alongside a matching GGUF base |
| Merged-and-quantized model | A newly produced GGUF after validated merge and conversion | One distributable inference file, but **none is verified or selected here** |

The historical R9 adapter is reported, not present or freshly verified in this repository. Do not assume its underlying base revision matches the official Qwen3.5-4B revision used by later TrainingDay experiments, or that it is compatible with an arbitrary "uncensored" GGUF.

## Two possible inference paths — neither activated here

**Transformers / PEFT:** Load the **exact original trainable base and tokenizer**, then load the matching R9 PEFT adapter with `PeftModel.from_pretrained(base, verified_adapter_dir)`. Confirm the adapter is active, the base/target module mapping matches the receipt, and inference differs appropriately from a frozen-base control. The LoRA weights are **not** applied by renaming a model. This remains an unexecuted procedure until exact adapter/base metadata and bytes are available.

**llama.cpp / GGUF:** First verify an exact compatible Qwen base GGUF. If the PEFT architecture is supported by the pinned version of `llama.cpp`, convert the PEFT adapter with its `convert_lora_to_gguf.py` tool; verify the resulting adapter GGUF and launch a **separate test instance** using `llama-server --model <verified-base.gguf> --lora <verified-adapter.gguf>`. Compare with the same base without LoRA and check that the server actually reports an applied adapter. `--lora` is a runtime argument, not proof of compatibility or correct behavior. Do not merge/convert an R9 adapter into a final GGUF before exact-base and regression checks.

Official source interfaces: [Hugging Face PEFT `PeftModel.from_pretrained`](https://huggingface.co/docs/peft/main/package_reference/peft_model), [llama.cpp LoRA conversion](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py) and [llama-server `--lora`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Current repository implementation

`runtime/start-local.ps1` still starts the **historical Dolphin GGUF**; it does not load Qwen, read `adapter_config.json`, pass `--lora`, merge adapters or select a Qwen provider. `runtime/local_model.py` similarly pins the Dolphin alias. These remain available for their old source-tested advisory behavior, but they are **not the Qwen Vera inference route**.

**Next executable gate:** acquire an independently verified R9 adapter/config and matching base pin, then add a separately testable, opt-in read-only Qwen inference loader and controlled frozen-vs-adapted comparison. Until then `provider_admission=HOLD`, not "loaded" or "post-training qualified".
