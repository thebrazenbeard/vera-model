"""Source-contract checks: selection is not runtime/model qualification."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrimarySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = json.loads((ROOT / "contracts/primary-model.json").read_text(encoding="utf8"))
        cls.oss = json.loads((ROOT / "candidates/VERA-GPT-OSS4B_Q4_K_M/manifest.json").read_text(encoding="utf8"))

    def test_primary_is_qwen_not_a_deployed_model(self):
        p = self.primary
        self.assertEqual(p["selected_family"], "Qwen3.5")
        self.assertEqual(p["selected_development_lineage"], "Vera V10R3R9 LoRA")
        self.assertEqual(p["selection_state"], "PRIMARY_DEVELOPMENT_NOT_DEPLOYED")
        self.assertEqual(p["provider_admission"], "HOLD")
        self.assertIs(p["protected_effect_authority"], False)

    def test_adapter_cannot_be_mislabeled_as_gguf_or_verified(self):
        a = self.primary["historical_r9_adapter"]
        self.assertEqual(a["artifact_type"], "PEFT_LORA_ADAPTER_NOT_GGUF_BASE")
        self.assertRegex(a["sha256_reported"], r"^[0-9a-f]{64}$")
        self.assertIs(a["artifact_bytes_in_this_repository"], False)
        self.assertIs(a["exact_base_and_tokenizer_binding_verified_here"], False)
        self.assertIs(a["active_inference_load_verified_here"], False)

    def test_r10_and_gguf_must_remain_on_hold(self):
        r = self.primary["r10_continuation"]
        self.assertEqual(r["status"], "HELD_BEFORE_OPTIMIZER")
        self.assertIs(r["completed_optimizer_update_verified"], False)
        self.assertIs(r["heldout_improvement_verified"], False)
        g = self.primary["gguf_serving"]
        self.assertIsNone(g["selected_gguf_sha256"])
        for field in ("adapter_converted_to_gguf_verified", "adapter_merged_into_gguf_verified", "llama_cpp_qwen_lora_load_verified"):
            self.assertIs(g[field], False)

    def test_oss4b_is_historical_without_rewriting_old_provenance(self):
        self.assertEqual(self.oss["selection"]["priority"], "SHELVED_HISTORICAL_NOT_PRIMARY")
        self.assertEqual(self.oss["selection"]["primary_successor"], "Qwen3.5 Vera V10R3R9 LoRA")
        self.assertTrue(self.oss["selection"]["reselection_requires_explicit_new_decision"])
        self.assertIs(self.oss["training"]["model_weight_updates_observed"], False)

    def test_existing_launcher_explicitly_is_not_qwen(self):
        ps = (ROOT / "runtime/start-local.ps1").read_text(encoding="utf8")
        self.assertIn("vera-dolphin-glm24b-iq2xs", ps)
        self.assertNotIn("--lora", ps)
        readme = (ROOT / "README.md").read_text(encoding="utf8")
        self.assertIn("Qwen3.5 Vera V10R3R9", readme)
        self.assertIn("Historical / non-primary alternatives", readme)


if __name__ == "__main__":
    unittest.main()
