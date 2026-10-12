"""Fail-closed local inference and advisory packet contract tests."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "integration"))
from local_model import LocalModelHold, advise, probe_model, verify_model, EXPECTED_SHA256
from portal_advisory_worker import process, validate_packet, make_receipt


def packet() -> dict:
    return {
        "run_id": "test-run",
        "subject_id": "portal",
        "repository": "thebrazenbeard/portal",
        "node_id": "lappy",
        "delivery_fencing_token": "bound-fence-1",
        "exact_head": "a" * 40,
        "plan_sha256": "b" * 64,
        "frontier": "Review the source and propose the safest next step.",
        "advisory_only": True,
        "execution_authorized": False,
        "protected_effects_authorized": False,
        "source_mutation_authorized": False,
        "target_ref_mutation_authorized": False,
        "execution_effect_class": None,
        "execution_promotion": None,
    }


class ModelTests(unittest.TestCase):
    def test_pinned_model_rejects_short_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "candidate.gguf"
            p.write_bytes(b"not a model")
            with self.assertRaisesRegex(LocalModelHold, "size mismatch"):
                verify_model(p)

    def test_probe_requires_exact_alias(self):
        with patch("local_model._request", return_value={
            "data": [{"id": "unrelated-qwen"}]}):
            with self.assertRaisesRegex(LocalModelHold, "alias"):
                probe_model("z" * 64)

    def test_probe_exact_alias(self):
        with patch("local_model._request", return_value={
            "data": [{"id": "vera-dolphin-glm24b-iq2xs"}]}):
            self.assertEqual(probe_model("z" * 64)["status"], "ENDPOINT_RESPONDS")

    def test_advice_does_not_send_tools_or_request_effects(self):
        with patch("local_model._request", return_value={
            "choices": [{"message": {"content": "Review first."}}]}) as send:
            self.assertEqual(advise("Audit this repository", "z" * 64), "Review first.")
            payload = send.call_args.args[2]
            self.assertNotIn("tools", payload)
            self.assertNotIn("tool_choice", payload)
            self.assertEqual(payload["stream"], False)
            self.assertEqual(payload["model"], "vera-dolphin-glm24b-iq2xs")

    def test_tool_call_response_is_rejected(self):
        with patch("local_model._request", return_value={
            "choices": [{"message": {"content": "done",
                                     "tool_calls": [{"function": "erase"}]}}]}):
            with self.assertRaisesRegex(LocalModelHold, "tool calls"):
                advise("Audit this repository", "z" * 64)

    def test_bounded_prompt_and_tokens(self):
        with self.assertRaisesRegex(LocalModelHold, "bound"):
            advise("x" * 12001, "z" * 64)
        with self.assertRaisesRegex(LocalModelHold, "bound"):
            advise("hi", "z" * 64, max_tokens=999)

    def test_bad_api_path_denied(self):
        from local_model import _request
        with self.assertRaisesRegex(LocalModelHold, "unsupported"):
            _request("/v1/admin", "z" * 64)


class AdvisoryWorkerTests(unittest.TestCase):
    def test_held_without_route_admission(self):
        receipt, answer = process(packet(), allowed=False)
        self.assertIsNone(answer)
        self.assertEqual(receipt["receipt_class"], "HELD")

    def test_denies_effect_authority(self):
        for field in ("protected_effects_authorized", "source_mutation_authorized",
                      "target_ref_mutation_authorized", "execution_authorized"):
            bad = packet()
            bad[field] = True
            with self.subTest(field=field), self.assertRaises(LocalModelHold):
                validate_packet(bad)

    def test_denies_promotion(self):
        bad = packet()
        bad["execution_promotion"] = {"kind": "mutation"}
        with self.assertRaises(LocalModelHold):
            validate_packet(bad)

    def test_missing_fence_denied(self):
        bad = packet()
        bad.pop("delivery_fencing_token")
        with self.assertRaisesRegex(LocalModelHold, "fencing"):
            validate_packet(bad)

    def test_advice_only_after_explicit_allowed_flag(self):
        with patch("portal_advisory_worker.advise", return_value="Review these changes.") as ask:
            receipt, answer = process(packet(), allowed=True, key="z" * 64)
            self.assertEqual(receipt["receipt_class"], "SUCCEEDED_NO_EFFECT")
            self.assertEqual(answer, "Review these changes.")
            self.assertEqual(ask.call_count, 1)

    def test_receipt_cannot_claim_source_effect(self):
        with self.assertRaises(ValueError):
            make_receipt("SUCCEEDED_SOURCE_CHANGE", "invalid")


class IntegrationContractTests(unittest.TestCase):
    def test_veraos_candidate_remains_held(self):
        provider = json.loads(
            (ROOT / "contracts/veraos-provider.json").read_text(encoding="utf8")
        )
        self.assertEqual(provider["schema"], "VERA_OS_COGNITION_PROVIDER_V1")
        self.assertEqual(provider["qualification"]["status"], "HOLD")
        self.assertEqual(provider["admission"]["status"], "HELD")
        self.assertFalse(provider["provenance"]["qualification_issued"])
        self.assertEqual(provider["admission"]["selector_owner"],
                         "thebrazenbeard/portal")
        self.assertTrue(all(value is False
                            for value in provider["authority"].values()))

    def test_donor_and_quant_are_exact_and_unqualified(self):
        manifest = json.loads(
            (ROOT / "contracts/model-lineage.json").read_text(encoding="utf8")
        )
        self.assertEqual(manifest["gguf_sha256"], EXPECTED_SHA256)
        self.assertEqual(manifest["gguf_bytes"], 7207034784)
        self.assertEqual(manifest["local_interface"]["gpu_offload_layers"], 14)
        self.assertFalse(manifest["model_behavior_qualified"])
        self.assertFalse(manifest["model_selected_by_portal"])
        self.assertFalse(manifest["production_deployed"])
        self.assertFalse(manifest["adapter_training_executed"])


if __name__ == "__main__":
    unittest.main()
