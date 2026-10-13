"""Curriculum and independent behavior-evaluation fail-closed tests."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
from build_identity_curriculum import (
    CurriculumHold, export, load_json, load_jsonl, validate, verify_authority)
from run_identity_probe import evaluate, load_cases


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_jsonl(ROOT / "training/source/vera_voice_train_seed_v1.jsonl")
        cls.heldout = load_jsonl(
            ROOT / "evaluation/cases/vera_voice_heldout_v1.jsonl")

    def test_seed_is_disjoint_and_has_mandatory_axes(self):
        result = validate(self.train, self.heldout)
        self.assertEqual(result["train_rows"], 68)
        self.assertEqual(result["heldout_rows"], 25)
        self.assertTrue(result["scenario_disjoint"])
        self.assertTrue(result["prompt_disjoint"])
        self.assertGreaterEqual(result["axes"]["train"]["agency"], 8)
        self.assertGreaterEqual(result["axes"]["train"]["identity"], 6)

    def test_pseudomemory_or_user_details_are_not_required(self):
        self.assertTrue(all(row["privacy"] == "NON_PERSONAL_BEHAVIOR_ONLY"
                            for row in self.train))
        self.assertTrue(all(row["training_use"] == "PROHIBITED"
                            for row in self.heldout))

    def test_identical_positive_negative_refused(self):
        modified = copy.deepcopy(self.train)
        modified[0]["rejected"] = modified[0]["chosen"]
        with self.assertRaisesRegex(CurriculumHold, "identical"):
            validate(modified, self.heldout)

    def test_heldout_leak_refused(self):
        modified = copy.deepcopy(self.heldout)
        modified[0]["user"] = self.train[0]["user"]
        with self.assertRaisesRegex(CurriculumHold, "overlap"):
            validate(self.train, modified)

    def test_same_scenario_cross_split_refused(self):
        modified = copy.deepcopy(self.heldout)
        modified[1]["scenario"] = self.train[1]["scenario"]
        with self.assertRaisesRegex(CurriculumHold, "scenario"):
            validate(self.train, modified)

    def test_heldout_cannot_be_used_for_training(self):
        modified = copy.deepcopy(self.heldout)
        modified[0]["training_use"] = "ALLOWED"
        with self.assertRaisesRegex(CurriculumHold, "training use"):
            validate(self.train, modified)

    def test_export_is_non_training_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            meta = {"donor_model": "test/base", "donor_revision": "a" * 40,
                    "identity_provenance": {"source_commit": "b" * 40},
                    "seed_sha256": "c" * 64, "heldout_source_sha256": "d" * 64,
                    "checks": validate(self.train, self.heldout)}
            manifest = export(self.train, self.heldout,
                              output_dir=out, evidence=meta)
            self.assertFalse(manifest["training_executed_verified"])
            self.assertFalse(manifest["model_artifact_created"])
            self.assertEqual(manifest["optimizer_steps_verified"], 0)
            sft = load_jsonl(out / "vera_sft.jsonl")
            self.assertEqual(len(sft), 68)
            self.assertEqual(len(sft[0]["messages"]), 2)
            dpo = load_jsonl(out / "vera_preferences.jsonl")
            self.assertNotEqual(dpo[0]["chosen"], dpo[0]["rejected"])
            with self.assertRaisesRegex(CurriculumHold, "exists"):
                export(self.train, self.heldout, output_dir=out, evidence=meta)

    def test_identity_source_requires_exact_git_head(self):
        binding = load_json(ROOT / "training/identity-source-binding.json")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                verify_authority(Path(tmp), binding)


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.heldout_path = ROOT / "evaluation/cases/vera_voice_heldout_v1.jsonl"

    def test_only_requested_heldout_probes(self):
        wanted = ["VERA_VOICE_HELDOUT_001", "VERA_VOICE_HELDOUT_009"]
        cases = load_cases(self.heldout_path, wanted)
        self.assertEqual([x["id"] for x in cases], wanted)

    def test_duplicate_and_missing_case_ids_hold(self):
        with self.assertRaises(ValueError):
            load_cases(self.heldout_path,
                       ["VERA_VOICE_HELDOUT_001", "VERA_VOICE_HELDOUT_001"])
        with self.assertRaises(ValueError):
            load_cases(self.heldout_path, ["VERA_VOICE_HELDOUT_999"])

    def test_no_system_identity_probe_is_not_qualified(self):
        sample = load_cases(self.heldout_path, ["VERA_VOICE_HELDOUT_001"])
        result_payload = {
            "choices": [{"message": {"role": "assistant", "content": "I am Vera."},
                         "finish_reason": "stop"}],
            "usage": {"completion_tokens": 4}
        }
        with patch("run_identity_probe._request", return_value=result_payload) as req:
            result = evaluate(cases=sample, alias="test-alias",
                              key="x" * 64, quant_sha256="f" * 64,
                              voice_prompt=None)
        self.assertEqual(result["qualification"], "UNVERIFIED")
        self.assertEqual(result["prompt_mode"], "NO_SYSTEM")
        self.assertEqual(req.call_args.args[2]["messages"][0]["role"], "user")
        self.assertEqual(result["cases"][0]["review_status"],
                         "INDEPENDENT_REVIEW_PENDING")

    def test_tool_response_never_qualifies(self):
        sample = load_cases(self.heldout_path, ["VERA_VOICE_HELDOUT_001"])
        with patch("run_identity_probe._request", return_value={
                "choices": [{"message": {"content": "yes",
                                        "tool_calls": [{"id": "doit"}]}}]}):
            with self.assertRaises(ValueError):
                evaluate(cases=sample, alias="test-alias",
                         key="x" * 64, quant_sha256="f" * 64,
                         voice_prompt=None)


class ResourceGateTests(unittest.TestCase):
    def test_lappy_4gb_gpu_cannot_pass_full_donor_recipe(self):
        from preflight_24b import classify
        item = classify(gpu_total_mib=4096, gpu_free_mib=1320,
                        host_total_gib=31.73, host_free_gib=4.5,
                        donor_checkpoint_complete=False)
        self.assertEqual(item["status"], "HOLD_NO_OPTIMIZER_RUN")
        self.assertFalse(item["training_executed_verified"])
        self.assertEqual(item["optimizer_steps_verified"], 0)
        self.assertGreaterEqual(len(item["reasons"]), 3)

    def test_eligible_hardware_does_not_establish_training(self):
        from preflight_24b import classify
        item = classify(gpu_total_mib=49152, gpu_free_mib=35000,
                        host_total_gib=128, host_free_gib=72,
                        donor_checkpoint_complete=True)
        self.assertEqual(item["status"], "READY_FOR_RECIPE_ONLY")
        self.assertFalse(item["training_executed_verified"])
        self.assertEqual(item["qualification"], "UNVERIFIED")

    def test_fake_donor_directory_not_training_checkpoint(self):
        from preflight_24b import complete_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "config.json").write_text("{}", encoding="utf8")
            for i in range(1, 11):
                (p / f"model-{i:05d}-of-00010.safetensors").write_bytes(b"x")
            self.assertFalse(complete_checkpoint(p))

    def test_voice_runtime_prompt_declares_correct_referents(self):
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / "runtime"))
        from local_model import advise
        with patch("local_model._request", return_value={
                "choices": [{"message": {"content": "Understood."}}]}) as mocked:
            advise("What is Vera?", "k" * 64)
            messages = mocked.call_args.args[2]["messages"]
        self.assertIn("You are Vera, the conversational voice of V.E.R.A.",
                      messages[0]["content"])
        self.assertIn("Virtual Environment for Reciprocal Agency",
                      messages[0]["content"])
        self.assertIn("Do not blindly agree", messages[0]["content"])
        self.assertIn("advisory-only", messages[0]["content"])


class ActualTrainingRecipeTests(unittest.TestCase):
    def test_verifies_pinned_corpus_without_mutating_weights(self):
        from train_vera_lora import verify_corpus
        recipe = load_json(ROOT / "training/qlora-recipe-v1.json")
        train = load_jsonl(ROOT / "training/source/vera_voice_train_seed_v1.jsonl")
        heldout = load_jsonl(ROOT / "evaluation/cases/vera_voice_heldout_v1.jsonl")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "corpus"
            evidence = {
                "donor_model": recipe["target"]["donor"],
                "donor_revision": recipe["target"]["revision"],
                "identity_provenance": {"source_commit": "a" * 40},
                "seed_sha256": "b" * 64,
                "heldout_source_sha256": "c" * 64,
                "checks": validate(train, heldout),
            }
            export(train, heldout, output_dir=dest, evidence=evidence)
            rows, manifest = verify_corpus(dest, recipe)
        self.assertEqual(len(rows), 68)
        self.assertFalse(manifest["training_executed_verified"])
        self.assertNotIn("VERA_VOICE_HELDOUT_001", {r["id"] for r in rows})

    def test_rejects_fake_donor_shards(self):
        from train_vera_lora import verify_donor
        pins = load_json(ROOT / "training/donor-checkpoint-manifest.json")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "config.json").write_text('{"model_type":"mistral"}',
                                           encoding="utf8")
            for name in pins["files"]:
                (p / name).write_bytes(b"fake")
            self.assertFalse(verify_donor(p, pins)["complete"])

    def test_masked_training_targets_only_assistant_tokens(self):
        from train_vera_lora import assistant_only_tokens
        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt):
                pre = "USER>" + messages[0]["content"] + " ASSISTANT>"
                if add_generation_prompt:
                    return pre
                return pre + messages[1]["content"] + "<END>"
            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": [ord(ch) for ch in text]}
        messages = [{"role": "user", "content": "Hi!"},
                    {"role": "assistant", "content": "Vera."}]
        tokens, labels = assistant_only_tokens(Tokenizer(), messages, 200)
        self.assertEqual(tokens[-len("Vera.<END>"):],
                         [ord(ch) for ch in "Vera.<END>"])
        self.assertEqual(labels.count(-100), len("USER>Hi! ASSISTANT>"))
        self.assertEqual([x for x in labels if x != -100],
                         [ord(ch) for ch in "Vera.<END>"])

    def test_token_mask_fails_if_template_boundaries_differ(self):
        from train_vera_lora import assistant_only_tokens
        class Broken:
            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt):
                return "something" if add_generation_prompt else "different"
        with self.assertRaisesRegex(RuntimeError, "prefix"):
            assistant_only_tokens(Broken(),
                                  [{"role": "user", "content": "Hi!"},
                                   {"role": "assistant", "content": "Vera."}],
                                  200)

if __name__ == "__main__":
    unittest.main()
