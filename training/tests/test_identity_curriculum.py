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
        import hashlib
        recipe = copy.deepcopy(load_json(ROOT / "training/qlora-recipe-v1.json"))
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
            recipe["data"]["files_sha256"] = {
                name: hashlib.sha256((dest / name).read_bytes()).hexdigest()
                for name in ("vera_sft.jsonl", "vera_preferences.jsonl",
                             "vera_heldout.jsonl")}
            recipe["data"]["manifest_sha256"] = hashlib.sha256(
                (dest / "manifest.json").read_bytes()).hexdigest()
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
            bos_token = "<s>"
            eos_token = "</s>"
            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": [ord(ch) for ch in text]}
        messages = [{"role": "user", "content": "Hi!"},
                    {"role": "assistant", "content": "Vera."}]
        tokens, labels = assistant_only_tokens(Tokenizer(), messages, 200)
        prefix = "<s>[INST] Hi! [/INST]"
        self.assertEqual(tokens[:len(prefix)], [ord(ch) for ch in prefix])
        self.assertEqual(labels[:len(prefix)], [-100] * len(prefix))
        self.assertEqual([x for x in labels if x != -100],
                         [ord(ch) for ch in " Vera.</s>"])

    def test_token_mask_fails_if_donor_special_tokens_differ(self):
        from train_vera_lora import assistant_only_tokens
        class Broken:
            bos_token = "<wrong>"
            eos_token = "</s>"
        with self.assertRaisesRegex(RuntimeError, "BOS/EOS"):
            assistant_only_tokens(Broken(),
                                  [{"role": "user", "content": "Hi!"},
                                   {"role": "assistant", "content": "Vera."}],
                                  200)

    def test_full_tokenizer_preflight_covers_exact_training_window(self):
        from preflight_tokenizer import validate_sft_tokenization
        class TinyTokenizer:
            bos_token = "<s>"
            eos_token = "</s>"
            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": [ord(char) for char in text]}
        rows = [{"id": f"row{i:03d}", "messages": [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "Vera."}]}
            for i in range(68)]
        evidence = validate_sft_tokenization(TinyTokenizer(), rows, 512)
        self.assertEqual(evidence["rows_checked"], 68)
        self.assertTrue(evidence["all_assistant_only_masking_valid"])
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            validate_sft_tokenization(TinyTokenizer(), rows, 10)
        with self.assertRaisesRegex(ValueError, "exact pilot"):
            validate_sft_tokenization(TinyTokenizer(), rows[:-1], 512)

    def test_tokenizer_assets_are_pinned_and_reject_tamper(self):
        import hashlib
        from train_vera_lora import verify_tokenizer_assets
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = {
                "config.json": '{"model_type":"mistral"}',
                "tokenizer.json": '{"version":"1.0"}',
                "special_tokens_map.json":
                    '{"bos_token":"<s>","eos_token":"</s>"}',
                "tokenizer_config.json": json.dumps({
                    "tokenizer_class": "LlamaTokenizerFast",
                    "bos_token": "<s>", "eos_token": "</s>",
                    "additional_special_tokens": ["[INST]", "[/INST]"]
                }),
            }
            for name, data in fields.items():
                (root / name).write_text(data, encoding="utf8")
            pins = {name: {
                "bytes": (root / name).stat().st_size,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                for name in fields}
            self.assertEqual(verify_tokenizer_assets(root, pins), 4)
            (root / "tokenizer.json").write_text("tampered", encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "tokenizer asset"):
                verify_tokenizer_assets(root, pins)

    def test_donor_without_chat_template_still_trains_assistant_only(self):
        from train_vera_lora import assistant_only_tokens
        class MinimalOfficialTokenizer:
            bos_token = "<s>"
            eos_token = "</s>"
            chat_template = None
            def __call__(self, value, *, add_special_tokens):
                assert add_special_tokens is False
                return {"input_ids": [ord(ch) for ch in value]}
        tokenizer = MinimalOfficialTokenizer()
        tokens, labels = assistant_only_tokens(
            tokenizer,
            [{"role": "user", "content": "Who are you?"},
             {"role": "assistant", "content": "I'm Vera."}], 300)
        rendered = "".join(chr(ch) for ch in tokens)
        self.assertEqual(rendered, "<s>[INST] Who are you? [/INST] I'm Vera.</s>")
        prefix = "<s>[INST] Who are you? [/INST]"
        self.assertEqual(labels[:len(prefix)], [-100] * len(prefix))
        self.assertEqual([chr(ch) for ch in labels if ch != -100],
                         list(" I'm Vera.</s>"))

    def test_self_consistent_corpus_tamper_rejected_by_recipe(self):
        import hashlib
        from train_vera_lora import verify_corpus
        train = load_jsonl(ROOT / "training/source/vera_voice_train_seed_v1.jsonl")
        heldout = load_jsonl(ROOT / "evaluation/cases/vera_voice_heldout_v1.jsonl")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "corpus"
            metadata = {"donor_model": "test/base",
                        "donor_revision": "f658f1724210316e17463d06b752ac243e5c85b0",
                        "identity_provenance": {"source_commit": "a" * 40},
                        "seed_sha256": "b" * 64, "heldout_source_sha256": "c" * 64,
                        "checks": validate(train, heldout)}
            export(train, heldout, output_dir=directory, evidence=metadata)
            recipe = copy.deepcopy(load_json(
                ROOT / "training/qlora-recipe-v1.json"))
            names = ("vera_sft.jsonl", "vera_preferences.jsonl", "vera_heldout.jsonl")
            recipe["data"]["files_sha256"] = {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in names}
            recipe["data"]["manifest_sha256"] = hashlib.sha256(
                (directory / "manifest.json").read_bytes()).hexdigest()
            verify_corpus(directory, recipe)
            f = directory / "vera_sft.jsonl"
            edited = f.read_text(encoding="utf8").replace(
                "VERA_VOICE_TRAIN_001", "VERA_VOICE_TRAIN_777", 1)
            self.assertTrue(edited != f.read_text(encoding="utf8"))
            f.write_text(edited, encoding="utf8")
            manifest = load_json(directory / "manifest.json")
            manifest["files_sha256"]["vera_sft.jsonl"] = hashlib.sha256(
                f.read_bytes()).hexdigest()
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf8")
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                verify_corpus(directory, recipe)


if __name__ == "__main__":
    unittest.main()
