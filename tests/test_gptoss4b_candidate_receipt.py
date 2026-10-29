"""Require exact Lappy custody evidence without promoting model capability."""
from __future__ import annotations

import json
import unittest
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "candidates" / "VERA-GPT-OSS4B_Q4_K_M" / "manifest.json"


class CandidateReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_historical_lappy_download_receipt_matches_pinned_file(self):
        model = self.model
        inf = model["inference"]
        receipt = inf["local_acquisition_receipt"]
        self.assertEqual(model["display_name"], "VERA-GPT-OSS4B Q4_K_M")
        self.assertTrue(inf["local_download_verified"])
        self.assertEqual(receipt["host"], "LAPPY")
        self.assertEqual(receipt["evidence_class"], "REAL_SYSTEM_PROBE")
        self.assertRegex(receipt["verified_at_utc"], r"^2026-10-29T\d{2}:\d{2}:\d{2}Z$")
        self.assertEqual(receipt["bytes"], inf["expected_bytes"])
        self.assertEqual(receipt["sha256"], inf["sha256"])
        self.assertRegex(receipt["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(PureWindowsPath(receipt["path"]).name.lower(), inf["filename"].lower())
        self.assertEqual(inf["expected_bytes"], 3233019552)
        self.assertEqual(inf["sha256"], "38b0279e4294beb54f97ac23051674174b34b2acd6c5e6532860c72298789586")

    def test_download_does_not_promote_training_or_runtime(self):
        model = self.model
        self.assertEqual(model["status"], "LOCAL_GGUF_VERIFIED_NOT_TRAINED_NOT_QUALIFIED")
        training = model["training"]
        for field in ("model_weight_updates_observed", "adapter_reload_verified", "heldout_improvement_verified"):
            self.assertIs(training[field], False)
        self.assertIs(model["inference"]["performance_verified"], False)
        self.assertEqual(model["integration"]["portal_admission"], "HOLD")
        self.assertTrue(model["integration"]["no_automatic_provider_selection"])


if __name__ == "__main__":
    unittest.main()
