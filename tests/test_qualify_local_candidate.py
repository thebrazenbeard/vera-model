"""No weights, network, subprocess inference, or real provider effects in tests."""
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "runtime" / "qualify_local_candidate.py"
spec = importlib.util.spec_from_file_location("qualify_local_candidate", SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidateProbeTests(unittest.TestCase):
    def test_false_and_missing_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-a-model.gguf"
            self.assertEqual(module.verify_artifact(path, 0, "x")["reason"], "MISSING_MODEL")
            path.write_bytes(b"a")
            self.assertEqual(module.verify_artifact(path, 2, "x")["reason"], "SIZE_MISMATCH")
            self.assertEqual(module.verify_artifact(path, 1, "0" * 64)["reason"], "HASH_MISMATCH")
            self.assertEqual(module.verify_artifact(path, 1, hashlib.sha256(b"a").hexdigest())["status"], "VERIFIED_LOCAL_BYTES")

    def test_reject_non_int_bounds_and_unregistered_candidates(self):
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", True)
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", 0, gpu_layers=100)
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", 0, predict=1000)
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", 0, reasoning="trust-me")
        with self.assertRaises(ValueError):
            module.probe("arbitrary", Path("missing"))

    def test_missing_file_never_launches_runner(self):
        with patch.object(module.subprocess, "run") as run:
            result = module.probe("gptoss-pruned4-q4", Path("missing.gguf"), execute=True)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["artifact"]["reason"], "MISSING_MODEL")
        run.assert_not_called()

    def test_pinned_bytes_do_not_auto_run(self):
        with patch.object(module, "verify_artifact", return_value={"status": "VERIFIED_LOCAL_BYTES", "sha256": "x"}), \
             patch.object(module.shutil, "which", return_value="llama-cli"), \
             patch.object(module.subprocess, "run") as run:
            result = module.probe("gptoss-pruned4-q4", Path("any"), execute=False)
        self.assertEqual(result["status"], "READY_FOR_OPT_IN_LOCAL_PROBE")
        self.assertFalse(result["provider_admitted"])
        self.assertFalse(result["behaviorally_qualified"])
        self.assertFalse(result["memory_consumed"])
        run.assert_not_called()

    def test_success_does_not_promote_identity_or_authority(self):
        class Done:
            returncode = 0
            stdout = "hello"
            stderr = ""
        with patch.object(module, "verify_artifact", return_value={"status": "VERIFIED_LOCAL_BYTES", "sha256": "x"}), \
             patch.object(module.shutil, "which", return_value="llama-cli"), \
             patch.object(module.subprocess, "run", return_value=Done()) as run:
            result = module.probe("qwen35-4b-q4-control", Path("any"), execute=True)
        self.assertEqual(result["status"], "RUNNER_EXIT_ZERO_UNQUALIFIED")
        self.assertFalse(result["provider_admitted"])
        self.assertFalse(result["effect_authority_granted"])
        self.assertFalse(result["behaviorally_qualified"])
        self.assertIn("--reasoning", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")

    def test_timeout_stays_unqualified(self):
        import subprocess
        with patch.object(module, "verify_artifact", return_value={"status": "VERIFIED_LOCAL_BYTES"}), \
             patch.object(module.shutil, "which", return_value="llama-cli"), \
             patch.object(module.subprocess, "run", side_effect=subprocess.TimeoutExpired("llama-cli", 120)):
            result = module.probe("gptoss-pruned4-q4", Path("any"), execute=True)
        self.assertEqual(result["status"], "INFERENCE_TIMEOUT")
        self.assertFalse(result["behaviorally_qualified"])


if __name__ == "__main__":
    unittest.main()
