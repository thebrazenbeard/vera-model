"""Fail-closed cloud budget and persistence contract unit tests.

These exercise source only; they DO NOT establish successful GPU training.
"""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from cloud_job import validate_budget, fresh_output, write_evidence, canonical_sha, mounted_output_verified


class CloudBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((ROOT/"training/hf_spend_and_storage_gate_v1.json").read_text())

    def test_blocked_without_writable_storage(self):
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            validate_budget(self.plan)

    def test_authorized_ready_budget_keeps_buffer(self):
        plan=copy.deepcopy(self.plan)
        plan["status"]="READY_FOR_LAUNCH"
        self.assertAlmostEqual(validate_budget(plan),4.2,places=5)

    def test_overspend_refused(self):
        plan=copy.deepcopy(self.plan)
        plan["status"]="READY_FOR_LAUNCH"
        plan["timeout_seconds"]=12000
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            validate_budget(plan)
        plan["timeout_seconds"]=8400
        plan["expected_gpu_price_usd_per_hour"]=2.5
        with self.assertRaisesRegex(RuntimeError, "budget"):
            validate_budget(plan)

    def test_retries_and_gpu_flavor_change_refused(self):
        plan=copy.deepcopy(self.plan)
        plan["status"]="READY_FOR_LAUNCH"
        plan["attempts"]=2
        with self.assertRaisesRegex(RuntimeError, "flavor or retries"):
            validate_budget(plan)
        plan["attempts"]=1
        plan["flavor"]="l40sx4"
        with self.assertRaisesRegex(RuntimeError, "flavor or retries"):
            validate_budget(plan)

    def test_ephemeral_outputs_are_not_accepted_as_bucket(self):
        mountinfo=("32 22 0:30 / / rw,relatime - overlay overlay rw\n"
                   "33 22 0:31 / /outputs rw,relatime - fuse.fuse-overlayfs bucket rw\n")
        self.assertTrue(mounted_output_verified(
            mountinfo=mountinfo,target=Path("/outputs")))
        self.assertFalse(mounted_output_verified(
            mountinfo=mountinfo.splitlines()[0],target=Path("/outputs")))
        self.assertFalse(mounted_output_verified(
            mountinfo=mountinfo,target=Path("/tmp/outputs")))

    def test_output_not_mounted_cannot_make_false_run(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "private bucket"):
                fresh_output(Path(d), "vera24b-20261013-demo001")
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_receipt_readback_bound_and_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)
            base={"schema":"V", "status":"TEST_ONLY", "steps":0}
            write_evidence(work,"JOB_ATTEMPT.json",base)
            content=json.loads((work/"JOB_ATTEMPT.json").read_text())
            checksum=content.pop("receipt_sha256")
            self.assertEqual(checksum,canonical_sha(content))
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                write_evidence(work,"JOB_ATTEMPT.json",base)

    def test_reject_injection_filename(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "unapproved"):
                write_evidence(Path(d),"../BAD.json",{"status":"test"})


if __name__=="__main__":
    unittest.main()
