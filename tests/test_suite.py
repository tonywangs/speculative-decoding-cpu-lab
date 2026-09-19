import copy
import json
from pathlib import Path
import tempfile
import unittest

from specdecode.experiment import RunConfig, sha256, validate_measurements, write_json
from specdecode.suite import SuiteConfig, run_suite, verify_suite


class SuiteTests(unittest.TestCase):
    def test_invalid_designs(self):
        for kwargs in ({"seeds": (17, 17, 29)}, {"widths": (64, 64)},
                       {"widths": (63, 96)}, {"repeats": 5}):
            with self.assertRaises(ValueError):
                SuiteConfig(**kwargs)

    def test_six_targets_shared_data_balanced_order_and_tampering(self):
        cfg = SuiteConfig(widths=(8, 16), steps=2, train_per_task=4, eval_per_task=1,
                          batch_size=2, warmups=1, repeats=5, draft_lengths=(1, 2), max_new_tokens=4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "suite"
            report = run_suite(path, cfg)
            verified = verify_suite(path)
            self.assertEqual(len(verified["configurations"]), 6)
            self.assertEqual(report["timed_samples"], 6 * 3 * 5 * 5)
            self.assertEqual(len({c["checkpoint_sha256"] for c in report["configurations"]}), 6)
            datasets = [(path / name / "dataset.json").read_bytes() for name, _ in cfg.configurations()]
            self.assertTrue(all(d == datasets[0] for d in datasets))
            child = path / "width8-seed17"
            raw = [json.loads(line) for line in (child / "timings.jsonl").read_text().splitlines()]
            checks = json.loads((child / "correctness.json").read_text())
            run_cfg = RunConfig(**json.loads((child / "config.json").read_text()))
            for case in {r["id"] for r in raw}:
                for method in {r["method"] for r in raw}:
                    self.assertEqual(sorted(r["order_in_pair"] for r in raw if r["id"] == case and r["method"] == method), list(range(5)))
            duplicate = copy.deepcopy(raw)
            duplicate[0] = duplicate[1]
            with self.assertRaisesRegex(AssertionError, "duplicate"):
                validate_measurements(duplicate, checks, run_cfg)
            damaged = copy.deepcopy(raw)
            damaged[0]["forward_calls"] += 1
            with self.assertRaisesRegex(AssertionError, "counter mismatch"):
                validate_measurements(damaged, checks, run_cfg)
            unbalanced = copy.deepcopy(raw)
            # Swap two positions within a pair: still paired, no longer balanced.
            unbalanced[0]["order_in_pair"], unbalanced[1]["order_in_pair"] = unbalanced[1]["order_in_pair"], unbalanced[0]["order_in_pair"]
            with self.assertRaisesRegex(AssertionError, "not balanced"):
                validate_measurements(unbalanced, checks, run_cfg)
            report["variation"][0]["max"] = 999
            write_json(path / "suite-report.json", report)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_suite(path)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["suite-report.json"] = sha256(path / "suite-report.json")
            write_json(path / "manifest.json", manifest)
            with self.assertRaisesRegex(AssertionError, "does not match"):
                verify_suite(path)


if __name__ == "__main__":
    unittest.main()
