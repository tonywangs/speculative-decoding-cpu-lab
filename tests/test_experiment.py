import json
from pathlib import Path
import tempfile
import unittest

from specdecode.data import generate_data
from specdecode.experiment import RunConfig, assert_agreement, load_experiment, run, summarize, verify


class ExperimentTests(unittest.TestCase):
    def test_disjoint_reproducible_data(self):
        first = generate_data(123, 128, 32)
        self.assertEqual(first, generate_data(123, 128, 32))
        train = {tuple(r["tokens"][:r["prompt_length"]]) for r in first["train"]}
        evaluation = {tuple(r["tokens"][:r["prompt_length"]]) for r in first["eval"]}
        self.assertFalse(train & evaluation)
        self.assertEqual(len(train), len(first["train"]))
        self.assertEqual(len(evaluation), len(first["eval"]))
        self.assertNotEqual(first, generate_data(124, 128, 32))

    def test_disagreement_is_fatal(self):
        with self.assertRaisesRegex(AssertionError, "disagreement"):
            assert_agreement([5, 6], [5, 7], "injected mismatch")

    def test_config_validation(self):
        for kwargs in ({"steps": 0}, {"repeats": 0}, {"warmups": 0},
                       {"draft_lengths": [1, 1]}, {"draft_lengths": []},
                       {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError):
                RunConfig(**kwargs)

    def test_complete_workflow_and_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "experiment"
            report = run(path, RunConfig(train_per_task=8, eval_per_task=2, steps=3,
                         batch_size=4, warmups=1, repeats=2, draft_lengths=(1, 4), max_new_tokens=8))
            self.assertTrue(report["agreement_passed"])
            self.assertEqual(report["timed_samples"], 6 * 5 * 2)
            verified = verify(path)
            self.assertEqual(verified["cases"], 6)
            self.assertTrue(verified["source_hashes_match"])
            self.assertTrue(verified["summary_recomputed"])
            raw = [json.loads(line) for line in (path / "timings.jsonl").read_text().splitlines()]
            pairs = {(r["id"], r["repeat"]) for r in raw}
            for pair in pairs:
                rows = [r for r in raw if (r["id"], r["repeat"]) == pair]
                self.assertEqual(len(rows), 5)
                self.assertEqual(sorted(r["order_in_pair"] for r in rows), list(range(5)))
                self.assertEqual(len({r["generated_tokens"] for r in rows}), 1)
                for row in rows:
                    self.assertGreaterEqual(row["elapsed_ns"], row["draft_ns"])
            checks = json.loads((path / "correctness.json").read_text())
            self.assertEqual(summarize(raw, checks), (report["summary"], report["accuracy"]))
            with self.assertRaises(ValueError):
                run(path, RunConfig(steps=1))
            (path / "dataset.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_experiment(path)


if __name__ == "__main__":
    unittest.main()
