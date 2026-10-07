import copy
import json
from pathlib import Path
import tempfile
import unittest

from specdecode.data import generate_data
from specdecode.experiment import sha256, write_json, validate_measurements
from specdecode.replication import (ReplicationConfig, audit_splits, case_estimates, freeze,
                                    read_gzip, run_replication, timing_cases, verify_replication)


class ReplicationTests(unittest.TestCase):
    def test_protocol_constraints_and_splits(self):
        for kwargs in ({"data_seeds": (1, 2, 5)}, {"training_seeds": (17, 17, 43)},
                       {"eval_per_task": 127}, {"timed_per_task": 15}, {"repeats": 5}):
            with self.assertRaises(ValueError):
                ReplicationConfig(**kwargs)
        cfg = ReplicationConfig()
        self.assertEqual(len(cfg.configurations()), 9)
        data = generate_data(3101, 8, 128)
        audit_splits(data)
        self.assertEqual(len(timing_cases(data, cfg)), 48)
        damaged = copy.deepcopy(data)
        damaged['eval'][0] = damaged['train'][0]
        with self.assertRaisesRegex(AssertionError, 'overlapping'):
            audit_splits(damaged)

    def test_source_drift_and_protocol_edit_are_rejected_before_training(self):
        cfg = ReplicationConfig(phase='development', data_seeds=(11003,), training_seeds=(7,),
                                train_per_task=2, eval_per_task=2, timed_per_task=1,
                                steps=1, bootstrap_draws=100)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = root / 'protocol.json'
            frozen = freeze(protocol, cfg)
            frozen['source_sha256']['replication.py'] = '0' * 64
            write_json(protocol, frozen)
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                run_replication(root / 'output', protocol)
            self.assertFalse((root / 'output').exists())
            frozen['datasets']['11003']['timing_ids'].pop()
            write_json(protocol, frozen)
            with self.assertRaisesRegex(ValueError, 'frozen generation'):
                run_replication(root / 'output', protocol)

    def test_case_bootstrap_does_not_count_repeats_as_cases(self):
        cfg = ReplicationConfig(bootstrap_draws=100)
        raw = []
        for task in ('repetitive', 'structured', 'unpredictable'):
            for case, ratio in enumerate((.5, 1, 4)):
                for repeat in range(6):
                    for method, time in [('greedy', 100), ('prompt-k2', 100 / ratio)]:
                        raw.append(dict(id=f'{task}-{case}', task=task, repeat=repeat,
                                        method=method, elapsed_ns=time))
        result = case_estimates(raw, cfg)
        for row in result:
            self.assertEqual(len(row['case_speedups']), 3)
            self.assertEqual(row['speedup'], 1)
            if row['method'] == 'greedy':
                self.assertEqual(row['case_bootstrap_95'], [1, 1])

    def test_nine_targets_replay_and_semantic_tampering(self):
        cfg = ReplicationConfig(phase='development', data_seeds=(11003, 14009, 17011),
                                training_seeds=(7, 11, 13), width=8, steps=1, train_per_task=2,
                                eval_per_task=2, timed_per_task=1, batch_size=2, warmups=1,
                                max_new_tokens=2, draft_lengths=(1, 2), bootstrap_draws=100)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = root / 'protocol.json'
            freeze(protocol, cfg)
            with self.assertRaises(ValueError):
                freeze(protocol, cfg)
            path = root / 'replication'
            report = run_replication(path, protocol)
            verified = verify_replication(path)
            self.assertEqual(verified['configurations'], 9)
            self.assertEqual(verified['evaluation_cases'], 54)
            self.assertEqual(verified['timed_samples'], 972)
            self.assertTrue(verified['source_hashes_match'])
            child = path / cfg.configurations()[0][0]
            raw = read_gzip(child / 'timings-k1.jsonl.gz')
            checks = [r for task in ('repetitive', 'structured', 'unpredictable')
                      for r in read_gzip(child / f'correctness-{task}.jsonl.gz') if r['id'].endswith('0000')]
            block_cfg = cfg.run_config(11003, 7, 1)
            damaged = copy.deepcopy(raw)
            damaged[0] = damaged[1]
            with self.assertRaisesRegex(AssertionError, 'duplicate'):
                validate_measurements(damaged, checks, block_cfg)
            damaged = copy.deepcopy(raw)
            damaged[0]['order_in_pair'], damaged[1]['order_in_pair'] = damaged[1]['order_in_pair'], damaged[0]['order_in_pair']
            with self.assertRaisesRegex(AssertionError, 'not balanced'):
                validate_measurements(damaged, checks, block_cfg)
            report['variation'][0]['dataset_min'] = 999
            write_json(path / 'replication-report.json', report)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_replication(path)
            manifest = json.loads((path / 'manifest.json').read_text())
            manifest['replication-report.json'] = sha256(path / 'replication-report.json')
            write_json(path / 'manifest.json', manifest)
            with self.assertRaisesRegex(AssertionError, 'aggregate report'):
                verify_replication(path)


if __name__ == '__main__':
    unittest.main()
