"""Focused controller tests. Small models test mechanics, never research claims."""
import json
from pathlib import Path
import signal
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from specdecode.decode import decode, uncached_greedy
from specdecode.draft import NGram, PromptLookup
from specdecode.experiment import configure
from specdecode.model import ModelConfig, TinyTransformer
from specdecode.policy import AcceptancePolicy, configurations, select_second
from specdecode.recurrent import ByteGRU, GRUDraft
from specdecode.research import (DEFAULT_LIMITS, Journal, LimitStop, locked, population,
                                 work_items)

ROOT = Path(__file__).resolve().parents[1]


def small_manifest():
    return {'calibration_units': [{'seed': 17, 'case_id': 'validation-test-0000000'}]}


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)

    def test_promotion_demotion_reset_and_one_shot_fallback(self):
        policy = AcceptancePolicy()
        for n in (2, 4):
            policy.observe(2, 2, False)
            self.assertEqual(policy.length, n)
        policy.observe(4, 1, True)
        self.assertEqual(policy.length, 2)
        policy.reset()
        self.assertEqual((policy.length, policy.trace), (2, []))
        fallback = AcceptancePolicy(fallback=True)
        for _ in range(3):
            fallback.observe(2, 0, True)
            self.assertEqual(fallback.length, 2)
        fallback.observe(2, 0, True)
        self.assertEqual(fallback.length, 0)
        boundary = AcceptancePolicy(fallback=True)
        for _ in range(4):
            boundary.observe(2, 1, True)
        self.assertEqual(boundary.length, 2)  # equality at 0.5 never falls back
        strict = AcceptancePolicy(promotion=4)
        for i in range(4):
            strict.observe(2, 2, False)
            self.assertEqual(strict.length, 4 if i == 3 else 2)

    def test_adaptive_and_fallback_match_uncached_on_real_transformer(self):
        for seed in (17, 29):
            torch.manual_seed(seed)
            model = TinyTransformer(ModelConfig(width=16, heads=4, layers=1, context=64)).eval()
            draft = ByteGRU(vocab_size=22, width=8).eval()
            prompt = [0, 5, 6, 7, 5, 6, 7]
            oracle = uncached_greedy(model, prompt, 32)
            for drafter in (PromptLookup(), NGram([prompt]), GRUDraft(draft)):
                for policy in (AcceptancePolicy(), AcceptancePolicy(4), AcceptancePolicy(fallback=True)):
                    for _ in range(2):
                        actual = decode(model, prompt, 32, drafter, 2, audit=True, policy=policy)
                        self.assertEqual(actual.tokens, oracle)
                        self.assertEqual(policy.blocks, len(policy.trace))
                        self.assertLessEqual(policy.accepted, policy.proposed)

    def test_result_dependent_selection_and_complete_population_gate(self):
        expected = [(17, 'a'), (29, 'a')]
        rows = [{'seed': seed, 'case_id': case, 'phase': 'timing', 'split': 'validation',
                 'method': method, 'repeat': repeat, 'agreement': True,
                 'elapsed_ns': 100 if method == 'greedy' else 110}
                for seed, case in expected for method in (v['name'] for v in configurations()) for repeat in range(7)]
        decision = select_second(rows, expected)
        self.assertEqual(decision['branch'], 'fallback')
        self.assertEqual(len(decision['evidence_keys']), 98)
        for row in rows:
            if row['method'] == 'gru-adaptive':
                row['elapsed_ns'] = 100
        self.assertEqual(select_second(rows, expected)['branch'], 'strict')
        next(r for r in rows if r['method'] == 'gru-adaptive')['elapsed_ns'] = 120
        self.assertEqual(select_second(rows, expected)['branch'], 'strict')
        with self.assertRaisesRegex(ValueError, 'complete validation'):
            select_second(rows[:-1], expected)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            select_second(rows + rows[:1], expected)
        rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'validation'):
            select_second(rows, expected)


class PopulationTests(unittest.TestCase):
    def test_full_real_population_exclusions_and_balanced_queue(self):
        manifest = population(ROOT / 'results/realtext-gru')
        counts = {(split, task): sum(r['split'] == split and r['task'] == task for r in manifest['windows'])
                  for split in ('validation', 'test') for task in ('dialogue', 'prose', 'python')}
        self.assertEqual(counts, {('validation', 'dialogue'): 697, ('validation', 'prose'): 96,
                                 ('validation', 'python'): 38, ('test', 'dialogue'): 697,
                                 ('test', 'prose'): 96, ('test', 'python'): 39})
        self.assertEqual(len(manifest['excluded']), 1)
        self.assertEqual(manifest['excluded'][0]['reason'], 'training_region_match')
        items = work_items(manifest)
        self.assertEqual(len(items), 6 * 73)
        self.assertEqual(len({r['work_id'] for r in items}), len(items))
        for unit in manifest['calibration_units']:
            for method in (v['name'] for v in configurations()):
                positions = [r['position'] for r in items if r['seed'] == unit['seed'] and
                             r['case_id'] == unit['case_id'] and r['method'] == method and r['phase'] == 'timing']
                self.assertEqual(sorted(positions), list(range(7)))
            first = next(r for r in manifest['windows'] if r['case_id'] == unit['case_id'])
            offsets = [r['source_offset'] for r in manifest['windows'] if r['task'] == first['task'] and r['split'] == 'validation']
            self.assertEqual(first['source_offset'], min(offsets))


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.journal = Journal(self.output)
        self.journal.initialize(small_manifest(), DEFAULT_LIMITS)
        self.deadline = time.time() + 60

    def tearDown(self):
        self.journal.close()
        self.tmp.cleanup()

    def test_atomic_completion_no_duplicate_and_append_only_measurements(self):
        item = self.journal.pending()[0]
        meter = self.journal.begin(item['work_id'], self.deadline)
        meter.count('target')
        self.journal.finish(meter, item)
        self.assertEqual(self.journal.usage()['calls'], 1)
        self.assertEqual(len(self.journal.pending()), 72)
        with self.assertRaises(sqlite3.IntegrityError):
            self.journal.finish(meter, item)
        for table in ('work', 'metadata', 'events', 'results'):
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'append-only'):
                self.journal.db.execute(f'DELETE FROM {table}')
        self.journal.close()
        self.journal = Journal(self.output)
        self.journal.recover()
        self.assertEqual(self.journal.results(), [item])
        self.assertEqual(self.journal.usage()['calls'], 1)

    def test_crash_recovery_charges_full_reservation_and_retains_pending(self):
        item = self.journal.pending()[0]
        meter = self.journal.begin(item['work_id'], self.deadline, max_seconds=5, max_calls=100)
        meter.count('target')
        self.journal.close()  # simulate process death before commit
        self.journal = Journal(self.output)
        self.journal.recover()
        self.assertEqual(self.journal.usage()['lost_calls'], 100)
        self.assertEqual(self.journal.usage()['seconds'], 5)
        self.assertEqual(self.journal.pending()[0], item)
        self.journal.recover()  # idempotent
        self.assertEqual(self.journal.usage()['calls'], 100)
        new = self.journal.begin(item['work_id'], self.deadline)
        new.count('target')
        self.journal.finish(new, item)
        self.assertEqual(self.journal.usage()['calls'], 101)
        self.assertEqual(len(self.journal.results()), 1)

    def test_call_limit_counts_oracle_audit_and_gru_and_survives_resume(self):
        meter = self.journal.begin(None, self.deadline, max_calls=3)
        with meter:
            model = TinyTransformer(ModelConfig(width=16, heads=4, layers=1)).eval()
            draft = ByteGRU(vocab_size=22, width=8).eval()
            meter.attach(model, draft)
            with torch.inference_mode():
                model(torch.tensor([[0, 5]]))
                draft(torch.tensor([[0, 5]]))
                model(torch.tensor([[0, 5]]))
                with self.assertRaisesRegex(LimitStop, 'model-call'):
                    draft(torch.tensor([[0, 5]]))
        self.journal.finish(meter, error='limit')
        self.assertEqual((meter.target_calls, meter.draft_calls), (2, 1))
        self.assertEqual(self.journal.usage()['calls'], 3)
        self.journal.close()
        self.journal = Journal(self.output)
        self.assertEqual(self.journal.usage()['calls'], 3)

    def test_cumulative_limit_never_resets_and_deadline_stops_before_work(self):
        with patch.object(self.journal, 'usage', return_value={'seconds': 600, 'calls': 1}):
            with self.assertRaisesRegex(LimitStop, 'time limit'):
                self.journal.begin(None, self.deadline)
        with patch.object(self.journal, 'usage', return_value={'seconds': 1, 'calls': 150000}):
            with self.assertRaisesRegex(LimitStop, 'model-call'):
                self.journal.begin(None, self.deadline)
        with self.assertRaisesRegex(LimitStop, 'deadline'):
            self.journal.begin(None, time.time() - 1)
        self.assertEqual(self.journal.usage()['calls'], 0)

    def test_atomic_rollback_leaves_retryable_reservation(self):
        item = self.journal.pending()[0]
        meter = self.journal.begin(item['work_id'], self.deadline)
        with patch.object(self.journal, 'event', side_effect=RuntimeError('crash before commit')):
            with self.assertRaisesRegex(RuntimeError, 'crash'):
                self.journal.finish(meter, item)
        self.assertEqual(self.journal.results(), [])
        self.journal.recover()
        self.assertEqual(self.journal.usage()['calls'], 512)
        self.assertEqual(self.journal.pending()[0], item)

    def test_signal_after_commit_does_not_rewrite_success_as_failure(self):
        item = self.journal.pending()[0]
        meter = self.journal.begin(item['work_id'], self.deadline)
        meter.count('target')
        self.journal.finish(meter, item)
        usage = self.journal.usage()
        self.journal.fail(meter, LimitStop('signal after commit'))
        self.assertEqual(self.journal.usage(), usage)
        self.assertEqual(self.journal.results(), [item])
        self.assertEqual(self.journal.db.execute('SELECT status FROM attempts WHERE id=?',
                                               (meter.attempt,)).fetchone()[0], 'complete')
        self.assertEqual(self.journal.db.execute("SELECT count(*) FROM events WHERE kind='failed'").fetchone()[0], 0)

    def test_wall_timer_interrupts_and_is_removed(self):
        self.journal.last_boundary = time.monotonic()
        meter = self.journal.begin(None, self.deadline, max_seconds=.05)
        with self.assertRaisesRegex(LimitStop, 'wall-clock'):
            with meter:
                time.sleep(.1)
        self.journal.finish(meter, error='limit')
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_concurrent_owner_is_rejected(self):
        with locked(self.output):
            with self.assertRaisesRegex(ValueError, 'another controller'):
                with locked(self.output):
                    pass

class ControllerIntegrationTests(unittest.TestCase):
    """Exercise real transactions and signals with a cheap deterministic target."""
    @classmethod
    def setUpClass(cls):
        configure(1)

    def setUp(self):
        from types import SimpleNamespace
        from torch import nn

        class ConstantTarget(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(context=161, vocab_size=261)
                self.eval()

            def forward(self, ids, cache=None):
                length = ids.shape[1] + (0 if cache is None else cache[0][0].shape[2])
                logits = torch.zeros((1, ids.shape[1], 261))
                logits[..., 5] = 10
                kv = torch.zeros((1, 1, length, 1))
                return logits, [(kv, kv)]

        import hashlib
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.inputs, self.output = root / 'inputs', root / 'output'
        (self.inputs / 'corpus').mkdir(parents=True)
        self.output.mkdir()
        chunk = bytes(range(32, 112)) * 2
        (self.inputs / 'corpus/dialogue.txt').write_bytes(chunk)
        self.manifest = small_manifest()
        self.manifest['windows'] = [{'case_id': 'validation-test-0000000', 'task': 'dialogue',
                                    'split': 'validation', 'source_offset': 0,
                                    'window_sha256': hashlib.sha256(chunk).hexdigest()}]
        self.journal = Journal(self.output)
        self.journal.initialize(self.manifest, DEFAULT_LIMITS)
        self.target = ConstantTarget()
        self.draft = ByteGRU(width=8).eval()
        self.data = {'train': [{'tokens': [0, 5, 5, 5, 5, 5]}]}

    def tearDown(self):
        self.journal.close()
        self.tmp.cleanup()

    def run_controller(self, **kwargs):
        from specdecode.research import calibrate
        with patch('specdecode.research.population', return_value=self.manifest), \
             patch('specdecode.research.load_text', return_value=(self.target, self.draft, self.data, None)):
            return calibrate(self.output, self.inputs, time.time() + 60, **kwargs)

    def test_pause_mid_timing_resume_rewarms_without_duplicate_records(self):
        first = self.run_controller(stop_after_records=25)
        self.assertEqual(first['completed_records'], 25)
        before = self.journal.results()
        second = self.run_controller()
        self.assertTrue(second['calibration_complete'])
        self.assertEqual(second['completed_records'], 73)
        self.assertEqual(self.journal.results()[:25], before)
        self.assertGreater(second['usage']['calls'], first['usage']['calls'])
        self.assertGreater(second['usage']['seconds'], first['usage']['seconds'])
        self.assertEqual(self.journal.db.execute("SELECT count(*) FROM events WHERE kind='resume_warmup'").fetchone()[0], 14)
        usage = second['usage']
        third = self.run_controller()  # completion is idempotent; no new inference
        self.assertEqual(third['usage'], usage)
        self.assertEqual(third['completed_full_experiments'], 0)

    def test_sigterm_records_failure_then_resume_and_limits_do_not_reset(self):
        from specdecode import research
        execute_original = research.execute
        calls = 0

        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                signal.raise_signal(signal.SIGTERM)
            return execute_original(*args, **kwargs)

        with patch('specdecode.research.execute', side_effect=interrupt):
            interrupted = self.run_controller()
        self.assertEqual(interrupted['completed_records'], 3)
        self.assertEqual(self.journal.db.execute("SELECT count(*) FROM attempts WHERE status='failed'").fetchone()[0], 1)
        resumed = self.run_controller()
        self.assertTrue(resumed['calibration_complete'])
        self.assertGreater(resumed['usage']['seconds'], interrupted['usage']['seconds'])
        self.assertGreater(resumed['usage']['calls'], interrupted['usage']['calls'])

    def test_real_call_limit_stop_at_exact_boundary_leaves_work_pending(self):
        # Separate immutable run with only ten aggregate forward calls allowed.
        self.journal.close()
        self.output = Path(self.tmp.name) / 'limited'
        self.output.mkdir()
        self.journal = Journal(self.output)
        self.journal.initialize(self.manifest, {**DEFAULT_LIMITS, 'calls': 10})
        first = self.run_controller()
        self.assertEqual(first['usage']['calls'], 10)
        self.assertEqual(first['completed_records'], 0)
        self.assertEqual(first['pending_calibration_records'], 73)
        second = self.run_controller()
        self.assertEqual(second['usage'], first['usage'])
        self.assertEqual(second['pending_calibration_records'], 73)


    def test_memory_ceiling_stops_allocation_and_restores_process_limit(self):
        import resource
        vm_kib = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                          if line.startswith('VmSize:')))
        ceiling = vm_kib * 1024 + 64 * 1024**2
        self.journal.close()
        self.output = Path(self.tmp.name) / 'memory-limited'
        self.output.mkdir()
        self.journal = Journal(self.output)
        self.journal.initialize(self.manifest, {**DEFAULT_LIMITS, 'memory_bytes': ceiling})
        prior = resource.getrlimit(resource.RLIMIT_AS)

        def allocate(*_):
            self.assertEqual(resource.getrlimit(resource.RLIMIT_AS)[0], ceiling)
            bytearray(ceiling)  # requires more virtual memory than the remaining allowance
            self.fail('allocation unexpectedly escaped the ceiling')

        with patch('specdecode.research.execute', side_effect=allocate):
            stopped = self.run_controller()
        self.assertEqual(stopped['completed_records'], 0)
        self.assertEqual(stopped['pending_calibration_records'], 73)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), prior)
        failures = [json.loads(r[0]) for r in self.journal.db.execute("SELECT data FROM events WHERE kind='failed'")]
        self.assertTrue(any('MemoryError' in r['error'] for r in failures))


if __name__ == '__main__':
    unittest.main()
