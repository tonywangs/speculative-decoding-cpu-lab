import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys

import torch

from specdecode.decode import decode
from specdecode.experiment import configure
from specdecode.model import ModelConfig, TinyTransformer
from specdecode.realtext import TextConfig, make_data, run_text, sha256, verify_text, write_json
from specdecode.recurrent import ByteGRU, GRUDraft


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)

    def test_committed_cache_matches_fresh_after_rejection_and_history_change(self):
        torch.manual_seed(42)
        net = ByteGRU(vocab_size=22, width=8).eval()
        cached = GRUDraft(net)
        histories = [[0, 5, 6], [0, 5, 6, 7], [0, 5, 6, 7, 8, 9], [0, 10], [0, 10]]
        for history in histories:
            self.assertEqual(cached.propose(history, 4), GRUDraft(net).propose(history, 4))
        cached.reset()
        self.assertIsNone(cached.state)
        self.assertEqual(cached.history, [])

    def test_request_reset_prevents_warm_cache_and_outputs_match(self):
        torch.manual_seed(1)
        model = TinyTransformer(ModelConfig(width=16, heads=4, layers=1)).eval()
        draft = GRUDraft(ByteGRU(vocab_size=22, width=8).eval())
        resets = []
        reset = draft.reset
        def counted_reset():
            resets.append(True)
            reset()
        draft.reset = counted_reset
        prompt = [0, 5, 6, 7]
        reference = decode(model, prompt, 12).tokens
        for _ in range(2):
            self.assertEqual(decode(model, prompt, 12, draft, 4, audit=True).tokens, reference)
        self.assertEqual(len(resets), 2)

    def test_real_sources_splits_training_replay_and_tamper_rejection(self):
        corpus = Path(__file__).resolve().parents[1]/'results/realtext-gru/corpus'
        cfg = TextConfig(steps=2, draft_steps=2, train_per_task=3, eval_per_task=1,
                         validation_per_task=1, width=16, draft_width=8,
                         sequence_bytes=48, prompt_bytes=32, max_new_tokens=16,
                         warmups=1, repeats=4, draft_lengths=(2,))
        data = make_data(corpus, cfg)
        self.assertEqual(len(data['train']), 9)
        for split in ('train', 'validation', 'eval'):
            for row in data[split]:
                lo, hi = data['bounds'][row['task']][split]
                self.assertGreaterEqual(row['source_offset'], lo)
                self.assertLessEqual(row['source_offset']+cfg.sequence_bytes, hi)
                self.assertEqual(len(row['tokens']), cfg.sequence_bytes+1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            report = run_text(output, corpus, cfg)
            self.assertEqual(report['timed_samples'], 48)
            self.assertTrue(verify_text(output, corpus)['agreement_passed'])
            # Separate hash-seeded processes catch nondeterministic ordering of
            # summaries (e.g. family sort-key ties over a Python set).
            for hash_seed in ('1', '2'):
                subprocess.run([sys.executable, '-c',
                    'import sys; from specdecode.realtext import verify_text; verify_text(sys.argv[1], sys.argv[2])',
                    str(output), str(corpus)], check=True,
                    env={**os.environ, 'PYTHONHASHSEED': hash_seed})
            # Rehash corruption so semantic checks, not just hashes, must catch it.
            report['timed_samples'] += 1
            write_json(output/'report.json', report)
            manifest = json.loads((output/'manifest.json').read_text())
            manifest['report.json'] = sha256(output/'report.json')
            write_json(output/'manifest.json', manifest)
            with self.assertRaisesRegex(AssertionError, 'counts'):
                verify_text(output, corpus)


if __name__ == '__main__':
    unittest.main()
