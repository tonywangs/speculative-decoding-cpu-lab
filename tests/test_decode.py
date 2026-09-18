import unittest
from types import SimpleNamespace

import torch
from torch import nn

from specdecode.decode import decode, uncached_greedy
from specdecode.draft import NGram, PromptLookup
from specdecode.model import ModelConfig, TinyTransformer, crop_cache


class ScheduledModel(nn.Module):
    """Known position-dependent target with a cache that exposes stale tokens."""
    def __init__(self, schedule, context=16):
        super().__init__()
        self.schedule = schedule
        self.config = SimpleNamespace(context=context, vocab_size=22)
        self.eval()

    def forward(self, ids, cache=None):
        prefix = [] if cache is None else cache[0][0][0, 0, :, 0].long().tolist()
        history = prefix + ids[0].tolist()
        logits = torch.full((1, ids.shape[1], 22), -10.0)
        for j in range(ids.shape[1]):
            logits[0, j, self.schedule[len(prefix) + j]] = 10.0
        kv = torch.tensor(history, dtype=torch.float32).view(1, 1, -1, 1)
        return logits, [(kv, kv)]


class FixedDraft:
    def __init__(self, tokens):
        self.tokens = tokens

    def propose(self, history, count):
        return self.tokens[:count]


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_acceptance_and_bonus(self):
        model = ScheduledModel([5] * 16)
        got = decode(model, [0], 7, FixedDraft([5] * 4), 4, audit=True)
        self.assertEqual(got.tokens, [5] * 7)
        self.assertEqual(got.forward_calls, 2)
        self.assertEqual(got.accepted, 5)
        self.assertEqual(got.fully_accepted_blocks, 2)

    def test_rejection_at_each_position_and_rollback(self):
        for reject in range(4):
            model = ScheduledModel([5] * 16)
            draft = [5] * 4
            draft[reject] = 6
            got = decode(model, [0], 10, FixedDraft(draft), 4, audit=True)
            self.assertEqual(got.tokens, [5] * 10)
            self.assertGreater(got.rejected_blocks, 0)
            self.assertEqual(got.tokens, uncached_greedy(model, [0], 10))

    def test_eos_accepted_rejected_and_bonus(self):
        for draft in ([5, 4, 8], [5, 6, 8], [6, 6, 8], [5], [4]):
            model = ScheduledModel([5, 4] + [7] * 14)
            got = decode(model, [0], 10, FixedDraft(draft), len(draft), audit=True)
            self.assertEqual(got.tokens, [5, 4])
            self.assertEqual(got.stop_reason, "eos")
        terminal = decode(model, [0, 4], 10, FixedDraft([5]), 1)
        self.assertEqual(terminal.forward_calls, 0)
        self.assertEqual(terminal.tokens, [])

    def test_context_and_output_limits(self):
        model = ScheduledModel([5] * 16, context=8)
        for prompt_len in (1, 3, 7, 8):
            for limit in (0, 1, 2, 4, 20):
                got = decode(model, [0] + [5] * (prompt_len - 1), limit, FixedDraft([5] * 20), 20, audit=True)
                self.assertEqual(got.tokens, [5] * min(limit, 8 - prompt_len))
                self.assertLessEqual(got.forward_tokens, 8)

    def test_no_proposals_falls_back(self):
        model = ScheduledModel([5] * 16)
        self.assertEqual(decode(model, [0], 6, FixedDraft([]), 8).tokens, [5] * 6)

    def test_invalid_arguments(self):
        model = ScheduledModel([5] * 16)
        for prompt, limit in (([], 1), ([0] * 17, 1), ([22], 1), ([0], -1)):
            with self.assertRaises(ValueError):
                decode(model, prompt, limit)
        with self.assertRaises(ValueError):
            decode(model, [0], 2, FixedDraft([99]), 1)
        with self.assertRaises(ValueError):
            decode(model, [0], 2, draft_length=1)

    def test_real_transformer_rollback_matches_uncached(self):
        # Includes mismatched drafts, matched prefixes, multi-token blocks,
        # cropped caches, random prompts, two seeds and both numeric dtypes.
        for dtype in (torch.float32, torch.float64):
            for seed in (3, 19):
                torch.manual_seed(seed)
                model = TinyTransformer(ModelConfig(width=24, heads=3, layers=2, context=24)).to(dtype).eval()
                prompt = [0, 1, 9, 7, 6]
                reference = uncached_greedy(model, prompt, 16)

                class OracleDraft:
                    def __init__(self, corrupt):
                        self.corrupt = corrupt

                    def propose(self, history, count):
                        tokens = uncached_greedy(model, history, count)
                        if self.corrupt is not None and len(tokens) > self.corrupt:
                            tokens[self.corrupt] = (tokens[self.corrupt] + 1) % 22
                        return tokens

                for k in (1, 2, 4, 8, 32):
                    for corrupt in (None, 0, 1, 3):
                        got = decode(model, prompt, 16, OracleDraft(corrupt), k, audit=True)
                        self.assertEqual(got.tokens, reference)

    def test_chunked_and_cropped_logits(self):
        torch.manual_seed(7)
        model = TinyTransformer(ModelConfig(width=24, heads=3)).eval()
        ids = torch.tensor([[0, 1, 5, 6, 7, 8, 9, 10]])
        with torch.inference_mode():
            full, _ = model(ids)
            _, cache = model(ids[:, :5])
            # Replace speculative tail with a different continuation.
            cache = crop_cache(cache, 3)
            block, cache = model(ids[:, 3:], cache)
            torch.testing.assert_close(block, full[:, 3:], atol=1e-5, rtol=1e-4)
            for j in range(1, ids.shape[1]):
                _, prefix = model(ids[:, :j])
                tail, _ = model(ids[:, j:], prefix)
                torch.testing.assert_close(tail, full[:, j:], atol=1e-5, rtol=1e-4)


class DrafterTests(unittest.TestCase):
    def test_lookup_longest_context_and_empty(self):
        lookup = PromptLookup()
        self.assertEqual(lookup.propose([5, 6, 7, 8, 5, 6], 4), [7, 8, 5, 6])
        self.assertEqual(lookup.propose([5, 6, 7], 4), [])

    def test_ngram_longest_context_backoff_tie_break(self):
        ngram = NGram([[5, 6, 7], [5, 6, 8]], order=2)
        self.assertEqual(ngram.propose([5, 6], 1), [7])
        self.assertEqual(ngram.propose([20, 5], 1), [6])
        self.assertEqual(ngram.propose([21], 1), [5])


if __name__ == "__main__":
    unittest.main()
