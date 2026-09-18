"""Exact greedy verification with rollback and an independent uncached oracle."""
from dataclasses import dataclass, field
from time import perf_counter_ns

import torch

from .data import EOS
from .model import cache_length, crop_cache


@dataclass
class DecodeResult:
    tokens: list[int] = field(default_factory=list)
    forward_calls: int = 0
    forward_tokens: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected_blocks: int = 0
    fully_accepted_blocks: int = 0
    draft_ns: int = 0
    stop_reason: str = ""


def validate(model, prompt, max_new_tokens):
    if not prompt or len(prompt) > model.config.context:
        raise ValueError("prompt must be nonempty and fit context")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if any(t < 0 or t >= model.config.vocab_size for t in prompt):
        raise ValueError("token out of vocabulary")
    if model.training:
        raise ValueError("decoding requires model.eval()")


@torch.inference_mode()
def decode(model, prompt, max_new_tokens, drafter=None, draft_length=0, audit=False):
    validate(model, prompt, max_new_tokens)
    if draft_length < 0 or (drafter is None and draft_length):
        raise ValueError("positive draft length requires a drafter")
    history = list(prompt)
    result = DecodeResult()
    budget = min(max_new_tokens, model.config.context - len(history))
    if not budget or history[-1] == EOS:
        result.stop_reason = "eos" if history[-1] == EOS else ("context" if len(history) == model.config.context else "length")
        return result
    cache = None

    def forward(tokens, past):
        result.forward_calls += 1
        result.forward_tokens += len(tokens)
        return model(torch.tensor([tokens], dtype=torch.long), past)

    # Both baseline and speculative methods leave the final committed token
    # pending. This gives them identical prefill, cache, dtype and kernels.
    if len(history) > 1:
        _, cache = forward(history[:-1], None)
    while len(result.tokens) < budget:
        remaining = budget - len(result.tokens)
        if drafter is not None and draft_length and remaining > 1:
            start = perf_counter_ns()
            proposals = list(drafter.propose(history, min(draft_length, remaining - 1)))
            result.draft_ns += perf_counter_ns() - start
            proposals = proposals[:min(draft_length, remaining - 1)]
            if any(t < 0 or t >= model.config.vocab_size for t in proposals):
                raise ValueError("drafter returned an invalid token")
            if EOS in proposals:
                proposals = proposals[:proposals.index(EOS) + 1]
        else:
            proposals = []
        result.proposed += len(proposals)
        logits, expanded = forward([history[-1], *proposals], cache)
        if audit:
            assert cache_length(cache) == len(history) - 1
            reference, _ = model(torch.tensor([history + proposals]))
            torch.testing.assert_close(logits, reference[:, len(history) - 1:], rtol=1e-4, atol=1e-5)
            if logits.argmax(-1).tolist() != reference[:, len(history) - 1:].argmax(-1).tolist():
                raise AssertionError("cached and uncached greedy predictions disagree")
        predictions = logits[0].argmax(-1).tolist()
        emitted = []
        rejected = False
        for j, token in enumerate(proposals):
            if token != predictions[j]:
                emitted.append(predictions[j])
                result.rejected_blocks += 1
                rejected = True
                break
            result.accepted += 1
            emitted.append(token)
            if token == EOS:
                break
        if proposals and not rejected and len(emitted) == len(proposals):
            result.fully_accepted_blocks += 1
        if not rejected and (not emitted or emitted[-1] != EOS):
            emitted.append(predictions[len(proposals)])
        if EOS in emitted:
            emitted = emitted[:emitted.index(EOS) + 1]
        history.extend(emitted)
        result.tokens.extend(emitted)
        # Discard rejected speculative keys/values. The correction/bonus stays
        # pending and will be processed exactly once on the next iteration.
        cache = crop_cache(expanded, len(history) - 1)
        if emitted[-1] == EOS:
            result.stop_reason = "eos"
            return result
    result.stop_reason = "context" if len(history) == model.config.context else "length"
    return result


@torch.inference_mode()
def uncached_greedy(model, prompt, max_new_tokens):
    validate(model, prompt, max_new_tokens)
    history = list(prompt)
    for _ in range(min(max_new_tokens, model.config.context - len(history))):
        if history[-1] == EOS:
            break
        logits, _ = model(torch.tensor([history], dtype=torch.long))
        history.append(int(logits[0, -1].argmax()))
    return history[len(prompt):]
