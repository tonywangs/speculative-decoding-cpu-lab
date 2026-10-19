"""Small byte GRU and request-local committed-prefix state for greedy drafting."""
import torch
from torch import nn


class ByteGRU(nn.Module):
    def __init__(self, vocab_size=261, width=32):
        super().__init__()
        self.vocab_size, self.width = vocab_size, width
        self.embedding = nn.Embedding(vocab_size, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, ids, state=None):
        features, state = self.gru(self.embedding(ids), state)
        return self.output(features), state


class GRUDraft:
    """Cache only committed tokens; rejected proposals never enter shared state.

    decode() resets this state for every request, including warmups and timings.
    Accepted draft tokens are processed again on the next proposal call, so the
    optimization avoids prefix replay without requiring speculative rollback.
    """
    def __init__(self, model):
        if model.training:
            raise ValueError("draft model must be in evaluation mode")
        self.model = model
        self.reset()

    def reset(self):
        self.history = []
        self.state = None
        self.logits = None

    @torch.inference_mode()
    def propose(self, history, count):
        if not history or count < 0:
            raise ValueError("nonempty history and nonnegative count required")
        if not count:
            return []
        if history[:len(self.history)] != self.history:
            self.reset()
        suffix = history[len(self.history):]
        if suffix:
            logits, self.state = self.model(torch.tensor([suffix]), self.state)
            self.logits = logits[:, -1:]
            self.history = list(history)
        logits, state = self.logits, self.state
        result = []
        for i in range(count):
            token = int(logits[0, -1].argmax())
            result.append(token)
            if i + 1 < count:
                logits, state = self.model(torch.tensor([[token]]), state)
        return result
