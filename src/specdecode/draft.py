"""Drafters operate only on committed history; neither sees evaluation answers."""
from collections import Counter, defaultdict


class PromptLookup:
    def __init__(self, order=4):
        if order < 1:
            raise ValueError("order must be positive")
        self.order = order

    def propose(self, history, count):
        # Longest suffix, earliest previous occurrence, bounded by known history.
        for n in range(min(self.order, len(history)), 0, -1):
            suffix = history[-n:]
            for start in range(len(history) - n):
                if history[start:start + n] == suffix:
                    return history[start + n:min(start + n + count, len(history))]
        return []


class NGram:
    def __init__(self, sequences, order=4):
        if order < 1:
            raise ValueError("order must be positive")
        self.order = order
        counts = defaultdict(Counter)
        for tokens in sequences:
            for i, token in enumerate(tokens):
                for n in range(min(order, i) + 1):
                    counts[tuple(tokens[i - n:i])][token] += 1
        # Longest-context backoff; deterministic smallest-token tie breaking.
        self.next_token = {key: min(c, key=lambda t: (-c[t], t)) for key, c in counts.items()}

    def propose(self, history, count):
        work = list(history)
        result = []
        for _ in range(count):
            for n in range(min(self.order, len(work)), -1, -1):
                key = tuple(work[len(work) - n:]) if n else ()
                if key in self.next_token:
                    token = self.next_token[key]
                    result.append(token)
                    work.append(token)
                    break
            else:
                break
        return result
