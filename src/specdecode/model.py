"""A pre-norm causal transformer with explicit, shared KV caching."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 22
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self):
        if min(asdict(self).values()) <= 0 or self.width % self.heads:
            raise ValueError("positive dimensions and width divisible by heads required")


# Each layer holds (key, value), shaped [batch, heads, sequence, head_width].
def cache_length(cache):
    return 0 if cache is None else cache[0][0].shape[2]


def crop_cache(cache, length):
    if length < 0 or length > cache_length(cache):
        raise ValueError("invalid cache rollback")
    return [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in cache]


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.ln1 = nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.ln2 = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(),
                                nn.Linear(4 * cfg.width, cfg.width))

    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [a.view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for a in (q, k, v)]
        offset = 0 if past is None else past[0].shape[2]
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # Explicit offset mask matters: a block after a cached prefix is not
        # the same mask as an uncached block with query length == key length.
        allowed = (torch.arange(k.shape[2], device=x.device)[None, :]
                   <= offset + torch.arange(t, device=x.device)[:, None])
        score = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        weight = score.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        y = (weight @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x)), (k, v)


class TinyTransformer(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, ids, cache=None):
        offset = cache_length(cache)
        if ids.ndim != 2 or ids.shape[1] == 0 or offset + ids.shape[1] > self.config.context:
            raise ValueError("nonempty [batch, time] input within context required")
        x = self.token(ids) + self.position(torch.arange(offset, offset + ids.shape[1], device=ids.device))
        updated = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if cache is None else cache[i])
            updated.append(kv)
        return self.output(self.norm(x)), updated
