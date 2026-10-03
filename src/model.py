import math
from dataclasses import dataclass, asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int = 0
    n_embd: int = 0
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    tie_word_emb: bool = True

    def __post_init__(self):
        if self.n_embd == 0:
            self.n_embd = self.n_heads * 128
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        assert self.n_embd % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in allowed})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.o = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads // self.n_kv_heads, self.n_kv_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:
            if kv_cache[0] is None:
                kv_cache[0], kv_cache[1] = k, v
            else:
                k, v = torch.cat([kv_cache[0], k], dim=2), torch.cat([kv_cache[1], v], dim=2)
                kv_cache[0], kv_cache[1] = k, v

        q = q.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        att = (q @ k.transpose(-2, -1)) * self.scale
        if T < k.size(2):
            mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(diagonal=k.size(2) - T)
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1, dtype=torch.float32).to(x.dtype)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = MLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.output = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_word_emb:
            self.output.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, positions=None, kv_caches=None):
        x = self.tok_embeddings(input_ids)
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device)
        inv_freq = 1.0 / (self.cfg.rope_base ** (
            torch.arange(0, self.cfg.head_dim, 2, device=x.device, dtype=torch.float32) / self.cfg.head_dim))
        freqs = torch.outer(positions.float(), inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0).unsqueeze(0)

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, kv_caches[i] if kv_caches else None)
        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=0, eos_id=None):
        kv_caches = [None] * self.cfg.n_layers
        cur = input_ids
        out = input_ids[0].tolist()
        for _ in range(max_new_tokens):
            logits = self.forward(cur, positions=torch.tensor([len(out) - 1], device=cur.device),
                                  kv_caches=kv_caches)[:, -1]
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            cur = nxt
            out.append(nxt.item())
            if eos_id is not None and nxt.item() == eos_id:
                break
        return torch.tensor([out])

    def state_dict_named(self) -> dict[str, torch.Tensor]:
        d = {}
        d["tok_embeddings.weight"] = self.tok_embeddings.weight
        d["norm.weight"] = self.norm.weight
        if not self.cfg.tie_word_emb:
            d["output.weight"] = self.output.weight
        for i, b in enumerate(self.blocks):
            d[f"blocks.{i}.attn.qkv.weight"] = b.attn.qkv.weight
            d[f"blocks.{i}.attn.o.weight"] = b.attn.o.weight
            d[f"blocks.{i}.ffn.gate.weight"] = b.ffn.gate.weight
            d[f"blocks.{i}.ffn.up.weight"] = b.ffn.up.weight
            d[f"blocks.{i}.ffn.down.weight"] = b.ffn.down.weight
            d[f"blocks.{i}.norm1.weight"] = b.norm1.weight
            d[f"blocks.{i}.norm2.weight"] = b.norm2.weight
        return d
