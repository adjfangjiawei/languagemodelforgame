"""
High-Speed GTX 1660 Ti (Turing sm_75, 6GB VRAM, FP16 AMP) Llama-Style Causal Language Model.

Speed & Parameter-Reduction Architecture Upgrades:
1. Factorized Tied Token Embedding (`d_embed < d_model`, ALBERT / MobileLLM technique):
   Decouples the 10,240-token vocabulary width (`d_embed=128`) from the Transformer hidden width (`d_model=256`),
   shrinking both embedding parameters AND output `lm_head` GEMM FLOPs by 4x vs `d_model=512` while keeping
   full `head_dim=64` inside every attention layer!
2. Grouped-Query Attention (GQA, `n_kv_heads <= n_heads`):
   Reduces KV projection parameters, activation bandwidth, and KV-cache footprint during CoT generation.
3. Native-Dtype RoPE (`cos`/`sin` in `[-1, 1]` stay in FP16 without FP32 round-trip casts) + `FP32SafeRMSNorm`
   (prevents >65504 overflow on Turing `sm_75` FP16).
4. Single-GEMM Fast Cross-Entropy for 2M~18M models (zero autograd checkpoint recomputation, zero CPU-GPU `.item()`
   synchronization barriers during training), with optional `chunked_ce=True` reserved only for 45M+ models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _check_sdpa_enable_gqa() -> bool:
    try:
        q = torch.zeros((1, 2, 2, 8), dtype=torch.float32)
        kv = torch.zeros((1, 1, 2, 8), dtype=torch.float32)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_HAS_ENABLE_GQA = _check_sdpa_enable_gqa()


@dataclass
class ModelConfig:
    vocab_size: int = 10240
    max_seq_len: int = 1024
    d_model: int = 256
    d_embed: Optional[int] = 128   # If < d_model, uses Factorized Tied Embedding (cuts 10k-vocab params & FLOPs!)
    n_layers: int = 4
    n_heads: int = 4               # head_dim = 64 (aligned for CUDA SDPA fast kernels)
    n_kv_heads: Optional[int] = 2  # GQA KV heads (defaults to n_heads if None)
    d_ff: int = 640                # Multiple of 64/128
    dropout: float = 0.0
    rope_theta: float = 10000.0
    chunked_ce: bool = False       # Direct single-GEMM CE is much faster for <=18M models; True for >=45M
    pad_token_id: int = 0
    eos_token_id: int = 3

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# Presets spanning 2.7M -> 88.4M (Default `fast-4m` = 4.13M params, ~10.6x smaller than `base-45m`!)
MODEL_PRESETS: Dict[str, Dict] = {
    "nano-2m":   {"d_embed": 96,  "d_model": 192, "n_layers": 4,  "n_heads": 3,  "n_kv_heads": 1,  "d_ff": 512,  "chunked_ce": False},  # ~2.68M
    "fast-4m":   {"d_embed": 128, "d_model": 256, "n_layers": 4,  "n_heads": 4,  "n_kv_heads": 2,  "d_ff": 640,  "chunked_ce": False},  # ~4.13M (~10.6x smaller!)
    "light-6m":  {"d_embed": 128, "d_model": 256, "n_layers": 6,  "n_heads": 4,  "n_kv_heads": 2,  "d_ff": 704,  "chunked_ce": False},  # ~5.74M
    "small-9m":  {"d_embed": 160, "d_model": 320, "n_layers": 6,  "n_heads": 5,  "n_kv_heads": 5,  "d_ff": 896,  "chunked_ce": False},  # ~9.38M
    "small-16m": {"d_embed": 384, "d_model": 384, "n_layers": 8,  "n_heads": 6,  "n_kv_heads": 6,  "d_ff": 1024, "chunked_ce": False},  # ~18.4M
    "base-45m":  {"d_embed": 512, "d_model": 512, "n_layers": 12, "n_heads": 8,  "n_kv_heads": 8,  "d_ff": 1408, "chunked_ce": True},   # ~43.8M
    "large-88m": {"d_embed": 640, "d_model": 640, "n_layers": 16, "n_heads": 10, "n_kv_heads": 10, "d_ff": 1792, "chunked_ce": True},   # ~88.4M
}


class FP32SafeRMSNorm(nn.Module):
    """RMSNorm accumulating squares in FP32 so Turing FP16 training never overflows 65,504."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(orig_dtype) * self.weight


def precompute_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Fast native-dtype RoPE (`cos`/`sin` in `[-1, 1]` never overflow FP16).
    Supports both shared `[T, D/2]` and per-batch `[B, T, D/2]` position tables (for multi-board self-play).
    """
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    if cos.dim() == 2:
        cos_b = cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin_b = sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    else:
        cos_b = cos.to(dtype=x.dtype).unsqueeze(1)
        sin_b = sin.to(dtype=x.dtype).unsqueeze(1)
    return torch.cat([x1 * cos_b - x2 * sin_b, x1 * sin_b + x2 * cos_b], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            new_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = (k, v)
            is_causal = (attn_mask is None) and (T > 1 and k.shape[2] == T)
        else:
            new_cache = None
            is_causal = attn_mask is None

        if self.n_rep > 1:
            if _SDPA_HAS_ENABLE_GQA:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal, enable_gqa=True
                )
            else:
                k_exp = k.repeat_interleave(self.n_rep, dim=1)
                v_exp = v.repeat_interleave(self.n_rep, dim=1)
                y = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal
                )
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y)), new_cache


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.drop(self.down_proj(F.silu(gate) * up))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = FP32SafeRMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = FP32SafeRMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache=kv_cache, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class XiangqiCausalLM(nn.Module):
    """Causal Dialogue Language Model supporting Factorized Embeddings, GQA, and Fast Cross-Entropy."""

    def __init__(self, cfg: ModelConfig, grad_checkpoint: bool = False):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = grad_checkpoint
        self.d_embed = cfg.d_embed if (cfg.d_embed is not None and cfg.d_embed < cfg.d_model) else cfg.d_model
        self.is_factorized = self.d_embed < cfg.d_model

        self.tok_emb = nn.Embedding(cfg.vocab_size, self.d_embed)
        self.emb_up: Optional[nn.Linear] = (
            nn.Linear(self.d_embed, cfg.d_model, bias=False) if self.is_factorized else None
        )
        self.emb_drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = FP32SafeRMSNorm(cfg.d_model)

        self.head_down: Optional[nn.Linear] = (
            nn.Linear(cfg.d_model, self.d_embed, bias=False) if self.is_factorized else None
        )
        self.lm_head = nn.Linear(self.d_embed, cfg.vocab_size, bias=False)

        # Weight tying between input token embedding and output vocabulary projection
        self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope_cache(cfg.max_seq_len, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        resid_std = 0.02 / ((2 * cfg.n_layers) ** 0.5)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o_proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(blk.ffn.down_proj.weight, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(input_ids)
        if self.emb_up is not None:
            h = self.emb_up(h)
        return self.emb_drop(h)

    def _project_pre_vocab(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_down(x) if self.head_down is not None else x

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def _chunked_fp32_ce_sum(
        self,
        h_act: torch.Tensor,
        y_act: torch.Tensor,
        chunk_size: int = 2048,
    ) -> Tuple[torch.Tensor, int]:
        """Optional low-VRAM chunked CE reserved for >=45M models."""
        w = self.lm_head.weight
        n_total = int(y_act.shape[0])

        def _chunk_nll(hc: torch.Tensor, yc: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            lg = F.linear(hc, weight).float()
            return F.cross_entropy(lg, yc, reduction="sum")

        total_nll: Optional[torch.Tensor] = None
        correct = 0
        for start in range(0, n_total, chunk_size):
            hc = h_act[start : start + chunk_size]
            yc = y_act[start : start + chunk_size]
            if torch.is_grad_enabled() and (hc.requires_grad or w.requires_grad):
                part = checkpoint(_chunk_nll, hc, yc, w, use_reentrant=False)
            else:
                lg = F.linear(hc, w).float()
                part = F.cross_entropy(lg, yc, reduction="sum")
                correct += int((lg.argmax(dim=-1) == yc).sum().item())
            total_nll = part if total_nll is None else (total_nll + part)
        assert total_nll is not None
        return total_nll, correct

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_logits: Optional[bool] = None,
        return_acc: bool = False,
    ):
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.cfg.max_seq_len}")

        x = self._embed_tokens(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(lambda h, b=blk: b(h, cos, sin, None)[0], x, use_reentrant=False)
            else:
                x, _ = blk(x, cos, sin, kv_cache=None)

        h = self._project_pre_vocab(self.norm_f(x))

        if return_logits is None:
            return_logits = labels is None

        if labels is None:
            logits = self.lm_head(h) if return_logits else None
            return (logits, None, 0, 0) if return_acc else (logits, None)

        # Fast path (default for 2M ~ 18M models): ONE single fused GEMM + zero CPU-GPU sync!
        if not self.cfg.chunked_ce:
            logits = self.lm_head(h)
            loss = F.cross_entropy(
                logits.float().view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            if return_acc:
                with torch.no_grad():
                    mask = labels != -100
                    n_active = int(mask.sum().item())
                    n_correct = int(((logits.argmax(dim=-1) == labels) & mask).sum().item()) if n_active > 0 else 0
                return (logits if return_logits else None), loss, n_correct, n_active
            return (logits if return_logits else None), loss

        # Low-VRAM chunked path (for 45M ~ 88M models on 6GB GPUs)
        logits_opt = self.lm_head(h) if return_logits else None
        flat_h = h.reshape(-1, self.d_embed)
        flat_y = labels.reshape(-1)
        mask = flat_y != -100
        n_active = int(mask.sum().item())
        if n_active > 0:
            total_nll, n_correct = self._chunked_fp32_ce_sum(flat_h[mask], flat_y[mask])
            loss = total_nll / float(n_active)
        else:
            loss = h.sum() * 0.0
            n_correct = 0

        if return_acc:
            return logits_opt, loss, n_correct, n_active
        return logits_opt, loss

    @torch.no_grad()
    def stream_generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 940,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.12,
        rep_window: int = 40,
        no_repeat_ngram_size: int = 8,
        exempt_rep_ids: Optional[set] = None,
        eos_token_id: Optional[int] = None,
        end_think_id: Optional[int] = None,
        force_close_ids: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> Iterator[int]:
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        stop_id = self.cfg.eos_token_id if eos_token_id is None else eos_token_id
        exempt = exempt_rep_ids if exempt_rep_ids is not None else set(range(16))

        max_prompt = self.cfg.max_seq_len - 16
        ids = list(prompt_ids[:max_prompt])
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        T0 = inp.shape[1]
        x = self._embed_tokens(inp)
        cos = self.rope_cos[:T0]
        sin = self.rope_sin[:T0]

        n_kv = self.cfg.n_kv_heads if self.cfg.n_kv_heads is not None else self.cfg.n_heads
        head_dim = self.cfg.d_model // self.cfg.n_heads
        empty_k = torch.empty((1, n_kv, 0, head_dim), dtype=x.dtype, device=device)
        empty_v = torch.empty((1, n_kv, 0, head_dim), dtype=x.dtype, device=device)
        caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for blk in self.blocks:
            x, c = blk(x, cos, sin, kv_cache=(empty_k, empty_v))
            assert c is not None
            caches.append(c)

        h_last = self._project_pre_vocab(self.norm_f(x[:, -1:, :]))
        logits = self.lm_head(h_last)[:, 0, :].float()

        generated: List[int] = []
        cur_pos = T0
        limit = min(max_new_tokens, self.cfg.max_seq_len - T0)
        reserve_for_close = (len(force_close_ids) + 8) if force_close_ids else 0

        def advance_one(token_id: int) -> None:
            nonlocal caches, logits, cur_pos
            generated.append(token_id)
            if cur_pos >= self.cfg.max_seq_len:
                return
            step_inp = torch.tensor([[token_id]], dtype=torch.long, device=device)
            hx = self._embed_tokens(step_inp)
            cos_step = self.rope_cos[cur_pos : cur_pos + 1]
            sin_step = self.rope_sin[cur_pos : cur_pos + 1]
            new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for idx, blk in enumerate(self.blocks):
                hx, c = blk(hx, cos_step, sin_step, kv_cache=caches[idx])
                assert c is not None
                new_caches.append(c)
            caches = new_caches
            h_step = self._project_pre_vocab(self.norm_f(hx))
            logits = self.lm_head(h_step)[:, 0, :].float()
            cur_pos += 1

        while len(generated) < limit and cur_pos < self.cfg.max_seq_len:
            if (
                force_close_ids
                and end_think_id is not None
                and end_think_id not in generated
                and (limit - len(generated)) <= reserve_for_close
            ):
                for cid in force_close_ids:
                    if cur_pos >= self.cfg.max_seq_len:
                        break
                    advance_one(cid)
                    yield cid
                continue

            step_logits = logits.clone()

            if repetition_penalty > 1.0 and generated:
                recent = set(generated[-rep_window:]) - exempt
                for tid in recent:
                    if step_logits[0, tid] > 0:
                        step_logits[0, tid] /= repetition_penalty
                    else:
                        step_logits[0, tid] *= repetition_penalty

            if no_repeat_ngram_size > 1 and len(generated) >= no_repeat_ngram_size - 1:
                prefix = tuple(generated[-(no_repeat_ngram_size - 1) :])
                banned: List[int] = []
                for i in range(len(generated) - no_repeat_ngram_size + 1):
                    if tuple(generated[i : i + no_repeat_ngram_size - 1]) == prefix:
                        banned.append(generated[i + no_repeat_ngram_size - 1])
                for b_id in banned:
                    if b_id not in exempt:
                        step_logits[0, b_id] = -1e9

            if temperature <= 1e-5:
                next_id = int(torch.argmax(step_logits, dim=-1).item())
            else:
                probs = F.softmax(step_logits / temperature, dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cum_probs > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = int(sorted_idx.gather(-1, sampled).item())
                else:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_id == stop_id:
                break
            advance_one(next_id)
            yield next_id

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> List[int]:
        return list(self.stream_generate(*args, **kwargs))

    @torch.no_grad()
    def generate_group(
        self,
        prompt_ids: List[int],
        group_size: int = 6,
        max_new_tokens: int = 920,
        temperature: float = 0.85,
        top_p: float = 0.95,
        repetition_penalty: float = 1.10,
        rep_window: int = 40,
        exempt_rep_ids: Optional[set] = None,
        eos_token_id: Optional[int] = None,
        end_think_id: Optional[int] = None,
        force_close_ids: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> List[List[int]]:
        """
        Batched parallel KV-cache rollout of `group_size` diverse CoT trajectories for one prompt `q`
        (used by GRPO post-training RL exploration — ~6x-8x faster on GPU than sequential sampling!).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        G = max(1, int(group_size))
        stop_id = self.cfg.eos_token_id if eos_token_id is None else eos_token_id
        exempt = exempt_rep_ids if exempt_rep_ids is not None else set(range(16))

        max_prompt = self.cfg.max_seq_len - 16
        ids = list(prompt_ids[:max_prompt])
        inp = torch.tensor([ids] * G, dtype=torch.long, device=device)  # [G, T0]

        T0 = inp.shape[1]
        x = self._embed_tokens(inp)
        cos = self.rope_cos[:T0]
        sin = self.rope_sin[:T0]

        n_kv = self.cfg.n_kv_heads if self.cfg.n_kv_heads is not None else self.cfg.n_heads
        head_dim = self.cfg.d_model // self.cfg.n_heads
        empty_k = torch.empty((G, n_kv, 0, head_dim), dtype=x.dtype, device=device)
        empty_v = torch.empty((G, n_kv, 0, head_dim), dtype=x.dtype, device=device)
        caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for blk in self.blocks:
            x, c = blk(x, cos, sin, kv_cache=(empty_k, empty_v))
            assert c is not None
            caches.append(c)

        h_last = self._project_pre_vocab(self.norm_f(x[:, -1:, :]))
        logits = self.lm_head(h_last)[:, 0, :].float()  # [G, V]

        gen_lists: List[List[int]] = [[] for _ in range(G)]
        finished = [False] * G
        has_closed_think = [False] * G
        close_queue_pos = [0] * G

        cur_pos = T0
        limit = min(max_new_tokens, self.cfg.max_seq_len - T0)
        reserve_for_close = (len(force_close_ids) + 8) if force_close_ids else 0

        for step_idx in range(limit):
            if cur_pos >= self.cfg.max_seq_len or all(finished):
                break

            step_logits = logits.clone()
            if repetition_penalty > 1.0:
                for b in range(G):
                    if finished[b] or not gen_lists[b]:
                        continue
                    recent = set(gen_lists[b][-rep_window:]) - exempt
                    for tid in recent:
                        if step_logits[b, tid] > 0:
                            step_logits[b, tid] /= repetition_penalty
                        else:
                            step_logits[b, tid] *= repetition_penalty

            if temperature <= 1e-5:
                sampled_ids = torch.argmax(step_logits, dim=-1).tolist()
            else:
                probs = F.softmax(step_logits / temperature, dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cum_probs > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    picks = torch.multinomial(sorted_probs, num_samples=1)
                    sampled_ids = sorted_idx.gather(-1, picks).squeeze(-1).tolist()
                else:
                    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

            next_tokens: List[int] = []
            for b in range(G):
                if finished[b]:
                    next_tokens.append(self.cfg.pad_token_id)
                    continue

                if (
                    force_close_ids
                    and end_think_id is not None
                    and not has_closed_think[b]
                    and (
                        close_queue_pos[b] > 0
                        or (limit - step_idx) <= reserve_for_close
                    )
                    and close_queue_pos[b] < len(force_close_ids)
                ):
                    tid = force_close_ids[close_queue_pos[b]]
                    close_queue_pos[b] += 1
                    gen_lists[b].append(tid)
                    if close_queue_pos[b] == len(force_close_ids):
                        has_closed_think[b] = True
                    next_tokens.append(tid)
                    continue

                tid = int(sampled_ids[b])
                if tid == stop_id:
                    finished[b] = True
                    next_tokens.append(self.cfg.pad_token_id)
                else:
                    gen_lists[b].append(tid)
                    if end_think_id is not None and tid == end_think_id:
                        has_closed_think[b] = True
                    next_tokens.append(tid)

            if all(finished):
                break

            step_inp = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)  # [G, 1]
            hx = self._embed_tokens(step_inp)
            cos_step = self.rope_cos[cur_pos : cur_pos + 1]
            sin_step = self.rope_sin[cur_pos : cur_pos + 1]
            new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for idx, blk in enumerate(self.blocks):
                hx, c = blk(hx, cos_step, sin_step, kv_cache=caches[idx])
                assert c is not None
                new_caches.append(c)
            caches = new_caches
            h_step = self._project_pre_vocab(self.norm_f(hx))
            logits = self.lm_head(h_step)[:, 0, :].float()
            cur_pos += 1

        return gen_lists

    def compute_token_logprobs(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_last_idx: Optional[torch.Tensor] = None,
    ):
        """
        Compute exact per-token FP32 log-probabilities `log pi(target | prefix)` of shape `[B, T]`
        plus approximate token-level entropy `[B, T]`.
        If `prompt_last_idx` (`[B]` long tensor) is given, also returns `prompt_hidden` (`[B, d_model]`)
        from the SAME single forward pass for AlphaGo Value Head `V(s)` joint training!
        """
        B, T = input_ids.shape
        x = self._embed_tokens(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(lambda h, b=blk: b(h, cos, sin, None)[0], x, use_reentrant=False)
            else:
                x, _ = blk(x, cos, sin, kv_cache=None)

        norm_x = self.norm_f(x)  # [B, T, d_model]
        h = self._project_pre_vocab(norm_x)
        logits_fp32 = self.lm_head(h).float()  # [B, T, V]

        safe_targets = target_ids.clamp(min=0, max=self.cfg.vocab_size - 1)
        logp = -F.cross_entropy(
            logits_fp32.view(-1, self.cfg.vocab_size),
            safe_targets.view(-1),
            reduction="none",
        ).view(B, T)

        with torch.no_grad():
            probs = F.softmax(logits_fp32, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)

        if prompt_last_idx is not None:
            row_idx = torch.arange(B, device=input_ids.device)
            safe_pidx = prompt_last_idx.clamp(min=0, max=T - 1)
            prompt_hidden = norm_x[row_idx, safe_pidx, :]  # [B, d_model]
            return logp, entropy, prompt_hidden

        return logp, entropy

    def forward_prompt_hidden(
        self,
        prompt_ids_batch: List[List[int]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Batched left-padded forward pass returning the final-token normalized hidden state `[B, d_model]`
        for `B` arbitrary-length board prompts (exact RoPE positions `0 .. L_b - 1`, NaN-safe mask).
        Used by the AlphaGo Self-Play Value Head `V_phi(s)` during lookahead & bootstrapping!
        """
        if device is None:
            device = next(self.parameters()).device
        B = len(prompt_ids_batch)
        trunc = [list(p[: self.cfg.max_seq_len]) for p in prompt_ids_batch]
        lens = [max(1, len(p)) for p in trunc]
        T0 = max(lens)

        inp = torch.full((B, T0), self.cfg.pad_token_id, dtype=torch.long, device=device)
        valid_kv = torch.zeros((B, T0), dtype=torch.bool, device=device)
        pos_ids = torch.zeros((B, T0), dtype=torch.long, device=device)
        ar = torch.arange(T0, device=device)

        for b, p in enumerate(trunc):
            Lb = len(p)
            pad_c = T0 - Lb
            inp[b, pad_c:] = torch.tensor(p, dtype=torch.long, device=device)
            valid_kv[b, pad_c:] = True
            pos_ids[b] = (ar - pad_c).clamp_min(0)

        causal = torch.tril(torch.ones((T0, T0), dtype=torch.bool, device=device))
        eye = torch.eye(T0, dtype=torch.bool, device=device)
        # Diagonal `eye` prevents softmax(-inf) NaNs on left-pad dummy queries while valid_kv blocks attending to them!
        attn_mask = ((causal.unsqueeze(0) & valid_kv.unsqueeze(1)) | eye.unsqueeze(0)).unsqueeze(1)

        x = self._embed_tokens(inp)
        cos = self.rope_cos[pos_ids]  # [B, T0, D/2]
        sin = self.rope_sin[pos_ids]
        for blk in self.blocks:
            x, _ = blk(x, cos, sin, kv_cache=None, attn_mask=attn_mask)

        norm_last = self.norm_f(x[:, -1, :])  # [B, d_model]
        return norm_last

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids_batch: List[List[int]],
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        temperatures: Optional[List[float]] = None,
        top_p: float = 0.95,
        repetition_penalty: float = 1.10,
        rep_window: int = 72,
        exempt_rep_ids: Optional[Set[int]] = None,
        eos_token_id: Optional[int] = None,
        end_think_id: Optional[int] = None,
        force_close_ids: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> List[List[int]]:
        """
        Exact-RoPE Left-Padded Parallel KV-Cache Generation across `B` distinct Xiangqi games/boards!
        Lets AlphaGo Self-Play run 8~16 simultaneous games (and K-CoT lookahead candidates) in ONE GPU batch!
        """
        self.eval()
        if not prompt_ids_batch:
            return []
        if device is None:
            device = next(self.parameters()).device

        B = len(prompt_ids_batch)
        stop_id = self.cfg.eos_token_id if eos_token_id is None else eos_token_id
        exempt = exempt_rep_ids if exempt_rep_ids is not None else set(range(16))

        max_prompt = self.cfg.max_seq_len - 16
        trunc = [list(p[:max_prompt]) for p in prompt_ids_batch]
        lens = [len(p) for p in trunc]
        T0 = max(lens)

        inp = torch.full((B, T0), self.cfg.pad_token_id, dtype=torch.long, device=device)
        valid_kv = torch.zeros((B, T0), dtype=torch.bool, device=device)
        pos_ids = torch.zeros((B, T0), dtype=torch.long, device=device)
        ar = torch.arange(T0, device=device)

        for b, p in enumerate(trunc):
            Lb = len(p)
            pad_c = T0 - Lb
            inp[b, pad_c:] = torch.tensor(p, dtype=torch.long, device=device)
            valid_kv[b, pad_c:] = True
            pos_ids[b] = (ar - pad_c).clamp_min(0)

        causal = torch.tril(torch.ones((T0, T0), dtype=torch.bool, device=device))
        eye = torch.eye(T0, dtype=torch.bool, device=device)
        prefill_mask = ((causal.unsqueeze(0) & valid_kv.unsqueeze(1)) | eye.unsqueeze(0)).unsqueeze(1)

        x = self._embed_tokens(inp)
        cos = self.rope_cos[pos_ids]
        sin = self.rope_sin[pos_ids]

        n_kv = self.cfg.n_kv_heads if self.cfg.n_kv_heads is not None else self.cfg.n_heads
        head_dim = self.cfg.d_model // self.cfg.n_heads
        empty_k = torch.empty((B, n_kv, 0, head_dim), dtype=x.dtype, device=device)
        empty_v = torch.empty((B, n_kv, 0, head_dim), dtype=x.dtype, device=device)
        caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for blk in self.blocks:
            x, c = blk(x, cos, sin, kv_cache=(empty_k, empty_v), attn_mask=prefill_mask)
            assert c is not None
            caches.append(c)

        h_last = self._project_pre_vocab(self.norm_f(x[:, -1:, :]))
        logits = self.lm_head(h_last)[:, 0, :].float()  # [B, V]

        gen_lists: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B
        has_closed_think = [False] * B
        close_queue_pos = [0] * B

        cur_pos_vec = torch.tensor(lens, dtype=torch.long, device=device)
        row_limits = [min(max_new_tokens, self.cfg.max_seq_len - Lb) for Lb in lens]
        max_steps = max(row_limits)
        reserve_for_close = (len(force_close_ids) + 8) if force_close_ids else 0

        temp_vec: Optional[torch.Tensor] = None
        if temperatures is not None:
            temp_vec = torch.tensor(
                [max(1e-5, float(t)) for t in temperatures], dtype=torch.float32, device=device
            ).unsqueeze(1)

        for step_idx in range(max_steps):
            for b in range(B):
                if step_idx >= row_limits[b]:
                    finished[b] = True
            if all(finished):
                break

            step_logits = logits.clone()
            if repetition_penalty > 1.0:
                for b in range(B):
                    if finished[b] or not gen_lists[b]:
                        continue
                    recent = set(gen_lists[b][-rep_window:]) - exempt
                    for tid in recent:
                        if step_logits[b, tid] > 0:
                            step_logits[b, tid] /= repetition_penalty
                        else:
                            step_logits[b, tid] *= repetition_penalty

            if temperatures is None and temperature <= 1e-5:
                sampled_ids = torch.argmax(step_logits, dim=-1).tolist()
            else:
                scaled = (step_logits / temp_vec) if temp_vec is not None else (step_logits / max(1e-5, temperature))
                probs = F.softmax(scaled, dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cum_probs > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    picks = torch.multinomial(sorted_probs, num_samples=1)
                    sampled_ids = sorted_idx.gather(-1, picks).squeeze(-1).tolist()
                else:
                    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()
                if temperatures is not None:
                    greedy_ids = torch.argmax(step_logits, dim=-1).tolist()
                    for b, t_val in enumerate(temperatures):
                        if t_val <= 1e-5:
                            sampled_ids[b] = greedy_ids[b]

            next_tokens: List[int] = []
            for b in range(B):
                if finished[b]:
                    next_tokens.append(self.cfg.pad_token_id)
                    continue

                if (
                    force_close_ids
                    and end_think_id is not None
                    and not has_closed_think[b]
                    and (
                        close_queue_pos[b] > 0
                        or (row_limits[b] - step_idx) <= reserve_for_close
                    )
                    and close_queue_pos[b] < len(force_close_ids)
                ):
                    tid = force_close_ids[close_queue_pos[b]]
                    close_queue_pos[b] += 1
                    gen_lists[b].append(tid)
                    if close_queue_pos[b] == len(force_close_ids):
                        has_closed_think[b] = True
                    next_tokens.append(tid)
                    continue

                tid = int(sampled_ids[b])
                if tid == stop_id:
                    finished[b] = True
                    next_tokens.append(self.cfg.pad_token_id)
                else:
                    gen_lists[b].append(tid)
                    if end_think_id is not None and tid == end_think_id:
                        has_closed_think[b] = True
                    next_tokens.append(tid)

            if all(finished):
                break

            step_inp = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)  # [B, 1]
            valid_kv = torch.cat([valid_kv, torch.ones((B, 1), dtype=torch.bool, device=device)], dim=1)
            decode_mask = valid_kv[:, None, None, :]  # [B, 1, 1, T_cur]

            safe_pos = cur_pos_vec.clamp(max=self.cfg.max_seq_len - 1)
            cos_step = self.rope_cos[safe_pos].unsqueeze(1)  # [B, 1, D/2]
            sin_step = self.rope_sin[safe_pos].unsqueeze(1)

            hx = self._embed_tokens(step_inp)
            new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for idx, blk in enumerate(self.blocks):
                hx, c = blk(hx, cos_step, sin_step, kv_cache=caches[idx], attn_mask=decode_mask)
                assert c is not None
                new_caches.append(c)
            caches = new_caches
            h_step = self._project_pre_vocab(self.norm_f(hx))
            logits = self.lm_head(h_step)[:, 0, :].float()
            cur_pos_vec = cur_pos_vec + 1

        return gen_lists
