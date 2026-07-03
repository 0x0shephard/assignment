"""GRPO: critic-free RL with group-relative advantages (§4.9 / Problem 4).

    A_{b,k}    = r_{b,k} - μ_b,      μ_b = 1/K Σ_k r_{b,k}
    L_GRPO(θ) = - 1/K Σ_k 1/T_k Σ_t min( ρ_{k,t} A_{k,t},
                                         clip(ρ_{k,t}, 1-ε, 1+ε) A_{k,t} )
                + β · KL(π_θ || π_ref)

Two KL modes:
  - "full"  — exact  Σ_v π_θ(v)[log π_θ(v) - log π_ref(v)]     (per-token)
  - "mc"    — sample-token approx  log π_θ(y_t) - log π_ref(y_t)  (per-token)

The manual notes MC is unbiased but higher variance; full-vocab is expensive
for big vocabularies but exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class GroupBatch:
    """One prompt-batch expanded to B*K rows.

    All tensors have leading dim (B*K).  The `group_id` tensor maps each row
    to its prompt so we can compute group means.
    """
    input_ids: torch.Tensor        # [B*K, T]
    attention_mask: torch.Tensor   # [B*K, T]
    response_mask: torch.Tensor    # [B*K, T]
    logprobs_old: torch.Tensor     # [B*K, T-1]  aligned to shifted grid
    r_task: torch.Tensor           # [B*K]
    group_id: torch.Tensor         # [B*K]
    prompt_len: int                # scalar
    K: int


def group_advantages(r_task: torch.Tensor, group_id: torch.Tensor,
                     K: int) -> torch.Tensor:
    """A_{b,k} = r_{b,k} - μ_b.  Returns per-row advantages of shape [B*K]."""
    B = group_id.max().item() + 1
    # scatter-mean per group
    sums = torch.zeros(B, dtype=r_task.dtype, device=r_task.device)
    counts = torch.zeros(B, dtype=r_task.dtype, device=r_task.device)
    sums.scatter_add_(0, group_id, r_task)
    counts.scatter_add_(0, group_id, torch.ones_like(r_task))
    mu = sums / counts.clamp(min=1)
    return r_task - mu[group_id]


def degenerate_group_fraction(r_task: torch.Tensor, group_id: torch.Tensor) -> float:
    """Fraction of groups where all K rewards are identical (zero-gradient batches)."""
    B = group_id.max().item() + 1
    frac_degen = 0
    for b in range(B):
        rs = r_task[group_id == b]
        if rs.numel() and (rs.max() - rs.min()).abs() < 1e-8:
            frac_degen += 1
    return frac_degen / B


def standardize_batch(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = mask.float()
    v = x[m.bool()]
    mu = v.mean()
    sd = v.std()
    return ((x - mu) / (sd + eps)) * m


def per_token_kl_full(logp_new_all: torch.Tensor, logp_ref_all: torch.Tensor) -> torch.Tensor:
    """Exact per-token KL(π_θ || π_ref) using full-vocab log-probs.
    Inputs:  [B, T, V] log-softmax outputs from both models.
    Returns: [B, T]
    """
    p_new = logp_new_all.exp()
    return (p_new * (logp_new_all - logp_ref_all)).sum(dim=-1)


@dataclass
class GRPOLossOut:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_term: torch.Tensor
    approx_kl_ratio: torch.Tensor
    clip_frac: torch.Tensor
    entropy: torch.Tensor


def grpo_step_loss(logits_new: torch.Tensor,
                   logits_ref: torch.Tensor,
                   input_ids: torch.Tensor,
                   response_mask: torch.Tensor,
                   logprobs_old: torch.Tensor,
                   advantages_std: torch.Tensor,
                   beta: float = 0.1,
                   eps_clip: float = 0.2,
                   kl_mode: str = "mc",
                   K: int = 4) -> GRPOLossOut:
    """One GRPO update step over a group batch.

    Args:
        logits_new: [B*K, T, V]  from current policy π_θ
        logits_ref: [B*K, T, V]  from frozen π_ref  (no_grad; may be None if kl_mode='mc' and precomputed lp_ref given via logprobs_old channel? -- kept for clarity)
        input_ids:  [B*K, T]
        response_mask: [B*K, T]
        logprobs_old:  [B*K, T-1]  cached from rollout time
        advantages_std: [B*K]      one scalar per row, broadcast over tokens
    """
    # shift-by-one alignment (same as PPO)
    ids_shift = input_ids[:, 1:]
    m = response_mask[:, 1:].float()

    # Sampled-token log-probs via logsumexp — avoids materializing the full
    # [B*K, T-1, V] log-softmax tensor which would OOM for large vocab.
    # log π(y_t) = logits[y_t] - logsumexp(logits, dim=-1)
    #
    # Memory discipline: we deliberately compute the "new" side, free the
    # temporary fp32 tensor, then compute the "ref" side. Holding both at
    # once doubles peak allocation and OOMs on T4 (~800 MB per side).
    ln_new = logits_new[:, :-1, :].float()
    gather_new = ln_new.gather(-1, ids_shift.unsqueeze(-1)).squeeze(-1)
    lse_new = torch.logsumexp(ln_new, dim=-1)
    lp_new_tok = gather_new - lse_new                                            # [B*K, T-1]
    if kl_mode == "full":
        lp_new_all = F.log_softmax(ln_new, dim=-1)
    del ln_new

    ln_ref = logits_ref[:, :-1, :].float()
    gather_ref = ln_ref.gather(-1, ids_shift.unsqueeze(-1)).squeeze(-1)
    lse_ref = torch.logsumexp(ln_ref, dim=-1)
    lp_ref_tok = gather_ref - lse_ref
    if kl_mode == "full":
        lp_ref_all = F.log_softmax(ln_ref, dim=-1)
    del ln_ref

    ratio = torch.exp(lp_new_tok - logprobs_old)                                # ρ_{k,t}
    adv_tok = advantages_std.unsqueeze(1).expand_as(m)                          # broadcast

    unclipped = ratio * adv_tok
    clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * adv_tok
    surrogate = torch.minimum(unclipped, clipped)

    # per-row (per completion) mean over tokens with 1/T_k, then mean over B*K
    tok_counts = m.sum(dim=1).clamp(min=1)                                       # T_k
    per_row = -(surrogate * m).sum(dim=1) / tok_counts
    policy_loss = per_row.mean()

    if kl_mode == "full":
        # Full-vocab KL: lp_new_all / lp_ref_all were computed above alongside
        # the sampled-token log-probs. Memory-heavy — prefer "mc" on tight VRAM.
        kl_tok = per_token_kl_full(lp_new_all, lp_ref_all)
    elif kl_mode == "mc":
        kl_tok = lp_new_tok - lp_ref_tok                                         # unbiased MC
    else:
        raise ValueError(kl_mode)

    kl_per_row = (kl_tok * m).sum(dim=1) / tok_counts
    kl_term = kl_per_row.mean()

    loss = policy_loss + beta * kl_term

    with torch.no_grad():
        approx_kl = ((logprobs_old - lp_new_tok) * m).sum() / m.sum().clamp(min=1)
        clip_frac = (((ratio - 1.0).abs() > eps_clip).float() * m).sum() / m.sum().clamp(min=1)
        # entropy of π_θ per token, computed memory-light without materializing
        # the full log-softmax: H = logsumexp - E[logit] where E is over softmax.
        # For monitoring only, use sampled-token entropy proxy.
        entropy = -(lp_new_tok * m).sum() / m.sum().clamp(min=1)
    return GRPOLossOut(loss=loss, policy_loss=policy_loss.detach(),
                       kl_term=kl_term.detach(),
                       approx_kl_ratio=approx_kl, clip_frac=clip_frac,
                       entropy=entropy)


# ------------------------------------------------------------------
# Sanity: group advantage sums to zero within each group
# ------------------------------------------------------------------

def sanity_group_advantage():
    r = torch.tensor([1.0, 2.0, 3.0, 0.0,   5.0, 5.0, 5.0, 5.0])   # B=2, K=4
    gid = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    A = group_advantages(r, gid, K=4)
    # group 0: mean=1.5 → A=[-0.5, 0.5, 1.5, -1.5]
    # group 1: mean=5.0 → A=[0, 0, 0, 0] (degenerate)
    expected = torch.tensor([-0.5, 0.5, 1.5, -1.5, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(A, expected), (A, expected)
    frac = degenerate_group_fraction(r, gid)
    assert abs(frac - 0.5) < 1e-8, frac
    return A, frac


if __name__ == "__main__":
    A, frac = sanity_group_advantage()
    print("group_advantages:", A.tolist())
    print("degenerate fraction:", frac)
    print("GRPO sanity OK")
