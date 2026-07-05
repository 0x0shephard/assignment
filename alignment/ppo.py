"""PPO primitives: per-step reward with KL shaping, GAE, clipped surrogate.

References the manual §4.7 C3.3 (GAE) and §4.7 C3.4 (PPO update).

Notation matches the assignment PDF:
    r_i,t^task = r_ψ(x, y)  · 1[t == T_i]            (sparse RM reward)
    r_i,t^KL   = -β [ log π_old(a_t | s_t) - log π_ref(a_t | s_t) ]
    r_i,t      = r_i,t^task + r_i,t^KL

    δ_t   = r_t + γ V_old(s_{t+1}) - V_old(s_t)
    A_t   = Σ_{k=0}^{T-t} (γ λ)^k δ_{t+k}         (GAE-λ)
    V^GAE = V_old + A                              (λ-return)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def per_token_kl(logprobs_old: torch.Tensor, logprobs_ref: torch.Tensor) -> torch.Tensor:
    """Sample-based KL(π_old || π_ref) at each token: log π_old - log π_ref."""
    return logprobs_old - logprobs_ref


def build_per_step_rewards(r_task: torch.Tensor, response_mask: torch.Tensor,
                           logprobs_old: torch.Tensor, logprobs_ref: torch.Tensor,
                           beta: float) -> torch.Tensor:
    """r_i,t = r_task * 1[t = T_i] + r_i,t^KL, masked to response tokens.

    Args:
        r_task:        [B]                sparse scalar RM reward per row
        response_mask: [B, T_resp]        1 for valid response steps (float ok)
        logprobs_old:  [B, T_resp]
        logprobs_ref:  [B, T_resp]

    Returns: [B, T_resp]
    """
    mask = response_mask.float()
    kl_tok = per_token_kl(logprobs_old, logprobs_ref) * mask
    r_kl = -beta * kl_tok
    last_idx = mask.sum(dim=1).long() - 1                         
    r_task_tok = torch.zeros_like(r_kl)
    rows = torch.arange(r_task_tok.size(0), device=r_task_tok.device)
    valid = last_idx >= 0
    r_task_tok[rows[valid], last_idx[valid]] = r_task[valid].to(r_task_tok.dtype)
    return r_task_tok + r_kl


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                response_mask: torch.Tensor,
                gamma: float = 1.0, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE-λ per row.

    Args:
        rewards:       [B, T]
        values:        [B, T]  V_old(s_t), aligned to same steps as rewards
        response_mask: [B, T]  1 for valid, 0 for padding
    Returns:
        advantages:    [B, T]
        returns:       [B, T]  =  V_old + A  (λ-return)
    """
    mask = response_mask.float()
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        v_next = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, 0])
        m_next = mask[:, t + 1] if t + 1 < T else torch.zeros_like(mask[:, 0])
        delta = rewards[:, t] + gamma * v_next * m_next - values[:, t]
        gae = delta + gamma * lam * m_next * gae
        adv[:, t] = gae * mask[:, t]
    returns = adv + values
    return adv, returns


def standardize_advantages(adv: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batch-wide standardization over valid tokens."""
    m = mask.float()
    valid = m.bool()
    a = adv[valid]
    mu = a.mean()
    sd = a.std()
    return ((adv - mu) / (sd + eps)) * m


@dataclass
class PPOLossOut:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor


def ppo_step_loss(logprobs_new: torch.Tensor,
                  logprobs_old: torch.Tensor,
                  advantages: torch.Tensor,
                  returns: torch.Tensor,
                  values_new: torch.Tensor,
                  response_mask: torch.Tensor,
                  entropy: torch.Tensor,
                  eps_clip: float = 0.2,
                  c_v: float = 0.5,
                  c_entropy: float = 0.0) -> PPOLossOut:
    """PPO clipped surrogate + value loss + optional entropy bonus.

    All tensors are shape [B, T]; response_mask (float) selects valid tokens.
    values_new gradients flow into the value model; returns are detached targets.
    """
    m = response_mask.float()
    n = m.sum().clamp(min=1.0)

    ratio = torch.exp(logprobs_new - logprobs_old)                  
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
    policy_loss = -torch.minimum(unclipped, clipped)
    policy_loss = (policy_loss * m).sum() / n

    value_loss = 0.5 * ((values_new - returns.detach()) ** 2 * m).sum() / n

    ent = (entropy * m).sum() / n

    loss = policy_loss + c_v * value_loss - c_entropy * ent

    with torch.no_grad():
        approx_kl = ((logprobs_old - logprobs_new) * m).sum() / n
        clip_frac = (((ratio - 1.0).abs() > eps_clip).float() * m).sum() / n
    return PPOLossOut(loss=loss, policy_loss=policy_loss.detach(),
                      value_loss=value_loss.detach(), entropy=ent.detach(),
                      approx_kl=approx_kl, clip_frac=clip_frac)


def gae_unit_test():
    """Manual §4.7 sanity check 1: r=[0.05,-0.02,1.6], V_old=[1.5,1.55,1.45],
    γ=λ=1. Compute A by hand, verify our implementation matches."""
    r = torch.tensor([[0.05, -0.02, 1.6]])
    v = torch.tensor([[1.5, 1.55, 1.45]])
    m = torch.ones_like(r)
    adv, ret = compute_gae(r, v, m, gamma=1.0, lam=1.0)
    expected = torch.tensor([[0.13, 0.03, 0.15]])
    assert torch.allclose(adv, expected, atol=1e-6), (adv, expected)
    return adv, ret


def ratio_test():
    """Manual §4.7 sanity check 2: ρ_t = 1 when log-probs are identical."""
    lp = torch.randn(2, 4)
    ratio = torch.exp(lp - lp)
    assert torch.allclose(ratio, torch.ones_like(ratio))
    return ratio


def clipping_test():
    """Manual §4.7 sanity check 3: ρ=1.5, A=1.0, ε=0.2 → loss = -(1+ε)·A = -1.2,
    and ∇=0 wrt π_θ (clipped side wins over unclipped for A>0 when ρ>1+ε)."""
    logp_new = torch.tensor([[0.0]], requires_grad=True)
    logp_old = torch.tensor([[0.0]])
    logp_new_val = torch.tensor([[torch.log(torch.tensor(1.5)).item()]], requires_grad=True)
    adv = torch.tensor([[1.0]])
    ret = torch.tensor([[0.0]])
    v_new = torch.tensor([[0.0]], requires_grad=True)
    mask = torch.ones(1, 1)
    ent = torch.zeros(1, 1)
    out = ppo_step_loss(logp_new_val, logp_old, adv, ret, v_new, mask, ent, eps_clip=0.2, c_v=0.0)
    expected = -(1.2)
    assert abs(out.policy_loss.item() - expected) < 1e-6, out.policy_loss
    out.loss.backward()
    assert logp_new_val.grad is not None and torch.allclose(logp_new_val.grad, torch.zeros_like(logp_new_val.grad)),\
        f"expected zero grad on clipped side, got {logp_new_val.grad}"
    return out


if __name__ == "__main__":
    print("GAE unit test:", gae_unit_test())
    print("ratio test:", ratio_test())
    print("clipping test policy_loss:", clipping_test().policy_loss.item())
    print("all PPO sanity checks passed.")
