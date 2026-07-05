"""DPO: log-prob computation + loss from Eq. (10) of the manual.

    L_DPO = -E [ log σ ( β·Δ_θ − β·Δ_ref ) ]
    Δ_θ   =  log π_θ(y+|x) − log π_θ(y-|x)
    Δ_ref =  log π_ref(y+|x) − log π_ref(y-|x)

Key implementation care items (§4.8 pitfalls):
- Sum log-probs over response tokens ONLY (mask prompt + padding).
- π_ref is frozen — always under `torch.no_grad()`.
- Padding mask must exclude pad tokens; verified by padding-invariance test.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def sequence_log_prob(model, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      prompt_len: torch.Tensor) -> torch.Tensor:
    """Sum log π_θ(y_t | x, y_<t) over response tokens only.

    Args:
        input_ids:      [B, T]         full (prompt ⊕ response) sequence.
        attention_mask: [B, T]         1 for real tokens (padding excluded).
        prompt_len:     [B]            index (in full seq) where response starts.

    We build a per-token mask that is 1 for positions t >= prompt_len AND
    attention_mask[t] == 1. Then use the causal shift-by-one alignment: the
    log-prob at token t comes from logits at position t-1.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=False).logits                       
    shift_logits = logits[:, :-1, :]                                          
    shift_ids = input_ids[:, 1:]                                
    shift_attn = attention_mask[:, 1:].float()                  

    B, Tm1 = shift_ids.shape
    pos = torch.arange(Tm1, device=shift_ids.device).unsqueeze(0).expand(B, Tm1) + 1
    resp_mask = (pos >= prompt_len.to(pos.device).unsqueeze(1)).float() * shift_attn

    lp_all = F.log_softmax(shift_logits.float(), dim=-1)
    lp = lp_all.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)             
    return (lp * resp_mask).sum(dim=1)                                 


@dataclass
class DPOLossOut:
    loss: torch.Tensor
    reward_margin_z: torch.Tensor                                               
    pref_accuracy: torch.Tensor
    delta_theta: torch.Tensor
    delta_ref: torch.Tensor


def dpo_loss(logp_pi_chosen, logp_pi_rejected,
             logp_ref_chosen, logp_ref_rejected,
             beta: float = 0.1) -> DPOLossOut:
    delta_theta = logp_pi_chosen - logp_pi_rejected
    delta_ref = logp_ref_chosen - logp_ref_rejected
    z = beta * (delta_theta - delta_ref)
    loss = -F.logsigmoid(z).mean()
    with torch.no_grad():
        acc = (logp_pi_chosen > logp_pi_rejected).float().mean()
    return DPOLossOut(
        loss=loss,
        reward_margin_z=z.detach(),
        pref_accuracy=acc.detach(),
        delta_theta=delta_theta.detach(),
        delta_ref=delta_ref.detach(),
    )
