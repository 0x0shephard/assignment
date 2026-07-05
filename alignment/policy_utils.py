"""Shared helpers for computing per-token log-probs and entropy under the
current policy π_θ. Used by PPO and GRPO update loops.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def logprobs_and_entropy(model, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         response_mask: torch.Tensor):
    """Teacher-forced forward. Returns:
        lp:  [B, T-1]  log π_θ(a_t | s_t) at each shifted position
        ent: [B, T-1]  categorical entropy at each shifted position
        m:   [B, T-1]  response mask aligned to the shifted grid
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]                                        
    ids = input_ids[:, 1:]                                             
    mask = response_mask[:, 1:].float()                                
    lp_all = F.log_softmax(logits.float(), dim=-1)
    lp = lp_all.gather(-1, ids.unsqueeze(-1)).squeeze(-1)              
    ent = -(lp_all.exp() * lp_all).sum(dim=-1)                         
    return lp * mask, ent * mask, mask
