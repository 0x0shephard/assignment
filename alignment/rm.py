"""Reward-model scoring + margin ranking loss.

The reward model is an AutoModelForSequenceClassification wrapping the Llama
backbone with a scalar head (num_labels=1). We use right-padding so the last
real token — the one whose hidden state feeds the scalar head — is at index -1
of the true sequence (see `score_last_token`).

Loss (task manual §4.5 C1.2):
    L_RM = -E[log sigmoid(r+ - r-)] + λ_reg * E[(r+)^2 + (r-)^2]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def score_last_token(model, input_ids, attention_mask):
    """Return the scalar reward per example.

    For right-padded inputs, the last non-pad token is at index
    (attention_mask.sum(dim=1) - 1).  AutoModelForSequenceClassification
    computes pooled logits internally when pad_token_id is set, but doing it
    manually is unambiguous and works even when padding_side setup drifts.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=False)
    logits = out.logits                                          
    if logits.dim() == 3:
        idx = attention_mask.sum(dim=1) - 1                         
        idx = idx.clamp(min=0)
        gather = idx.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        logits = logits.gather(1, gather).squeeze(1)                    
    return logits.squeeze(-1)                                         


@dataclass
class RMLossOut:
    loss: torch.Tensor
    r_pos: torch.Tensor
    r_neg: torch.Tensor
    accuracy: torch.Tensor


def rm_loss(r_pos, r_neg, lambda_reg: float = 1e-3) -> RMLossOut:
    ranking = -F.logsigmoid(r_pos - r_neg).mean()
    reg = ((r_pos ** 2) + (r_neg ** 2)).mean()
    loss = ranking + lambda_reg * reg
    acc = (r_pos > r_neg).float().mean()
    return RMLossOut(loss=loss, r_pos=r_pos.detach(), r_neg=r_neg.detach(), accuracy=acc.detach())
