"""K-rollouts-per-prompt collector for GRPO.

Repeats each prompt K times to a flat (B*K)-batch, generates one completion
per row, caches log π_old and RM scores. Also returns the group_id tensor.
"""
from __future__ import annotations

import torch

from alignment.rollout import generate_batch, compute_response_logprobs
from alignment.grpo import GroupBatch
from model.rm_loader import score_texts
from model.loading import frozen_ref


@torch.no_grad()
def collect_group_rollout(policy, rm, rm_tok, policy_tok, prompts,
                          device, K: int = 4,
                          max_new_tokens: int = 128,
                          temperature: float = 0.7, top_p: float = 0.9) -> GroupBatch:
    B = len(prompts)
    flat_prompts = [p for p in prompts for _ in range(K)]
    group_id = torch.tensor([b for b in range(B) for _ in range(K)],
                            dtype=torch.long)

    gen = generate_batch(policy, policy_tok, flat_prompts, device,
                         max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=top_p)
    input_ids = gen["input_ids"]
    attn = gen["attention_mask"]
    resp_mask = gen["response_mask"]

    # cached π_old log-probs on shifted grid
    lp_old, _ = compute_response_logprobs(policy, input_ids, attn, resp_mask)

    # RM score on full text
    texts = policy_tok.batch_decode(input_ids, skip_special_tokens=True)
    r_task = score_texts(rm, rm_tok, texts, device, batch_size=B * K)

    return GroupBatch(
        input_ids=input_ids.cpu(),
        attention_mask=attn.cpu(),
        response_mask=resp_mask.cpu(),
        logprobs_old=lp_old.detach().float().cpu(),
        r_task=r_task.detach().float().cpu(),
        group_id=group_id,
        prompt_len=gen["prompt_len"],
        K=K,
    )
