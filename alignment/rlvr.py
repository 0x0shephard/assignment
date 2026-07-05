"""RLVR: verifiable reward r_v ∈ {0,1} = 1[extract(y) == gold(x)].

Also a group-rollout collector that plugs into the GRPO training loop but
scores completions with r_v instead of a learned RM.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

from data.gsm8k import extract_answer, answers_equal
from alignment.rollout import generate_batch, compute_response_logprobs
from alignment.grpo import GroupBatch


def verifiable_reward(generated_text: str, gold: float) -> float:
    got = extract_answer(generated_text)
    return 1.0 if answers_equal(got, gold) else 0.0


def has_final_number(generated_text: str) -> bool:
    """Format-compliance metric (§4.10 C6.4 step 12): did the model produce
    ANY parseable number at all, regardless of correctness?"""
    return extract_answer(generated_text) is not None


@torch.no_grad()
def collect_rlvr_rollout(policy, policy_tok, prompts: Sequence[str],
                         golds: Sequence[float], device,
                         K: int = 4,
                         max_new_tokens: int = 256,
                         temperature: float = 0.7, top_p: float = 0.9) -> GroupBatch:
    """K completions per prompt. r_v computed from the *response substring*
    (skip the prompt when decoding to avoid parsing the problem statement)."""
    B = len(prompts)
    flat_prompts = [p for p in prompts for _ in range(K)]
    flat_golds = [g for g in golds for _ in range(K)]
    group_id = torch.tensor([b for b in range(B) for _ in range(K)], dtype=torch.long)

    gen = generate_batch(policy, policy_tok, flat_prompts, device,
                         max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=top_p)
    input_ids = gen["input_ids"]
    attn = gen["attention_mask"]
    resp_mask = gen["response_mask"]
    prompt_len = gen["prompt_len"]

    lp_old, _ = compute_response_logprobs(policy, input_ids, attn, resp_mask)

    response_ids = input_ids[:, prompt_len:]
    response_texts = policy_tok.batch_decode(response_ids, skip_special_tokens=True)
    r_task = torch.tensor(
        [verifiable_reward(t, g) for t, g in zip(response_texts, flat_golds)],
        dtype=torch.float32,
    )

    return GroupBatch(
        input_ids=input_ids.cpu(),
        attention_mask=attn.cpu(),
        response_mask=resp_mask.cpu(),
        logprobs_old=lp_old.detach().float().cpu(),
        r_task=r_task,
        group_id=group_id,
        prompt_len=prompt_len,
        K=K,
    )


def format_compliance_fraction(response_texts: List[str]) -> float:
    if not response_texts:
        return 0.0
    return sum(has_final_number(t) for t in response_texts) / len(response_texts)
