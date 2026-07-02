"""Rollout collection for PPO/GRPO.

Given prompts, this module:
  1) Generates responses with `model.generate` (temperature=0.7, top-p=0.9).
  2) Runs teacher-forced forward passes on (prompt ⊕ response) to cache the
     per-token log-probs of the *actually sampled* tokens under
        - π_old  (the policy at rollout time, before any update)
        - π_ref  (frozen SFT reference — for KL shaping)
  3) Runs the value model to cache V_old(s_t) at each response step.
  4) Scores the full prompt+response string with the frozen reward model.

Padding convention: prompts are left-padded (needed for `.generate`). Responses
appended by generate are right-appended after the left-padding block, so the
"response region" in the concatenated tensor starts at index = prompt_len
(the fixed left-padded length).

Per §4.7 C3.2 Common Pitfalls: we cache π_old log-probs here and MUST NOT
recompute them during the PPO update step, otherwise ρ_t == 1 by construction
and the clip loses meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Rollout:
    # All tensors are cached on CPU (float32) unless explicitly moved.
    input_ids: torch.Tensor          # [B, T_prompt + T_resp]  full sequences
    attention_mask: torch.Tensor     # [B, T_full]
    response_mask: torch.Tensor      # [B, T_full]  1 for response tokens
    logprobs_old: torch.Tensor       # [B, T_resp]  log π_old(a_t | s_t)
    logprobs_ref: torch.Tensor       # [B, T_resp]  log π_ref(a_t | s_t)
    values_old: torch.Tensor         # [B, T_resp]  V_old(s_t)
    r_task: torch.Tensor             # [B]          scalar RM score per prompt
    prompt_len: int                  # scalar (left-padded prompt block length)

    def to(self, device):
        for name in ("input_ids", "attention_mask", "response_mask",
                     "logprobs_old", "logprobs_ref", "values_old", "r_task"):
            setattr(self, name, getattr(self, name).to(device))
        return self


def _sampled_token_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Given logits [B, T, V] and target ids [B, T], return log p(ids) per pos.
    Note we return log-probs aligned to the *target position*, i.e. output[b, t]
    is log p(ids[b, t] | prefix up to and including position t-1).
    Caller is responsible for shifting logits by -1 relative to ids.
    """
    logprobs = F.log_softmax(logits.float(), dim=-1)
    return logprobs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def compute_response_logprobs(model, input_ids, attention_mask, response_mask):
    """Teacher-forced log-probs for the sampled response tokens.

    Returns tensor of shape [B, T_resp] with 0's where response_mask is 0.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits                                # [B, T, V]
    # position t's logits predict token t+1  =>  align to next token
    shift_logits = logits[:, :-1, :]                   # [B, T-1, V]
    shift_ids = input_ids[:, 1:]                       # [B, T-1]
    shift_mask = response_mask[:, 1:]                  # [B, T-1]
    lp = _sampled_token_logprobs(shift_logits, shift_ids)
    lp = lp * shift_mask
    # Extract only the response region into [B, T_resp]. T_resp varies per row
    # if generation stopped early; we keep full T-1 length with mask instead.
    return lp, shift_mask


@torch.no_grad()
def generate_batch(policy, tok, prompts, device, max_new_tokens: int = 128,
                   temperature: float = 0.7, top_p: float = 0.9):
    """Batch-generate responses. Returns dict with input_ids, attention_mask,
    response_mask, prompt_len."""
    enc = tok(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    if temperature is None or temperature <= 0:
        gen_kwargs = dict(do_sample=False)
    else:
        gen_kwargs = dict(do_sample=True, temperature=temperature, top_p=top_p)
    gen = policy.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=False,
        use_cache=True,
        **gen_kwargs,
    )
    full_ids = gen                              # [B, prompt_len + T_resp]
    # attention mask: 1 for real prompt tokens and all response tokens; the
    # response can include trailing pad_id if it hit EOS early. We treat every
    # position >= prompt_len as a real response step until pad_token_id appears.
    attn = torch.zeros_like(full_ids)
    attn[:, :prompt_len] = enc["attention_mask"]
    # For the response region, mark positions until first EOS token as valid.
    resp = full_ids[:, prompt_len:]
    resp_mask = torch.ones_like(resp)
    if tok.eos_token_id is not None:
        # any token AFTER the first eos (per row) is padding; keep the eos itself
        eos_hits = (resp == tok.eos_token_id)
        # position of first eos per row (or last col if none)
        first_eos = torch.where(
            eos_hits.any(dim=1),
            eos_hits.float().argmax(dim=1),
            torch.full((resp.size(0),), resp.size(1) - 1, device=resp.device),
        )
        idx = torch.arange(resp.size(1), device=resp.device).unsqueeze(0)
        resp_mask = (idx <= first_eos.unsqueeze(1)).long()
    attn[:, prompt_len:] = resp_mask
    response_mask = torch.zeros_like(full_ids)
    response_mask[:, prompt_len:] = resp_mask

    return {
        "input_ids": full_ids,
        "attention_mask": attn,
        "response_mask": response_mask,
        "prompt_len": prompt_len,
    }


@torch.no_grad()
def collect_rollout(policy, ref_policy, value_model, rm, rm_tok, policy_tok,
                    prompts, device,
                    max_new_tokens: int = 128,
                    temperature: float = 0.7, top_p: float = 0.9) -> Rollout:
    """One rollout batch. All frozen forwards under no_grad."""
    # 1) sample
    gen = generate_batch(policy, policy_tok, prompts, device,
                         max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=top_p)
    input_ids = gen["input_ids"]
    attn = gen["attention_mask"]
    resp_mask = gen["response_mask"]
    prompt_len = gen["prompt_len"]

    # 2) π_old log-probs (policy at rollout time — the "old" policy for PPO)
    lp_old, shift_mask = compute_response_logprobs(policy, input_ids, attn, resp_mask)

    # 3) π_ref log-probs (LoRA disabled → base SFT weights == reference)
    from model.loading import frozen_ref
    with frozen_ref(policy):
        lp_ref, _ = compute_response_logprobs(policy, input_ids, attn, resp_mask)

    # 4) V_old per token from the separate value model
    v_old = value_model(input_ids=input_ids, attention_mask=attn)  # [B, T]
    # Values are for state s_t, not sampled token. Align to same shift.
    v_old_shift = v_old[:, :-1]                                     # [B, T-1]

    # Trim to response region only (start at prompt_len-1 in shifted view):
    resp_start = prompt_len - 1
    lp_old_r = lp_old[:, resp_start:]
    lp_ref_r = lp_ref[:, resp_start:]
    v_old_r = v_old_shift[:, resp_start:]
    m_r = shift_mask[:, resp_start:].float()

    # zero-out padded response steps
    lp_old_r = lp_old_r * m_r
    lp_ref_r = lp_ref_r * m_r
    v_old_r = v_old_r * m_r

    # 5) RM score on full x⊕y string (RM has its own tokenizer)
    texts = policy_tok.batch_decode(input_ids, skip_special_tokens=True)
    from model.rm_loader import score_texts
    r_task = score_texts(rm, rm_tok, texts, device,
                         max_len=1024, batch_size=input_ids.size(0))

    return Rollout(
        input_ids=input_ids.cpu(),
        attention_mask=attn.cpu(),
        response_mask=resp_mask.cpu(),
        logprobs_old=lp_old_r.detach().float().cpu(),
        logprobs_ref=lp_ref_r.detach().float().cpu(),
        values_old=v_old_r.detach().float().cpu(),
        r_task=r_task.detach().float().cpu(),
        prompt_len=prompt_len,
    )
