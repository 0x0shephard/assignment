"""Task C6: RLVR on GSM8K — GRPO with verifiable reward r_v ∈ {0,1}.

CRITICAL init requirement (§4.10 pitfalls): initialize from the **plain SFT**
checkpoint (Task C2), not from PPO/GRPO/DPO HH-RLHF ckpts. Ensures a fair
comparison and avoids artifacts from HH-RLHF fine-tuning.

Metrics logged every 25 steps:
- pass@1 (greedy) on eval subset
- mean group reward μ_b across batch prompts
- fraction of degenerate batches (all-same reward → zero gradient)
- mean response length (verbosity drift)
- KL from π_ref
- format compliance (fraction of responses containing ANY parseable number)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from data.gsm8k import load_gsm8k, format_prompt
from model.loading import LoadCfg, DEFAULT_POLICY, load_policy, param_stats, vram_footprint_gb, frozen_ref
from alignment.rlvr import collect_rlvr_rollout, verifiable_reward, format_compliance_fraction
from alignment.grpo import (
    group_advantages,
    degenerate_group_fraction,
    grpo_step_loss,
)
from alignment.policy_utils import logprobs_and_entropy


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_policy_from_sft(sft_dir: str, policy_name: str, device, load_in_8bit=False,
                          grad_ckpt: bool = False):
    """grad_ckpt disabled by default — RLVR rollouts call .generate() which
    is incompatible with grad_ckpt on this transformers version."""
    from peft import PeftModel
    base, tok = load_policy(LoadCfg(policy_name, load_in_8bit=load_in_8bit, device_map=None))
    model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)
    return model, tok


@torch.no_grad()
def eval_pass_at_1(policy, policy_tok, eval_items, device, max_new_tokens=256,
                   batch_size=8):
    """Greedy decode + verifiable reward for pass@1."""
    from alignment.rollout import generate_batch
    policy.eval()
    correct = 0
    total = 0
    responses = []
    lens = []
    kl_sum = 0.0
    kl_n = 0
    for i in range(0, len(eval_items), batch_size):
        chunk = eval_items[i:i + batch_size]
        prompts = [format_prompt(it["question"]) for it in chunk]
        golds = [it["gold"] for it in chunk]
        gen = generate_batch(policy, policy_tok, prompts, device,
                             max_new_tokens=max_new_tokens,
                             temperature=0.0, top_p=1.0)
        ids, attn, m = gen["input_ids"], gen["attention_mask"], gen["response_mask"]
        lp, _, mv = logprobs_and_entropy(policy, ids, attn, m)
        with frozen_ref(policy):
            lpr, _, _ = logprobs_and_entropy(policy, ids, attn, m)
        row_kl = ((lp - lpr) * mv).sum(dim=1) / mv.sum(dim=1).clamp(min=1)
        kl_sum += row_kl.sum().item()
        kl_n += row_kl.size(0)
        resp_ids = ids[:, gen["prompt_len"]:]
        texts = policy_tok.batch_decode(resp_ids, skip_special_tokens=True)
        responses.extend(texts)
        for t, g in zip(texts, golds):
            correct += int(verifiable_reward(t, g))
            total += 1
        lens.append(mv.sum(dim=1).float().cpu())
    policy.train()
    pass1 = correct / max(1, total)
    kl_mean = kl_sum / max(1, kl_n)
    mean_len = torch.cat(lens).mean().item() if lens else 0.0
    format_ok = format_compliance_fraction(responses)
    return {"pass@1": pass1, "kl_ref": kl_mean, "mean_len": mean_len,
            "format_compliance": format_ok, "sample_responses": responses[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--sft_dir", default="checkpoints/sft",
                    help="MUST be plain SFT; never a HH-RLHF RL checkpoint.")
    ap.add_argument("--out", default="checkpoints/rlvr")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--eps_clip", type=float, default=0.2)
    ap.add_argument("--kl_mode", choices=["mc", "full"], default="mc")
    ap.add_argument("--epochs_per_batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--train_limit", type=int, default=None)
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")

    print("Loading GSM8K...")
    train_items = load_gsm8k("train", limit=args.train_limit)
    eval_items = load_gsm8k("test", limit=args.eval_n)
    print(f"train={len(train_items)}  eval={len(eval_items)}")

    print(f"Loading policy π_θ from SFT ckpt: {args.sft_dir}")
    policy, policy_tok = _load_policy_from_sft(args.sft_dir, args.policy, device,
                                              load_in_8bit=False)
    print("policy trainable:", param_stats(policy), "vram:", vram_footprint_gb(), "GB")

    optim = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    ev0 = eval_pass_at_1(policy, policy_tok, eval_items[:64], device,
                         max_new_tokens=args.max_new_tokens)
    print(f"[init] pass@1 (partial 64): {ev0['pass@1']:.3f}  "
          f"format_ok={ev0['format_compliance']:.3f}  "
          f"mean_len={ev0['mean_len']:.1f}")

    log = []
    for step in range(1, args.steps + 1):
        batch = random.sample(train_items, k=args.prompts_per_step)
        prompts = [format_prompt(it["question"]) for it in batch]
        golds = [it["gold"] for it in batch]

        gb = collect_rlvr_rollout(policy, policy_tok, prompts, golds, device,
                                  K=args.K, max_new_tokens=args.max_new_tokens)

        r_dev = gb.r_task.to(device)
        gid_dev = gb.group_id.to(device)
        A_raw = group_advantages(r_dev, gid_dev, K=args.K)

        input_ids = gb.input_ids.to(device)
        attn = gb.attention_mask.to(device)
        resp_mask = gb.response_mask.to(device)
        lp_old = gb.logprobs_old.to(device)

        m_shift = resp_mask[:, 1:].float()
        A_tok = A_raw.unsqueeze(1).expand_as(m_shift) * m_shift
        A_flat = A_tok[m_shift.bool()]
        if A_flat.numel() > 0 and A_flat.std().item() > 1e-8:
            mu, sd = A_flat.mean(), A_flat.std().clamp(min=1e-8)
            A_std = (A_raw - mu) / sd
        else:
            A_std = A_raw.clone()                                             

        frac_degen = degenerate_group_fraction(gb.r_task, gb.group_id)

        for _ep in range(args.epochs_per_batch):
            logits_new = policy(input_ids=input_ids, attention_mask=attn,
                                use_cache=False).logits
            with torch.no_grad(), frozen_ref(policy):
                logits_ref = policy(input_ids=input_ids, attention_mask=attn,
                                    use_cache=False).logits
            out = grpo_step_loss(
                logits_new=logits_new, logits_ref=logits_ref,
                input_ids=input_ids, response_mask=resp_mask,
                logprobs_old=lp_old, advantages_std=A_std,
                beta=args.beta, eps_clip=args.eps_clip,
                kl_mode=args.kl_mode, K=args.K,
            )
            optim.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0,
            )
            optim.step()

        entry = {
            "step": step,
            "group_reward_mean": gb.r_task.mean().item(),               
            "policy_loss": out.policy_loss.item(),
            "kl_term": out.kl_term.item(),
            "clip_frac": out.clip_frac.item(),
            "frac_degen": frac_degen,
            "mean_resp_len": m_shift.sum(dim=1).float().mean().item(),
        }
        log.append(entry)
        print(f"[{step:03d}] μ_r={entry['group_reward_mean']:.3f} "
              f"pol={entry['policy_loss']:+.4f} kl={entry['kl_term']:+.4f} "
              f"clip%={entry['clip_frac']:.3f} degen={entry['frac_degen']:.2f} "
              f"len={entry['mean_resp_len']:.1f}")

        if step % args.eval_every == 0:
            ev = eval_pass_at_1(policy, policy_tok, eval_items, device,
                               max_new_tokens=args.max_new_tokens)
            print(f"    eval@{step}: pass@1={ev['pass@1']:.3f}  "
                  f"kl_ref={ev['kl_ref']:+.4f}  "
                  f"format_ok={ev['format_compliance']:.3f}  "
                  f"len={ev['mean_len']:.1f}")
            entry.update({"eval_pass1": ev["pass@1"],
                          "eval_kl": ev["kl_ref"],
                          "eval_format_ok": ev["format_compliance"],
                          "eval_len": ev["mean_len"]})
            out_dir_step = Path(args.out); out_dir_step.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(out_dir_step)
            policy_tok.save_pretrained(out_dir_step)
            (out_dir_step / "log.json").write_text(json.dumps({"log": log, "last_eval": ev}, indent=2))
            print(f"    (saved checkpoint at step {step})")

    ev = eval_pass_at_1(policy, policy_tok, eval_items, device,
                       max_new_tokens=args.max_new_tokens)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    policy_tok.save_pretrained(out_dir)
    (out_dir / "log.json").write_text(json.dumps({"log": log, "final_eval": ev}, indent=2))
    print(f"[final] pass@1={ev['pass@1']:.3f}  "
          f"format_ok={ev['format_compliance']:.3f}  "
          f"kl_ref={ev['kl_ref']:+.4f}  saved to {out_dir}")


if __name__ == "__main__":
    main()
