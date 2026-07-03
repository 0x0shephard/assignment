"""Task C5: GRPO training loop (critic-free).

Pieces:
- Policy π_θ (SmolLM + LoRA), loaded from SFT ckpt.
- Reference π_ref = same base with LoRA disabled (via frozen_ref context).
- Frozen RM r_ψ used to score all K completions per prompt.

Loop:
    for step in 1..200:
        prompts = 8 sampled from HH-RLHF prompt pool
        rollout = collect_group_rollout(policy, rm, prompts, K=4)  # -> B*K rows
        A_{b,k} = r_{b,k} - μ_b                                    # group advantage
        Â = standardize batch-wide over valid tokens
        for _ in 1..epochs_per_batch:
            L = -min(ρ·Â, clip(ρ,1±ε)·Â) + β·KL(π_θ‖π_ref)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from data.hh_rlhf import load_hh_triples
from model.loading import LoadCfg, DEFAULT_POLICY, DEFAULT_BACKBONE, load_policy, param_stats, vram_footprint_gb, frozen_ref
from model.rm_loader import load_frozen_rm, score_texts
from alignment.group_rollout import collect_group_rollout
from alignment.grpo import (
    group_advantages,
    degenerate_group_fraction,
    standardize_batch,
    grpo_step_loss,
)
from alignment.policy_utils import logprobs_and_entropy


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_policy_from_sft(sft_dir: str, policy_name: str, device, load_in_8bit=False,
                          grad_ckpt: bool = False):
    """grad_ckpt disabled by default — needed OFF for the .generate() rollout
    inside GRPO (else HF forces use_cache=False → SDPA shape mismatch)."""
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
def eval_step(policy, policy_tok, rm, rm_tok, prompts, device, max_new_tokens=128):
    from alignment.rollout import generate_batch
    policy.eval()
    gen = generate_batch(policy, policy_tok, prompts, device,
                         max_new_tokens=max_new_tokens,
                         temperature=0.0, top_p=1.0)
    ids, attn, m = gen["input_ids"], gen["attention_mask"], gen["response_mask"]
    lp_new, _, mv = logprobs_and_entropy(policy, ids, attn, m)
    with frozen_ref(policy):
        lp_ref, _, _ = logprobs_and_entropy(policy, ids, attn, m)
    kl = ((lp_new - lp_ref) * mv).sum(dim=1) / mv.sum(dim=1).clamp(min=1)
    texts = policy_tok.batch_decode(ids, skip_special_tokens=True)
    r = score_texts(rm, rm_tok, texts, device, batch_size=len(prompts))
    policy.train()
    return {"rm_mean": r.mean().item(), "kl_mean": kl.mean().item(),
            "resp_len": mv.sum(dim=1).float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--sft_dir", default="checkpoints/sft")
    ap.add_argument("--rm_dir", default="checkpoints/rm")
    ap.add_argument("--out", default="checkpoints/grpo")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--eps_clip", type=float, default=0.2)
    ap.add_argument("--kl_mode", choices=["mc", "full"], default="mc")
    ap.add_argument("--epochs_per_batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_prompts", type=int, default=32)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--prompt_limit", type=int, default=4096)
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")

    # ---- prompt pool ----
    print("Loading prompt pool...")
    train_triples = load_hh_triples("train", limit=args.prompt_limit)
    eval_triples = load_hh_triples("test", limit=args.eval_prompts)
    prompt_pool = [t.prompt for t in train_triples]
    eval_prompts = [t.prompt for t in eval_triples]
    print(f"train prompts={len(prompt_pool)}  eval prompts={len(eval_prompts)}")

    # ---- policy π_θ ----
    print("Loading policy π_θ from SFT ckpt...")
    policy, policy_tok = _load_policy_from_sft(args.sft_dir, args.policy, device,
                                              load_in_8bit=False)
    print("policy trainable:", param_stats(policy))

    # ---- RM ----
    print("Loading frozen reward model...")
    rm, rm_tok = load_frozen_rm(args.rm_dir, args.backbone,
                               load_in_8bit=args.load_in_8bit, device=device)
    print("initial vram:", vram_footprint_gb(), "GB")

    optim = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    log = []
    for step in range(1, args.steps + 1):
        prompts = random.sample(prompt_pool, k=args.prompts_per_step)
        gb = collect_group_rollout(policy, rm, rm_tok, policy_tok, prompts,
                                   device, K=args.K,
                                   max_new_tokens=args.max_new_tokens)

        # advantages (on device for the update)
        r_dev = gb.r_task.to(device)
        gid_dev = gb.group_id.to(device)
        A_raw = group_advantages(r_dev, gid_dev, K=args.K)               # [B*K]

        input_ids = gb.input_ids.to(device)
        attn = gb.attention_mask.to(device)
        resp_mask = gb.response_mask.to(device)
        lp_old = gb.logprobs_old.to(device)

        # standardize batch-wide across VALID tokens: use per-row advantage
        # replicated over valid response tokens, then normalize.
        m_shift = resp_mask[:, 1:].float()
        A_tok = A_raw.unsqueeze(1).expand_as(m_shift) * m_shift
        A_flat = A_tok[m_shift.bool()]
        mu, sd = A_flat.mean(), A_flat.std().clamp(min=1e-8)
        A_std_rows = (A_raw - mu) / sd

        frac_degen = degenerate_group_fraction(gb.r_task, gb.group_id)

        # -- update epochs --
        for _ep in range(args.epochs_per_batch):
            out_pi = policy(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits_new = out_pi.logits
            with torch.no_grad(), frozen_ref(policy):
                logits_ref = policy(input_ids=input_ids, attention_mask=attn,
                                    use_cache=False).logits
            out = grpo_step_loss(
                logits_new=logits_new,
                logits_ref=logits_ref,
                input_ids=input_ids,
                response_mask=resp_mask,
                logprobs_old=lp_old,
                advantages_std=A_std_rows,
                beta=args.beta,
                eps_clip=args.eps_clip,
                kl_mode=args.kl_mode,
                K=args.K,
            )
            optim.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0,
            )
            optim.step()

        entry = {
            "step": step,
            "rm_mean": gb.r_task.mean().item(),
            "policy_loss": out.policy_loss.item(),
            "kl_term": out.kl_term.item(),
            "clip_frac": out.clip_frac.item(),
            "approx_kl": out.approx_kl_ratio.item(),
            "entropy": out.entropy.item(),
            "frac_degen": frac_degen,
            "mean_resp_len": m_shift.sum(dim=1).float().mean().item(),
        }
        log.append(entry)
        print(f"[{step:03d}] rm={entry['rm_mean']:+.3f} "
              f"pol={entry['policy_loss']:+.4f} kl={entry['kl_term']:+.4f} "
              f"clip%={entry['clip_frac']:.3f} degen={entry['frac_degen']:.2f} "
              f"len={entry['mean_resp_len']:.1f}")

        if step % args.eval_every == 0:
            ev = eval_step(policy, policy_tok, rm, rm_tok, eval_prompts, device,
                          max_new_tokens=args.max_new_tokens)
            print(f"    eval@{step}: rm={ev['rm_mean']:+.3f} "
                  f"kl_ref={ev['kl_mean']:+.4f} len={ev['resp_len']:.1f}")
            entry.update({"eval_rm": ev["rm_mean"], "eval_kl": ev["kl_mean"]})
            # Save intermediate checkpoint so long runs survive OOM crashes.
            out_dir_step = Path(args.out); out_dir_step.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(out_dir_step)
            policy_tok.save_pretrained(out_dir_step)
            (out_dir_step / "log.json").write_text(json.dumps(log, indent=2))
            print(f"    (saved checkpoint at step {step})")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    policy_tok.save_pretrained(out_dir)
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
