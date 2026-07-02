"""Task C3: PPO training loop.

Pieces (see manual §4.7):
- Policy π_θ (SmolLM + LoRA, trainable), loaded from SFT ckpt.
- Reference π_ref = same base + adapters disabled (via frozen_ref context).
- Value model V_φ = Llama-3.2-1B backbone + scalar head (backbone frozen).
- Reward model r_ψ = frozen Llama-3.2-1B + trained LoRA adapters + score head.

Loop:
    for step in 1..200:
        prompts = sample 8 from prompt pool
        rollout = collect_rollout(policy, ref, V_φ, RM, prompts)     # π_old
        rewards = r_task + r_KL                                       (§4.7 C3.2)
        adv, ret = GAE(rewards, V_old)                                (§4.7 C3.3)
        for k in 1..4 mini-batch epochs:                              (§4.7 C3.4)
            L = L_clip + c_v L_V - c_ent H(π_θ)
            step optimizer

Every 25 steps: RM mean score + KL from π_ref on a held-out prompt set.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.hh_rlhf import load_hh_triples
from model.loading import LoadCfg, DEFAULT_POLICY, DEFAULT_BACKBONE, load_policy, param_stats, vram_footprint_gb
from model.lora import apply_lora_causal
from model.value_head import build_value_model, value_trainable_params
from model.rm_loader import load_frozen_rm, score_texts
from alignment.rollout import collect_rollout
from alignment.ppo import (
    build_per_step_rewards,
    compute_gae,
    standardize_advantages,
    ppo_step_loss,
)
from alignment.policy_utils import logprobs_and_entropy


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")   # PPO on MPS is fragile; disable by default


def _load_policy_from_sft(sft_dir: str, policy_name: str, device, load_in_8bit=False):
    """Reload base policy and attach the SFT LoRA adapters as the starting point."""
    from peft import PeftModel
    base, tok = load_policy(LoadCfg(policy_name, load_in_8bit=load_in_8bit, device_map=None))
    model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
    # gradient checkpointing + LoRA
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(device)
    return model, tok


@torch.no_grad()
def evaluate(policy, policy_tok, rm, rm_tok, ref_model_call, eval_prompts, device,
             max_new_tokens=128):
    """Return (RM mean score, KL_from_ref (MC), mean response length)."""
    from alignment.rollout import generate_batch
    policy.eval()
    gen = generate_batch(policy, policy_tok, eval_prompts, device,
                         max_new_tokens=max_new_tokens,
                         temperature=0.0, top_p=1.0)  # greedy for eval
    input_ids, attn, resp_mask = gen["input_ids"], gen["attention_mask"], gen["response_mask"]
    # KL vs π_ref via sampled-token approximation
    lp_pol, _, m = logprobs_and_entropy(policy, input_ids, attn, resp_mask)
    with ref_model_call():
        lp_ref, _, _ = logprobs_and_entropy(policy, input_ids, attn, resp_mask)
    kl = ((lp_pol - lp_ref) * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    texts = policy_tok.batch_decode(input_ids, skip_special_tokens=True)
    r_task = score_texts(rm, rm_tok, texts, device, batch_size=len(eval_prompts))
    resp_lens = m.sum(dim=1).float().cpu()
    policy.train()
    return {"rm_mean": r_task.mean().item(), "kl_mean": kl.mean().item(),
            "resp_len": resp_lens.mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--sft_dir", default="checkpoints/sft")
    ap.add_argument("--rm_dir", default="checkpoints/rm")
    ap.add_argument("--out", default="checkpoints/ppo")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.1)          # KL coefficient
    ap.add_argument("--eps_clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--epochs_per_batch", type=int, default=4)
    ap.add_argument("--policy_lr", type=float, default=1e-5)
    ap.add_argument("--value_lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_prompts", type=int, default=32)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--prompt_limit", type=int, default=4096)
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")

    # ---- prompt pool from HH-RLHF ----
    print("Loading prompt pool...")
    train_triples = load_hh_triples("train", limit=args.prompt_limit)
    eval_triples = load_hh_triples("test", limit=args.eval_prompts)
    prompt_pool = [t.prompt for t in train_triples]
    eval_prompts = [t.prompt for t in eval_triples]
    print(f"train prompts={len(prompt_pool)}  eval prompts={len(eval_prompts)}")

    # ---- policy π_θ from SFT ckpt ----
    print("Loading policy π_θ from SFT ckpt...")
    policy, policy_tok = _load_policy_from_sft(args.sft_dir, args.policy, device,
                                               load_in_8bit=False)
    print("policy trainable:", param_stats(policy))

    # ---- reference π_ref: use disable_adapter_layers context ----
    from model.loading import frozen_ref
    def ref_ctx():
        return frozen_ref(policy)

    # ---- value model ----
    print("Loading value backbone (Llama-3.2-1B, frozen except head)...")
    value_model, _vtok = build_value_model(args.backbone, load_in_8bit=args.load_in_8bit,
                                          lora_backbone=False)
    value_model.to(device)
    print("value trainable params:", sum(p.numel() for p in value_trainable_params(value_model)))

    # ---- reward model ----
    print("Loading frozen reward model...")
    rm, rm_tok = load_frozen_rm(args.rm_dir, args.backbone,
                                load_in_8bit=args.load_in_8bit, device=device)

    print("initial vram:", vram_footprint_gb(), "GB")

    # ---- optimizers (separate: policy LoRA vs value head) ----
    optim_pol = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.policy_lr)
    optim_val = AdamW(value_trainable_params(value_model), lr=args.value_lr)

    log = []
    for step in range(1, args.steps + 1):
        # 1) sample prompts
        batch_prompts = random.sample(prompt_pool, k=args.prompts_per_step)
        # 2) rollout
        r = collect_rollout(policy, None, value_model, rm, rm_tok, policy_tok,
                           batch_prompts, device,
                           max_new_tokens=args.max_new_tokens)
        r.to(device)

        # 3) build rewards + GAE (over the response region only)
        # shift response_mask to align with the shifted grid used in ppo.rollout
        resp_mask_shift = r.response_mask[:, 1:].float()
        # slice the response region matching lp_old / lp_ref / v_old that were
        # stored aligned to same shifted grid but only kept from prompt_len-1
        # Re-derive resp region mask consistently.
        resp_start = r.prompt_len - 1
        resp_mask_r = resp_mask_shift[:, resp_start:]
        rewards = build_per_step_rewards(
            r.r_task, resp_mask_r, r.logprobs_old, r.logprobs_ref, beta=args.beta,
        )
        adv, ret = compute_gae(rewards, r.values_old, resp_mask_r,
                              gamma=args.gamma, lam=args.lam)
        adv_std = standardize_advantages(adv, resp_mask_r)

        # 4) PPO update: 4 mini-batch epochs over the rollout
        for _epoch in range(args.epochs_per_batch):
            # single mini-batch = the full rollout batch (small B)
            lp_new_full, ent_full, m_full = logprobs_and_entropy(
                policy, r.input_ids, r.attention_mask, r.response_mask,
            )
            v_new_full = value_model(input_ids=r.input_ids, attention_mask=r.attention_mask)
            # slice to response region under the shifted grid
            lp_new_r = lp_new_full[:, resp_start:]
            ent_r = ent_full[:, resp_start:]
            v_new_r = v_new_full[:, :-1][:, resp_start:]

            out = ppo_step_loss(
                lp_new_r, r.logprobs_old, adv_std, ret, v_new_r, resp_mask_r,
                ent_r, eps_clip=args.eps_clip, c_v=0.5, c_entropy=0.0,
            )
            optim_pol.zero_grad(set_to_none=True)
            optim_val.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0,
            )
            torch.nn.utils.clip_grad_norm_(value_trainable_params(value_model), 1.0)
            optim_pol.step()
            optim_val.step()

        entry = {
            "step": step,
            "r_task_mean": r.r_task.mean().item(),
            "kl_shaping_mean": ((r.logprobs_old - r.logprobs_ref) * resp_mask_r).sum().item()
                              / resp_mask_r.sum().clamp(min=1).item(),
            "policy_loss": out.policy_loss.item(),
            "value_loss": out.value_loss.item(),
            "approx_kl": out.approx_kl.item(),
            "clip_frac": out.clip_frac.item(),
            "mean_resp_len": resp_mask_r.sum(dim=1).float().mean().item(),
        }
        log.append(entry)
        print(f"[{step:03d}] rm={entry['r_task_mean']:+.3f} "
              f"kl_shape={entry['kl_shaping_mean']:+.3f} "
              f"pol={entry['policy_loss']:+.4f} val={entry['value_loss']:.4f} "
              f"clip%={entry['clip_frac']:.3f} akl={entry['approx_kl']:+.4f} "
              f"len={entry['mean_resp_len']:.1f}")

        if step % args.eval_every == 0:
            ev = evaluate(policy, policy_tok, rm, rm_tok, ref_ctx, eval_prompts, device,
                         max_new_tokens=args.max_new_tokens)
            print(f"    eval@{step}: rm={ev['rm_mean']:+.3f} "
                  f"kl_ref={ev['kl_mean']:+.4f} len={ev['resp_len']:.1f}")
            entry.update({"eval_rm": ev["rm_mean"], "eval_kl": ev["kl_mean"]})

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    policy_tok.save_pretrained(out_dir)
    torch.save(value_model.value_head.state_dict(), out_dir / "value_head.pt")
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
