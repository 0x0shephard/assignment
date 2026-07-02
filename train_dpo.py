"""Task C4: DPO training loop.

Offline, reward-model-free alignment via Eq. (10). Uses the frozen SFT
policy as π_ref (LoRA adapters disabled) and a LoRA-trainable copy of the
same base as π_θ.

Init sanity (§4.8 pitfalls): before any updates, Δ_θ ≈ Δ_ref → z ≈ 0 →
σ(z) ≈ 0.5, so pref accuracy on the eval set should be ~50%. After 100
steps it should be above ~55–60%.

Usage:
    python train_dpo.py --limit 4096 --epochs 1 --batch_size 8 --grad_accum 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.hh_rlhf import load_hh_triples, build_preference_loader
from model.loading import LoadCfg, DEFAULT_POLICY, load_policy, param_stats, vram_footprint_gb, frozen_ref
from alignment.dpo import sequence_log_prob, dpo_loss


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_policy_from_sft(sft_dir: str, policy_name: str, device, load_in_8bit=False,
                          grad_ckpt: bool = False):
    """DPO doesn't call .generate() during training (only teacher-forced
    forwards), so grad_ckpt is safe here — but disabled by default to match
    PPO/GRPO/RLVR behavior and keep speed high on small policies."""
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
def eval_pref_accuracy(policy, loader, device, max_batches: int | None = None):
    policy.eval()
    correct = total = 0
    z_sum = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        lp_c = sequence_log_prob(
            policy,
            batch["chosen_input_ids"].to(device),
            batch["chosen_attention_mask"].to(device),
            batch["prompt_len_chosen"],
        )
        lp_r = sequence_log_prob(
            policy,
            batch["rejected_input_ids"].to(device),
            batch["rejected_attention_mask"].to(device),
            batch["prompt_len_rejected"],
        )
        correct += (lp_c > lp_r).sum().item()
        total += lp_c.size(0)
        z_sum += (lp_c - lp_r).sum().item()
    policy.train()
    return correct / max(1, total), z_sum / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--sft_dir", default="checkpoints/sft")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval_limit", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")

    # ---- data ----
    print("Loading HH-RLHF preference pairs...")
    train_triples = load_hh_triples("train", limit=args.limit)
    eval_triples = load_hh_triples("test", limit=args.eval_limit)
    print(f"train={len(train_triples)}  eval={len(eval_triples)}")

    # ---- policy π_θ from SFT ckpt (π_ref = same, LoRA disabled) ----
    print("Loading π_θ from SFT ckpt...")
    policy, tok = _load_policy_from_sft(args.sft_dir, args.policy, device,
                                        load_in_8bit=args.load_in_8bit)
    print("param stats:", param_stats(policy), "vram:", vram_footprint_gb(), "GB")

    train_loader = build_preference_loader(
        train_triples, tok, batch_size=args.batch_size, max_len=args.max_len,
        shuffle=True, side="left",
    )
    eval_loader = build_preference_loader(
        eval_triples, tok, batch_size=args.batch_size, max_len=args.max_len,
        shuffle=False, side="left",
    )

    optim = AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    # ---- init sanity: should be ~50% (Δ_θ = Δ_ref at start) ----
    acc0, z0 = eval_pref_accuracy(policy, eval_loader, device, max_batches=16)
    print(f"[init] pref accuracy (partial): {acc0:.3f}   mean lp_c - lp_r: {z0:+.3f}")

    policy.train()
    optim.zero_grad(set_to_none=True)
    step = 0
    t0 = time.time()
    log = []
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for i, batch in enumerate(pbar):
            # π_θ log-probs (trainable)
            lp_pi_c = sequence_log_prob(
                policy,
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
                batch["prompt_len_chosen"],
            )
            lp_pi_r = sequence_log_prob(
                policy,
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
                batch["prompt_len_rejected"],
            )
            # π_ref log-probs (frozen — adapters disabled)
            with torch.no_grad(), frozen_ref(policy):
                lp_ref_c = sequence_log_prob(
                    policy,
                    batch["chosen_input_ids"].to(device),
                    batch["chosen_attention_mask"].to(device),
                    batch["prompt_len_chosen"],
                )
                lp_ref_r = sequence_log_prob(
                    policy,
                    batch["rejected_input_ids"].to(device),
                    batch["rejected_attention_mask"].to(device),
                    batch["prompt_len_rejected"],
                )
            out = dpo_loss(lp_pi_c, lp_pi_r, lp_ref_c, lp_ref_r, beta=args.beta)
            (out.loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0,
                )
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    pbar.set_postfix(
                        step=step,
                        loss=f"{out.loss.item():.4f}",
                        acc=f"{out.pref_accuracy.item():.3f}",
                        z=f"{out.reward_margin_z.mean().item():+.3f}",
                    )
                if step % args.eval_every == 0:
                    acc, z = eval_pref_accuracy(policy, eval_loader, device, max_batches=16)
                    entry = {"step": step, "eval_pref_acc": acc, "eval_z": z,
                             "loss": out.loss.item()}
                    log.append(entry)
                    print(f"  [step {step}] eval pref acc={acc:.3f}   mean z={z:+.3f}")

    # ---- final ----
    acc_final, z_final = eval_pref_accuracy(policy, eval_loader, device)
    dt = time.time() - t0
    print(f"[final] eval pref acc={acc_final:.3f}   mean z={z_final:+.3f}   ({dt/60:.1f} min)")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    summary = {
        "policy": args.policy,
        "sft_dir": args.sft_dir,
        "n_train": len(train_triples),
        "n_eval": len(eval_triples),
        "beta": args.beta,
        "lr": args.lr,
        "init_pref_acc": acc0,
        "final_pref_acc": acc_final,
        "log": log,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
