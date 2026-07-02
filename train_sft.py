"""Task C2: Supervised fine-tuning (SFT) warm-up on HH-RLHF chosen responses.

Produces:
- checkpoints/sft/       — trainable π_θ^(0) starting point for PPO/DPO/GRPO/RLVR
- checkpoints/sft_ref/   — same weights, marked as π_ref (frozen KL anchor)

At inference we get π_ref simply by calling `disable_adapter_layers()` on the
policy (see `model.loading.frozen_ref`); the on-disk `sft_ref/` copy is just a
convenience so we can reload π_ref standalone.

Usage (local smoke):
    .venv/bin/python train_sft.py --limit 128 --batch_size 2 --max_len 256

Usage (GPU baseline):
    python train_sft.py --epochs 1 --batch_size 8 --grad_accum 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.hh_rlhf import load_hh_triples, build_sft_loader
from model.loading import (
    LoadCfg,
    DEFAULT_POLICY,
    load_policy,
    param_stats,
    vram_footprint_gb,
    frozen_ref,
)
from model.lora import apply_lora_causal
from alignment.sft import sft_forward_loss, eval_perplexity


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_generations(model, tok, prompts, device, max_new_tokens=128):
    model.eval()
    outputs = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text)
    model.train()
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--ref_out", default="checkpoints/sft_ref")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval_limit", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")

    # --- data ---
    print("Loading HH-RLHF triples...")
    train_triples = load_hh_triples("train", limit=args.limit)
    eval_triples = load_hh_triples("test", limit=args.eval_limit)
    print(f"train={len(train_triples)}  eval={len(eval_triples)}")

    # --- model + LoRA ---
    cfg = LoadCfg(args.policy, load_in_8bit=args.load_in_8bit, device_map=None)
    policy, tok = load_policy(cfg)
    policy = apply_lora_causal(policy, grad_ckpt=True)
    policy.to(device)
    print("param stats:", param_stats(policy))
    print("initial vram:", vram_footprint_gb(), "GB")

    train_loader = build_sft_loader(train_triples, tok, batch_size=args.batch_size,
                                    max_len=args.max_len, shuffle=True)
    eval_loader = build_sft_loader(eval_triples, tok, batch_size=args.batch_size,
                                   max_len=args.max_len, shuffle=False)

    # --- optim ---
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr)

    # --- perplexity sanity: prompt-token loss would give suspiciously low PPL
    # (<5 per manual). Log PPL on response tokens only ---
    ppl_init = eval_perplexity(policy, eval_loader, device, max_batches=8)
    print(f"[init] response-token PPL (partial): {ppl_init:.2f}")

    policy.train()
    optim.zero_grad(set_to_none=True)
    step = 0
    accum_loss = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for i, batch in enumerate(pbar):
            loss = sft_forward_loss(policy, batch, device) / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            if (i + 1) % args.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    pbar.set_postfix(step=step, loss=f"{accum_loss:.4f}")
                accum_loss = 0.0
                if step % args.eval_every == 0:
                    ppl = eval_perplexity(policy, eval_loader, device, max_batches=8)
                    print(f"  [step {step}] response-token PPL (partial): {ppl:.2f}")

    # flush any tail-partial-accum
    if accum_loss != 0.0:
        optim.step()
        optim.zero_grad(set_to_none=True)

    # --- final eval ---
    ppl_final = eval_perplexity(policy, eval_loader, device)
    dt = time.time() - t0
    print(f"[final] response-token PPL: {ppl_final:.2f}  ({dt/60:.1f} min)")

    # --- 5 sample generations ---
    sample_prompts = [t.prompt for t in eval_triples[:5]]
    samples = sample_generations(policy, tok, sample_prompts, device)
    for i, (p, s) in enumerate(zip(sample_prompts, samples)):
        print(f"\n--- sample {i} ---")
        print("[PROMPT tail]", p[-120:])
        print("[GEN]", s[:300])

    # --- save π_θ^(0) (LoRA adapters + tokenizer) ---
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    tok.save_pretrained(out)

    # --- save a "π_ref" marker directory. We keep the base weights + a flag;
    # at load time we instantiate the base, attach these adapters, and call
    # disable_adapter_layers() to act as π_ref. Storing the same adapters is
    # what the manual asks for -- π_ref is literally the SFT checkpoint with
    # adapters disabled (see model.loading.frozen_ref). ---
    ref_out = Path(args.ref_out); ref_out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ref_out)
    tok.save_pretrained(ref_out)
    (ref_out / "REF_README.txt").write_text(
        "This directory holds the SFT LoRA adapters that serve as π_ref.\n"
        "Load the base policy, attach these adapters, then call\n"
        "peft_model.disable_adapter_layers() to expose the frozen SFT policy.\n"
    )

    summary = {
        "policy": args.policy,
        "n_train": len(train_triples),
        "n_eval": len(eval_triples),
        "final_response_ppl": ppl_final,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "samples": samples,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nsaved policy to", out)
    print("saved π_ref adapters to", ref_out)


if __name__ == "__main__":
    main()
