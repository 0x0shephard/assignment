"""Task C1: Reward model fine-tuning on HH-RLHF preference pairs.

Usage (local smoke on CPU/mps):
    python train_rm.py --limit 128 --batch_size 2 --max_len 256 --epochs 1

Usage (GPU / Colab / Kaggle):
    python train_rm.py --epochs 1 --batch_size 8 --load_in_8bit

Saves LoRA adapters + tokenizer under checkpoints/rm/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from data.hh_rlhf import load_hh_triples, build_preference_loader
from model.loading import LoadCfg, DEFAULT_BACKBONE, load_reward_model, param_stats, vram_footprint_gb
from model.lora import apply_lora_seqcls
from alignment.rm import score_last_token, rm_loss


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device, max_batches: int | None = None):
    model.eval()
    correct = total = 0
    r_pos_all, r_neg_all = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            r_pos = score_last_token(
                model,
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            r_neg = score_last_token(
                model,
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )
            correct += (r_pos > r_neg).sum().item()
            total += r_pos.size(0)
            r_pos_all.append(r_pos.float().cpu())
            r_neg_all.append(r_neg.float().cpu())
    acc = correct / max(1, total)
    r_pos = torch.cat(r_pos_all) if r_pos_all else torch.tensor([])
    r_neg = torch.cat(r_neg_all) if r_neg_all else torch.tensor([])
    return acc, r_pos, r_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--out", default="checkpoints/rm")
    ap.add_argument("--limit", type=int, default=None, help="cap number of train triples")
    ap.add_argument("--test_limit", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_reg", type=float, default=1e-3)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")

    print("Loading HH-RLHF (harmless-base)...")
    train_triples = load_hh_triples("train", limit=args.limit)
    test_triples = load_hh_triples("test", limit=args.test_limit)
    print(f"train={len(train_triples)}  test={len(test_triples)}")

    print("Loading RM backbone (AutoModelForSequenceClassification)...")
    cfg = LoadCfg(args.backbone,
                  load_in_8bit=args.load_in_8bit,
                  load_in_4bit=args.load_in_4bit,
                  device_map=None)                                             
    model, tok = load_reward_model(cfg)
    model = apply_lora_seqcls(model, grad_ckpt=False)
    model.to(device)
    print("param stats:", param_stats(model))
    print("initial vram:", vram_footprint_gb(), "GB")

    train_loader = build_preference_loader(
        train_triples, tok, batch_size=args.batch_size, max_len=args.max_len, side="right"
    )
    test_loader = build_preference_loader(
        test_triples, tok, batch_size=args.batch_size, max_len=args.max_len,
        shuffle=False, side="right",
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=args.lr)

    acc0, _, _ = evaluate(model, test_loader, device, max_batches=32)
    print(f"[init] preference accuracy (partial eval): {acc0:.3f}")

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            r_pos = score_last_token(
                model,
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            r_neg = score_last_token(
                model,
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )
            out = rm_loss(r_pos, r_neg, lambda_reg=args.lambda_reg)
            out.loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{out.loss.item():.4f}",
                                 acc=f"{out.accuracy.item():.3f}",
                                 rp=f"{out.r_pos.mean().item():.3f}",
                                 rn=f"{out.r_neg.mean().item():.3f}")

    acc, rp, rn = evaluate(model, test_loader, device)
    dt = time.time() - t0
    print(f"[final] test preference accuracy: {acc:.3f}  "
          f"r+ mean={rp.mean().item():.3f}  r- mean={rn.mean().item():.3f}  "
          f"({dt/60:.1f} min)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)                                                  
    tok.save_pretrained(out_dir)
    summary = {
        "backbone": args.backbone,
        "n_train": len(train_triples),
        "n_test": len(test_triples),
        "final_test_pref_accuracy": acc,
        "r_pos_mean": rp.mean().item(),
        "r_neg_mean": rn.mean().item(),
        "r_pos_std": rp.std().item(),
        "r_neg_std": rn.std().item(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_reg": args.lambda_reg,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("saved to", out_dir)
    if acc < 0.60:
        print("WARNING: preference accuracy below 60% target — train longer or "
              "increase LoRA rank / unfreeze more of the backbone.")


if __name__ == "__main__":
    main()
