"""C0 verification: parse triples, load policy + backbone + RM, wrap with LoRA,
print param counts and VRAM. Confirms the pipeline is wired before Phase 3.

Run:  python verify_c0.py --limit 5 --small
"""
from __future__ import annotations

import argparse
import torch

from data.hh_rlhf import (
    load_hh_triples,
    build_sft_loader,
    build_preference_loader,
)
from model.loading import (
    LoadCfg,
    DEFAULT_POLICY,
    DEFAULT_BACKBONE,
    load_policy,
    load_backbone,
    load_reward_model,
    freeze,
    param_stats,
    vram_footprint_gb,
    frozen_ref,
)
from model.lora import apply_lora_causal, apply_lora_seqcls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    # ---- C0.1 verify parsing ----
    print("Loading HH-RLHF triples...")
    triples = load_hh_triples(split="train", limit=args.limit)
    for i, t in enumerate(triples[:3]):
        print(f"\n=== Triple {i} ===")
        print("[PROMPT tail]", t.prompt[-160:])
        print("[CHOSEN head]", t.chosen[:160])
        print("[REJECTED head]", t.rejected[:160])

    # ---- C0.2 load models ----
    print("\nLoading policy...")
    policy, ptok = load_policy(LoadCfg(args.policy))
    print("policy raw:", param_stats(policy), "vram=", vram_footprint_gb(), "GB")

    print("\nLoading Llama-3.2-1B backbone (frozen ref for value/RM setup checks)...")
    backbone, btok = load_backbone(LoadCfg(args.backbone, load_in_8bit=args.load_in_8bit))
    freeze(backbone)
    print("backbone (frozen):", param_stats(backbone), "vram=", vram_footprint_gb(), "GB")

    print("\nLoading reward model (SeqCls)...")
    rm, rtok = load_reward_model(LoadCfg(args.backbone, load_in_8bit=args.load_in_8bit))
    print("rm raw:", param_stats(rm), "vram=", vram_footprint_gb(), "GB")

    # ---- C0.3 LoRA wrap ----
    print("\nApplying LoRA to policy (r=8, alpha=16, q/v proj)...")
    policy = apply_lora_causal(policy)
    policy.print_trainable_parameters()
    print("policy LoRA:", param_stats(policy))

    print("\nApplying LoRA to reward model...")
    rm = apply_lora_seqcls(rm)
    rm.print_trainable_parameters()

    # ---- frozen ref via disable_adapters ----
    print("\nFrozen-ref sanity: same input under adapter enabled vs disabled...")
    device = next(policy.parameters()).device
    inp = ptok("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        logits_on = policy(**inp).logits[0, -1, :5].float().cpu()
        with frozen_ref(policy):
            logits_off = policy(**inp).logits[0, -1, :5].float().cpu()
    print("adapter ON  logits[:5]:", logits_on.tolist())
    print("adapter OFF logits[:5]:", logits_off.tolist())
    print("max abs diff:", (logits_on - logits_off).abs().max().item(),
          "  (LoRA is zero-init on B, so these should be identical at init)")

    # ---- Dataloaders smoke test ----
    print("\nBuilding SFT + DPO/RM loaders...")
    sft_dl = build_sft_loader(triples, ptok, batch_size=2, max_len=512)
    pref_dl_policy = build_preference_loader(triples, ptok, batch_size=2, max_len=512)
    pref_dl_rm = build_preference_loader(triples, rtok, batch_size=2, max_len=512, side="right")
    b = next(iter(sft_dl))
    print("SFT batch: input_ids", b["input_ids"].shape, "labels non-mask=",
          (b["labels"] != -100).sum().item())
    b = next(iter(pref_dl_policy))
    print("DPO batch: chosen", b["chosen_input_ids"].shape,
          "rejected", b["rejected_input_ids"].shape,
          "prompt_lens_chosen", b["prompt_len_chosen"].tolist())
    b = next(iter(pref_dl_rm))
    print("RM batch (right-pad):", b["chosen_input_ids"].shape)

    print("\nFinal VRAM:", vram_footprint_gb(), "GB")
    print("\nC0 verification complete.")


if __name__ == "__main__":
    main()
