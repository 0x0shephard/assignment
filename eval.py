"""Task C8: Evaluation harness.

Compares SFT (baseline) against each SFT+{PPO,DPO,GRPO,RLVR} checkpoint on
the HH-RLHF held-out test set (except RLVR, which is math — reported
separately).

Metrics per method (§4.12):
    1. RM win-rate vs SFT: fraction of prompts where aligned RM score > SFT RM score.
    2. KL from π_ref (Monte-Carlo, sampled-token approximation).
    3. Sample response table (5 prompts × N methods, with RM score).
    4. Resource table (peak VRAM + inference time). Training resource stats
       come from each ckpt's saved log; we read them if present.

Usage:
    .venv/bin/python eval.py \
        --sft_dir checkpoints/sft \
        --aligned ppo=checkpoints/ppo dpo=checkpoints/dpo grpo=checkpoints/grpo \
        --rm_dir checkpoints/rm \
        --n_prompts 200 \
        --out eval_results/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from data.hh_rlhf import load_hh_triples
from model.loading import LoadCfg, DEFAULT_POLICY, DEFAULT_BACKBONE, load_policy, vram_footprint_gb, frozen_ref
from model.rm_loader import load_frozen_rm, score_texts
from alignment.rollout import generate_batch
from alignment.policy_utils import logprobs_and_entropy


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_policy_from_ckpt(ckpt_dir: str, policy_name: str, device, load_in_8bit=False):
    from peft import PeftModel
    base, tok = load_policy(LoadCfg(policy_name, load_in_8bit=load_in_8bit, device_map=None))
    model = PeftModel.from_pretrained(base, ckpt_dir, is_trainable=False)
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def generate_all(policy, tok, prompts: List[str], device,
                 max_new_tokens: int = 128, batch_size: int = 8):
    """Greedy generation for each prompt. Returns list of response strings
    (prompt substring stripped) and per-row response-token counts."""
    responses = []
    lens = []
    ids_all = []
    attn_all = []
    resp_mask_all = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        gen = generate_batch(policy, tok, chunk, device,
                             max_new_tokens=max_new_tokens,
                             temperature=0.0, top_p=1.0)
        ids, attn, mask = gen["input_ids"], gen["attention_mask"], gen["response_mask"]
        pl = gen["prompt_len"]
        resp_ids = ids[:, pl:]
        texts = tok.batch_decode(resp_ids, skip_special_tokens=True)
        responses.extend(texts)
        lens.extend(mask[:, pl:].sum(dim=1).cpu().tolist())
        ids_all.append((ids, attn, mask))
    return responses, lens, ids_all


@torch.no_grad()
def compute_kl_from_ref(policy, cached, device):
    """MC KL(π_θ || π_ref) using the sampled tokens.  cached is list of
    (input_ids, attention_mask, response_mask) tuples from generate_all."""
    total = 0.0
    count = 0
    for ids, attn, m in cached:
        lp, _, mv = logprobs_and_entropy(policy, ids.to(device), attn.to(device), m.to(device))
        with frozen_ref(policy):
            lpr, _, _ = logprobs_and_entropy(policy, ids.to(device), attn.to(device), m.to(device))
        kl_row = ((lp - lpr) * mv).sum(dim=1) / mv.sum(dim=1).clamp(min=1)
        total += kl_row.sum().item()
        count += kl_row.size(0)
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--sft_dir", default="checkpoints/sft")
    ap.add_argument("--rm_dir", default="checkpoints/rm")
    ap.add_argument("--aligned", nargs="*", default=[],
                    help="Space-separated name=path entries, e.g. ppo=checkpoints/ppo dpo=...")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- prompts ----
    triples = load_hh_triples("test", limit=args.n_prompts)
    prompts = [t.prompt for t in triples]
    print(f"Loaded {len(prompts)} held-out prompts.")

    # ---- reward model (shared, frozen) ----
    print("Loading reward model...")
    rm, rm_tok = load_frozen_rm(args.rm_dir, args.backbone,
                               load_in_8bit=args.load_in_8bit, device=device)

    aligned_pairs = []
    for a in args.aligned:
        if "=" not in a:
            raise SystemExit(f"bad --aligned entry: {a}")
        name, path = a.split("=", 1)
        aligned_pairs.append((name, path))

    # ---- SFT baseline ----
    print("\n=== SFT baseline ===")
    sft_model, sft_tok = _load_policy_from_ckpt(args.sft_dir, args.policy, device,
                                               load_in_8bit=args.load_in_8bit)
    t0 = time.time()
    sft_responses, sft_lens, sft_cached = generate_all(
        sft_model, sft_tok, prompts, device,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    sft_gen_time = time.time() - t0
    # RM score on full x⊕y
    sft_full = [p + r for p, r in zip(prompts, sft_responses)]
    sft_scores = score_texts(rm, rm_tok, sft_full, device,
                             batch_size=args.batch_size)
    print(f"SFT: RM mean={sft_scores.mean().item():+.3f}  "
          f"mean_len={sum(sft_lens)/max(1,len(sft_lens)):.1f}  "
          f"gen_time={sft_gen_time:.1f}s  peak_vram={vram_footprint_gb():.2f}GB")
    # free
    del sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: Dict[str, dict] = {
        "sft": {
            "rm_scores": sft_scores.tolist(),
            "responses": sft_responses,
            "resp_lens": sft_lens,
            "gen_time_s": sft_gen_time,
        }
    }

    # ---- each aligned checkpoint ----
    for name, path in aligned_pairs:
        print(f"\n=== {name.upper()} @ {path} ===")
        model, tok = _load_policy_from_ckpt(path, args.policy, device,
                                           load_in_8bit=args.load_in_8bit)
        t0 = time.time()
        resp, lens, cached = generate_all(
            model, tok, prompts, device,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        gen_time = time.time() - t0
        full = [p + r for p, r in zip(prompts, resp)]
        scores = score_texts(rm, rm_tok, full, device, batch_size=args.batch_size)
        # KL vs π_ref (LoRA disabled on the aligned model — the ref is the
        # base model, i.e. pre-SFT. If instead you want KL vs SFT, you need
        # to attach the SFT adapters as ref; kept simple here.)
        kl = compute_kl_from_ref(model, cached, device)
        # Win-rate vs SFT
        win = (scores.cpu() > sft_scores.cpu()).float().mean().item()
        print(f"{name}: RM mean={scores.mean().item():+.3f}  win-rate vs SFT={win:.3f}  "
              f"KL vs ref={kl:+.4f}  mean_len={sum(lens)/max(1,len(lens)):.1f}  "
              f"gen_time={gen_time:.1f}s  peak_vram={vram_footprint_gb():.2f}GB")
        results[name] = {
            "rm_scores": scores.tolist(),
            "responses": resp,
            "resp_lens": lens,
            "win_rate_vs_sft": win,
            "kl_vs_ref": kl,
            "gen_time_s": gen_time,
        }
        # attach training log if present
        for candidate in ("log.json", "summary.json"):
            p = Path(path) / candidate
            if p.exists():
                results[name]["training_log_file"] = str(p)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- sample response table (5 prompts × N methods) ----
    n_show = min(5, len(prompts))
    method_order = ["sft"] + [n for n, _ in aligned_pairs]
    sample_rows = []
    for i in range(n_show):
        row = {"prompt": prompts[i][-200:]}
        for m in method_order:
            row[m] = {
                "response": results[m]["responses"][i][:300],
                "rm_score": results[m]["rm_scores"][i],
            }
        sample_rows.append(row)

    # ---- resource table (from generation only; training resource comes from log) ----
    resource_rows = []
    for m in method_order:
        r = results[m]
        rmm = torch.tensor(r["rm_scores"]).mean().item()
        entry = {
            "method": m,
            "rm_mean": rmm,
            "win_rate_vs_sft": r.get("win_rate_vs_sft"),
            "kl_vs_ref": r.get("kl_vs_ref"),
            "mean_resp_len": sum(r["resp_lens"]) / max(1, len(r["resp_lens"])),
            "gen_time_s": r["gen_time_s"],
        }
        resource_rows.append(entry)

    (out_dir / "results.json").write_text(json.dumps({
        "n_prompts": args.n_prompts,
        "resource_table": resource_rows,
        "sample_table": sample_rows,
        "results": results,
    }, indent=2))

    # ---- pretty-print summary ----
    print("\n=== SUMMARY ===")
    cols = ["method", "rm_mean", "win_rate_vs_sft", "kl_vs_ref", "mean_resp_len", "gen_time_s"]
    print("  ".join(f"{c:>18}" for c in cols))
    for r in resource_rows:
        print("  ".join(f"{str(r.get(c, '-'))[:18]:>18}" for c in cols))

    print(f"\nWrote {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
