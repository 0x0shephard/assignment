"""Load the trained reward model as a frozen scorer for PPO/GRPO rollouts."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from model.loading import LoadCfg, DEFAULT_BACKBONE, load_reward_model, freeze


def load_frozen_rm(adapter_dir: str, backbone: str = DEFAULT_BACKBONE,
                   load_in_8bit: bool = False, device: str | torch.device = "cuda"):
    """Reload the SeqCls backbone, attach the trained LoRA adapters, freeze all."""
    base, _tok = load_reward_model(LoadCfg(backbone, load_in_8bit=load_in_8bit, device_map=None))
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = freeze(model).eval().to(device)
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    tok.padding_side = "right"
    return model, tok


@torch.no_grad()
def score_texts(model, tok, texts, device, max_len: int = 1024, batch_size: int = 8):
    """Score a list of full (prompt+response) strings. Returns 1-D tensor."""
    from alignment.rm import score_last_token
    scores = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        r = score_last_token(model, enc["input_ids"], enc["attention_mask"])
        scores.append(r.float().cpu())
    return torch.cat(scores) if scores else torch.tensor([])
