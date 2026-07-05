"""Model + tokenizer loading utilities.

- `load_policy`: SmolLM2/3 in bf16 (trainable via LoRA).
- `load_backbone`: Llama-3.2-1B-Instruct (RM / value backbone).
- `load_reward_model`: AutoModelForSequenceClassification wrapper.
- `disable_ref_grads`: turn off grads on all params (for π_ref via full copy).
- `frozen_ref_context`: contextmanager that disables LoRA adapters so the same
  base model acts as π_ref.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


DEFAULT_POLICY = "HuggingFaceTB/SmolLM2-360M"
DEFAULT_BACKBONE = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass
class LoadCfg:
    name: str
    dtype: torch.dtype = torch.bfloat16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: Optional[str] = "auto"


def _bnb(cfg: LoadCfg) -> Optional[BitsAndBytesConfig]:
    if cfg.load_in_4bit:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=cfg.dtype)
    if cfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _prep_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                        
    return tok


def load_policy(cfg: LoadCfg = LoadCfg(DEFAULT_POLICY)):
    tok = _prep_tokenizer(cfg.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=cfg.dtype,
        device_map=cfg.device_map,
        quantization_config=_bnb(cfg),
    )
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_backbone(cfg: LoadCfg = LoadCfg(DEFAULT_BACKBONE)):
    """Llama-3.2-1B-Instruct as CausalLM (used as value-model backbone)."""
    tok = _prep_tokenizer(cfg.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=cfg.dtype,
        device_map=cfg.device_map,
        quantization_config=_bnb(cfg),
    )
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_reward_model(cfg: LoadCfg = LoadCfg(DEFAULT_BACKBONE), num_labels: int = 1):
    """AutoModelForSequenceClassification on the Llama backbone.

    NOTE: for RM we use *right* padding so the last real token is at the end of
    the sequence; the SC head reads that position.
    """
    tok = AutoTokenizer.from_pretrained(cfg.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.name,
        num_labels=num_labels,
        torch_dtype=cfg.dtype,
        device_map=cfg.device_map,
        quantization_config=_bnb(cfg),
    )
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def param_stats(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": 100 * trainable / max(1, total)}


def vram_footprint_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)


@contextmanager
def frozen_ref(peft_model):
    """Disable LoRA adapters so the wrapped base model acts as π_ref (frozen).
    Use inside torch.no_grad() at call sites."""
    peft_model.disable_adapter_layers()
    try:
        yield peft_model
    finally:
        peft_model.enable_adapter_layers()


if __name__ == "__main__":
    m, t = load_policy()
    print("policy:", param_stats(m), "vram:", vram_footprint_gb(), "GB")
