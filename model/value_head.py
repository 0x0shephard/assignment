"""Value model = Llama-3.2-1B backbone + scalar linear head over hidden states.

Per §4.7 C3.1:
- Value head is a single linear layer R^{d_model} -> R.
- Weight init: near-zero (std=0.01) to avoid corrupting early training.
- Backbone can be LoRA-adapted OR frozen with only the head trained
  (`lora_backbone=False` is the safer starting choice).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.loading import LoadCfg, DEFAULT_BACKBONE, load_backbone, freeze
from model.lora import apply_lora_causal


class ValueModel(nn.Module):
    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.01)

    def forward(self, input_ids, attention_mask):
        """Return per-token scalar values V(s_t): shape [B, T]."""
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[-1]                               
        v = self.value_head(h.to(self.value_head.weight.dtype)).squeeze(-1)          
        return v


def build_value_model(backbone_name: str = DEFAULT_BACKBONE,
                      load_in_8bit: bool = False,
                      lora_backbone: bool = False):
    base, tok = load_backbone(LoadCfg(backbone_name, load_in_8bit=load_in_8bit,
                                      device_map=None))
    if lora_backbone:
        base = apply_lora_causal(base)
    else:
        freeze(base)
    hidden_size = base.config.hidden_size
    vm = ValueModel(base, hidden_size)
    vm.value_head.to(torch.float32)
    return vm, tok


def value_trainable_params(vm: ValueModel):
    return [p for p in vm.parameters() if p.requires_grad]
