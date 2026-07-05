"""LoRA/PEFT setup for policy, reward model, and value backbone."""
from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model


def apply_lora_causal(model, r: int = 8, alpha: int = 16, dropout: float = 0.05,
                     target_modules=("q_proj", "v_proj"),
                     grad_ckpt: bool = True):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return get_peft_model(model, cfg)


def apply_lora_seqcls(model, r: int = 8, alpha: int = 16, dropout: float = 0.05,
                      target_modules=("q_proj", "v_proj"),
                      grad_ckpt: bool = True):
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        modules_to_save=["score"],                                          
    )
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return get_peft_model(model, cfg)
