"""SFT loss: causal-LM cross-entropy on response tokens only.

Labels for prompt tokens are set to -100 by the data pipeline (see
`data/hh_rlhf.SFTDataset`).  The HF CausalLM heads shift-by-one internally,
so we just pass `labels=` and read `outputs.loss`. Perplexity below is the
exp of the same masked cross-entropy.
"""
from __future__ import annotations

import math
import torch


def sft_forward_loss(model, batch, device):
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    return out.loss


@torch.no_grad()
def eval_perplexity(model, loader, device, max_batches: int | None = None) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # Reproduce HF's shift internally so we can weight by real token count.
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tok = (shift_labels != -100).sum().item()
        total_nll += loss.item()
        total_tokens += n_tok
    model.train()
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)
