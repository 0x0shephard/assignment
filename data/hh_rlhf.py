"""HH-RLHF loading + parsing into (prompt, chosen, rejected) triples.

The dataset gives each example a `chosen` and `rejected` string containing the
full dialogue. The two strings share the same prefix (prompt) and diverge on
the final assistant turn. We split on the last `\n\nAssistant:` occurrence
that appears in BOTH fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


@dataclass
class Triple:
    prompt: str
    chosen: str      # response text only (no prompt)
    rejected: str


def _split_prompt_response(chosen: str, rejected: str) -> Optional[Triple]:
    # Longest common prefix
    n = min(len(chosen), len(rejected))
    i = 0
    while i < n and chosen[i] == rejected[i]:
        i += 1
    prefix = chosen[:i]
    # Cut prefix back to the last "\n\nAssistant:" boundary
    marker = "\n\nAssistant:"
    cut = prefix.rfind(marker)
    if cut < 0:
        return None
    prompt = chosen[: cut + len(marker)]
    chosen_resp = chosen[len(prompt):].lstrip()
    rejected_resp = rejected[len(prompt):].lstrip()
    if not chosen_resp or not rejected_resp:
        return None
    return Triple(prompt=prompt, chosen=chosen_resp, rejected=rejected_resp)


def load_hh_triples(split: str = "train", data_dir: str = "harmless-base", limit: Optional[int] = None):
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split=split)
    triples = []
    for ex in ds:
        t = _split_prompt_response(ex["chosen"], ex["rejected"])
        if t is not None:
            triples.append(t)
        if limit is not None and len(triples) >= limit:
            break
    return triples


# ---------- Datasets ----------

class SFTDataset(Dataset):
    """Prompt + chosen response. Labels mask prompt tokens with -100."""
    def __init__(self, triples, tokenizer, max_len: int = 1024):
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        t = self.triples[idx]
        prompt_ids = self.tok(t.prompt, add_special_tokens=False)["input_ids"]
        resp_ids = self.tok(t.chosen, add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            resp_ids = resp_ids + [self.tok.eos_token_id]
        ids = (prompt_ids + resp_ids)[: self.max_len]
        labels = [-100] * min(len(prompt_ids), self.max_len) + resp_ids
        labels = labels[: self.max_len]
        return {"input_ids": ids, "labels": labels}


class PreferenceDataset(Dataset):
    """Raw triples for RM / DPO. Tokenization is done in the collator so we can
    tokenize prompt+chosen and prompt+rejected together."""
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        t = self.triples[idx]
        return {"prompt": t.prompt, "chosen": t.chosen, "rejected": t.rejected}


# ---------- Collators (left-padding for decoder-only) ----------

def _pad(seqs, pad_id, side="left"):
    L = max(len(s) for s in seqs)
    out, mask = [], []
    for s in seqs:
        pad_len = L - len(s)
        if side == "left":
            out.append([pad_id] * pad_len + list(s))
            mask.append([0] * pad_len + [1] * len(s))
        else:
            out.append(list(s) + [pad_id] * pad_len)
            mask.append([1] * len(s) + [0] * pad_len)
    return torch.tensor(out), torch.tensor(mask)


def sft_collator(tokenizer, side="left"):
    pad_id = tokenizer.pad_token_id

    def _c(batch):
        ids, mask = _pad([b["input_ids"] for b in batch], pad_id, side)
        # Pad labels with -100 on the same side
        L = ids.shape[1]
        labels = []
        for b in batch:
            pad_len = L - len(b["labels"])
            if side == "left":
                labels.append([-100] * pad_len + b["labels"])
            else:
                labels.append(b["labels"] + [-100] * pad_len)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": torch.tensor(labels),
        }
    return _c


def preference_collator(tokenizer, max_len: int = 1024, side: str = "left"):
    """For DPO: returns tokenized (prompt+chosen) and (prompt+rejected), plus
    the length of the prompt in tokens so we know where to start summing log-probs.
    For RM training use the same collator; RM tokenizer differs from policy tokenizer."""
    pad_id = tokenizer.pad_token_id

    def _c(batch):
        chosen_ids, rejected_ids, prompt_lens_c, prompt_lens_r = [], [], [], []
        for b in batch:
            p_ids = tokenizer(b["prompt"], add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(b["chosen"], add_special_tokens=False)["input_ids"]
            r_ids = tokenizer(b["rejected"], add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                c_ids = c_ids + [tokenizer.eos_token_id]
                r_ids = r_ids + [tokenizer.eos_token_id]
            full_c = (p_ids + c_ids)[:max_len]
            full_r = (p_ids + r_ids)[:max_len]
            chosen_ids.append(full_c)
            rejected_ids.append(full_r)
            prompt_lens_c.append(min(len(p_ids), len(full_c)))
            prompt_lens_r.append(min(len(p_ids), len(full_r)))

        c_pad, c_mask = _pad(chosen_ids, pad_id, side)
        r_pad, r_mask = _pad(rejected_ids, pad_id, side)
        return {
            "chosen_input_ids": c_pad,
            "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_pad,
            "rejected_attention_mask": r_mask,
            "prompt_len_chosen": torch.tensor(prompt_lens_c),
            "prompt_len_rejected": torch.tensor(prompt_lens_r),
            "pad_side": side,
        }
    return _c


def build_sft_loader(triples, tokenizer, batch_size=8, max_len=1024, shuffle=True):
    ds = SFTDataset(triples, tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=sft_collator(tokenizer, side="left"))


def build_preference_loader(triples, tokenizer, batch_size=8, max_len=1024,
                            shuffle=True, side="left"):
    ds = PreferenceDataset(triples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=preference_collator(tokenizer, max_len=max_len, side=side))


if __name__ == "__main__":
    # Verify: print 3 parsed triples
    triples = load_hh_triples(split="train", limit=3)
    for i, t in enumerate(triples):
        print(f"=== Triple {i} ===")
        print("[PROMPT]", t.prompt[-200:])
        print("[CHOSEN]", t.chosen[:200])
        print("[REJECTED]", t.rejected[:200])
        print()
