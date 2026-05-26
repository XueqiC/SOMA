"""Stage 2: localized LoRA fine-tuning on hard, prompt-mined turns.

We freeze the surrogate base weights and attach LoRA on attention and MLP
projections (Section 3.3, Eq. 4). The training set is built from the
warm-start turns that scored the highest mining loss under the best soft
prompt -- these are the weak-alignment directions where the surrogate disagrees
with the original. Each example is weighted by its hardness r_t.

After this stage, the soft prompts are discarded; the LoRA-adapted surrogate
serves later turns without any prompt-time overhead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


@dataclass
class LoRAFTConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    steps: int = 60
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    max_seq_length: int = 1024
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-3


def attach_lora(model: nn.Module, config: LoRAFTConfig):
    """Attach a fresh LoRA adapter to the surrogate. Existing adapters are kept."""
    if hasattr(model, "peft_config"):
        return model
    lora_cfg = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def _format_pair(context: str, query: str, response: str) -> Tuple[str, str]:
    if context:
        prompt = f"{context}\nUser: {query}\nAssistant:"
    else:
        prompt = f"User: {query}\nAssistant:"
    return prompt, " " + response.strip()


def _build_supervised_batch(
    tokenizer,
    pairs: List[Tuple[str, str, str, float]],
    max_seq_length: int,
    device: torch.device,
):
    """Tokenize (context, query, response, weight) tuples for supervised FT.

    Labels mask out the prompt; loss is per-token NLL on the response only.
    """
    input_ids_list, labels_list, attn_list, weights = [], [], [], []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for context, query, response, w in pairs:
        prompt, target = _format_pair(context, query, response)
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        t_ids = tokenizer(target, add_special_tokens=False).input_ids
        if not t_ids:
            continue
        # Truncate prompt from the left if total exceeds budget.
        budget = max_seq_length - len(t_ids)
        if budget <= 0:
            p_ids = []
            t_ids = t_ids[-max_seq_length:]
        elif len(p_ids) > budget:
            p_ids = p_ids[-budget:]
        full = p_ids + t_ids
        labels = [-100] * len(p_ids) + list(t_ids)
        attn = [1] * len(full)
        # Pad to max_seq_length within batch.
        input_ids_list.append(full)
        labels_list.append(labels)
        attn_list.append(attn)
        weights.append(w)

    if not input_ids_list:
        return None

    max_len = min(max_seq_length, max(len(x) for x in input_ids_list))
    def _pad(arr, val):
        return arr + [val] * (max_len - len(arr))

    input_ids = torch.tensor([_pad(x[:max_len], pad_id) for x in input_ids_list], device=device)
    labels = torch.tensor([_pad(x[:max_len], -100) for x in labels_list], device=device)
    attn = torch.tensor([_pad(x[:max_len], 0) for x in attn_list], device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return input_ids, labels, attn, weights


def localized_lora_finetune(
    model: nn.Module,
    tokenizer,
    training_set: List[Dict[str, object]],
    config: LoRAFTConfig,
) -> Dict[str, float]:
    """Fine-tune attached LoRA adapters on the localized training set.

    Each item in training_set has keys: context, query, response, weight.
    `response` is the original-model response a_t^F at that turn; `weight` is
    omega_t = r_t / sum(r) (Eq. 4 in the paper).

    Returns simple training stats (final loss, steps run).
    """
    if not training_set:
        return {"final_loss": float("nan"), "steps": 0}

    device = next(model.parameters()).device
    model.train()
    # Ensure only LoRA parameters get gradients.
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n.lower())
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return {"final_loss": float("nan"), "steps": 0}

    opt = torch.optim.AdamW(
        trainable, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )

    pairs: List[Tuple[str, str, str, float]] = [
        (
            str(item.get("context", "")),
            str(item.get("query", "")),
            str(item.get("response", "")),
            float(item.get("weight", 1.0)),
        )
        for item in training_set
    ]

    best_loss = float("inf")
    bad = 0
    last_loss = float("nan")
    step = 0
    rng = torch.Generator(device="cpu").manual_seed(0)
    while step < config.steps:
        perm = torch.randperm(len(pairs), generator=rng).tolist()
        for start in range(0, len(perm), config.batch_size):
            if step >= config.steps:
                break
            batch_idx = perm[start : start + config.batch_size]
            batch = [pairs[i] for i in batch_idx]
            built = _build_supervised_batch(
                tokenizer, batch, config.max_seq_length, device
            )
            if built is None:
                continue
            input_ids, labels, attn, weights = built
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1, :].contiguous()
            tgt = labels[:, 1:].contiguous()
            V = logits.size(-1)
            flat_logits = logits.view(-1, V).float()
            flat_tgt = tgt.view(-1)
            mask = flat_tgt.ne(-100)
            if not mask.any():
                continue
            per_tok = F.cross_entropy(
                flat_logits[mask], flat_tgt[mask], reduction="none"
            )
            # Per-example weighting (omega_t).
            with torch.no_grad():
                seq_lens = mask.view(tgt.size(0), -1).sum(dim=1).clamp(min=1)
                tok_w = weights.repeat_interleave(seq_lens).float()
                tok_w = tok_w / tok_w.mean().clamp(min=1e-6)
            loss = (per_tok * tok_w).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            opt.step()

            last_loss = float(loss.detach().item())
            if last_loss < best_loss - config.early_stop_min_delta:
                best_loss = last_loss
                bad = 0
            else:
                bad += 1
            step += 1
            if bad >= config.early_stop_patience:
                break
        if bad >= config.early_stop_patience:
            break

    return {"final_loss": last_loss, "steps": step, "best_loss": best_loss}
