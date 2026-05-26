"""Model loading and generation helpers.

We keep this thin so the rest of SOMA stays decoupled from any particular
backend. The original model F can be any HuggingFace causal LM; in the paper
F is a large proprietary model and only its text outputs are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LMHandle:
    name: str
    model: torch.nn.Module
    tokenizer: object


def load_lm(
    model_id: str,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> LMHandle:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return LMHandle(name=model_id, model=model, tokenizer=tok)


@torch.no_grad()
def generate_text(
    handle: LMHandle,
    prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    soft_prompt=None,
) -> str:
    """Greedy text generation with an optional embedding-level soft prompt prefix."""
    model = handle.model
    tok = handle.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if soft_prompt is None:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[0][ids.input_ids.shape[-1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()

    embed = model.get_input_embeddings()
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    x = embed(ids).to(dtype)
    x = soft_prompt.prepend(x)
    attn = torch.ones(x.shape[:2], dtype=torch.long, device=device)
    out = model.generate(
        inputs_embeds=x,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    # When using inputs_embeds, only the newly generated tokens are returned.
    return tok.decode(out[0], skip_special_tokens=True).strip()


def format_dialogue_context(history: List[dict]) -> str:
    """Render past (user, assistant) turns into a single context string."""
    lines = []
    for turn in history:
        u = turn.get("user", "")
        a = turn.get("assistant", "")
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)
