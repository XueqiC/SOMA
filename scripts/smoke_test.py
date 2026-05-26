"""Lightweight import / shape smoke-test that does not require a GPU.

Builds tiny stand-in modules instead of loading real LLMs, then exercises the
soft-prompt, mining, LoRA-FT, and gate code-paths to verify wiring. Useful
for CI and quick sanity checks before scheduling a full run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soma.gate import CosineGate, GateConfig
from soma.prompt_mining import SemanticDivergenceConfig, mine_soft_prompts
from soma.soft_prompt import SoftPrompt, verbalize_soft_prompt


class TinyLM(nn.Module):
    """Minimal causal-LM with the methods SOMA expects."""

    def __init__(self, vocab_size: int = 64, emb_dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.layer = nn.Linear(emb_dim, emb_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_hidden_states=False, use_cache=False):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        h = torch.tanh(self.layer(inputs_embeds))
        logits = self.head(h)
        result = type("Out", (), {})()
        result.logits = logits
        result.hidden_states = (h,)
        return result


class TinyTokenizer:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=64, add_special_tokens=False, padding=False):
        if isinstance(text, str):
            text = [text]
        ids_batch = []
        for t in text:
            ids = [(c % (self.vocab_size - 2)) + 2 for c in (t.encode("utf-8")[:max_length])]
            if not ids:
                ids = [2]
            ids_batch.append(ids)
        max_len = max(len(x) for x in ids_batch)
        padded = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids_batch]
        ids_tensor = torch.tensor(padded, dtype=torch.long)
        return type("Enc", (), {"input_ids": ids_tensor})

    def convert_ids_to_tokens(self, ids):
        return [f"<{int(i)}>" for i in ids]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


def main() -> None:
    print("[smoke] building tiny model + tokenizer")
    model = TinyLM()
    tok = TinyTokenizer()

    sp = SoftPrompt(emb_dim=32, length=4, dtype=torch.float32, device=torch.device("cpu"))
    text, ids = verbalize_soft_prompt(sp, model, tok)
    print(f"[smoke] verbalize -> '{text}' ids={ids}")

    cfg = SemanticDivergenceConfig(
        soft_len=4,
        n_candidates=2,
        iters=3,
        neighborhood_k=4,
        teacher_topm=8,
        max_prompt_tokens=32,
        max_teacher_tokens=16,
    )
    turns = [
        {"context": "", "query": "hi", "teacher": "hello"},
        {"context": "earlier chat", "query": "now what?", "teacher": "we continue."},
    ]
    prompts, losses, hardness = mine_soft_prompts(model, tok, turns, cfg)
    assert len(prompts) == cfg.n_candidates
    assert len(hardness) == len(turns)
    print(f"[smoke] mining losses={losses} hardness={hardness}")

    gate = CosineGate(GateConfig(encoder_max_tokens=32))
    gate.set_warm_start(model, tok, [t["query"] for t in turns])
    accept = gate.acceptance_check(
        ["hello there", "we go on."], ["hi there", "we continue!"], model, tok
    )
    print(f"[smoke] gate accept={accept}")

    print("[smoke] OK")


if __name__ == "__main__":
    main()
