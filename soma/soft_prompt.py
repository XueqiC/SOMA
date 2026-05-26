"""Soft prompt parameter module and nearest-token verbalization.

The soft prompt is a learnable P in R^{L x d} that is prepended to the
surrogate at the embedding layer. For the original (black-box) model, the same
direction is read out by snapping each row of P to its nearest token
embedding (V(P)) so the two streams can be compared under a comparable text
context (Section 3.1 of the paper).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    """Learnable soft prompt prepended at the embedding layer.

    Stored in fp32 for stable optimization and cast to the model dtype on use.
    A tanh squash + scale keeps the prompt energy bounded, which the paper's
    anti-degeneration analysis relies on to keep ||P||_F well-behaved.
    """

    def __init__(
        self,
        emb_dim: int,
        length: int,
        dtype: torch.dtype,
        device: torch.device,
        init_std: float = 0.02,
        scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.length = length
        self.emb_dim = emb_dim
        self.scale = scale
        self._model_dtype = dtype
        self.prompt = nn.Parameter(
            torch.zeros(length, emb_dim, dtype=torch.float32, device=device)
        )
        nn.init.normal_(self.prompt, mean=0.0, std=init_std)

    def squashed(self) -> torch.Tensor:
        return torch.tanh(self.prompt) * self.scale

    def prepend(self, x_emb: torch.Tensor) -> torch.Tensor:
        sp = self.squashed().to(x_emb.dtype)
        sp = sp.unsqueeze(0).expand(x_emb.size(0), -1, -1)
        return torch.cat([sp, x_emb], dim=1)

    def frob_sq(self) -> torch.Tensor:
        return self.prompt.float().pow(2).sum()

    def clone_detached(self) -> "SoftPrompt":
        clone = SoftPrompt(
            emb_dim=self.emb_dim,
            length=self.length,
            dtype=self._model_dtype,
            device=self.prompt.device,
            scale=self.scale,
        )
        with torch.no_grad():
            clone.prompt.copy_(self.prompt.detach())
        return clone


@torch.no_grad()
def verbalize_soft_prompt(
    soft_prompt: SoftPrompt,
    model: nn.Module,
    tokenizer,
) -> Tuple[str, List[int]]:
    """Snap each row of P to its nearest vocab embedding (top-1 cosine).

    Returns (V(P) text, token ids). The text is used as a discrete prompt
    prefix for the black-box original model so the two streams can be probed
    under a comparable dialogue state (Eq. before 3.2 in the paper).
    """
    E = model.get_input_embeddings().weight.detach()  # [V, d]
    P = soft_prompt.squashed().to(E.dtype)  # [L, d]
    P_n = P / (P.norm(dim=1, keepdim=True) + 1e-9)
    E_n = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    sims = P_n @ E_n.T  # [L, V]
    ids = sims.argmax(dim=1).tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    text = tokenizer.convert_tokens_to_string(toks)
    return text.strip(), ids
