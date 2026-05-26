"""Stage 3: cosine-similarity gate for switching and drift-aware rollback.

Implements the warm-start switching rule (Section 3.3 / Section 4.1):

  Gap(S) = 1 - cos(phi(a_F), phi(a_G))         in [0, 1]

The gate accepts switching to G when (a) the mean Gap over an acceptance batch
falls below an output-fidelity threshold AND (b) the current query stays close
to the warm-start centroid (locality gate). After switching, we keep the
locality and Gap checks alive on the served turns and roll back to F if the
recent moving average drifts above the threshold (Corollary 4.3).

To keep the implementation portable and not require an extra dependency, we
use the surrogate's own hidden states (mean-pooled last layer) as the encoder
phi(.). This is a faithful instantiation of the bounded discrepancy score in
the paper -- any sentence encoder works in its place.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch
import torch.nn.functional as F


@dataclass
class GateConfig:
    accept_threshold: float = 0.20      # eps in Cor. 4.3 (max average Gap to accept)
    locality_threshold: float = 0.35    # max cosine distance from warm-start centroid
    drift_window: int = 4               # moving window for post-switch drift check
    drift_threshold: float = 0.30
    consecutive_to_rollback: int = 2    # m_cons in Section 3.3
    encoder_max_tokens: int = 256


@torch.no_grad()
def _encode(text: str, model, tokenizer, max_tokens: int) -> torch.Tensor:
    """Mean-pooled last-hidden-state of `model` over `text` (fp32, on CPU)."""
    if not text.strip():
        return torch.zeros(model.get_input_embeddings().embedding_dim, dtype=torch.float32)
    device = next(model.parameters()).device
    ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
    ).input_ids.to(device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1][0].float()  # [T, d]
    return h.mean(dim=0).cpu()


def _gap(a_F: str, a_G: str, model, tokenizer, max_tokens: int) -> float:
    fF = _encode(a_F, model, tokenizer, max_tokens)
    fG = _encode(a_G, model, tokenizer, max_tokens)
    cos = float(F.cosine_similarity(fF.unsqueeze(0), fG.unsqueeze(0)).item())
    return float(1.0 - cos)


class CosineGate:
    """Bounded discrepancy gate with locality and drift-aware rollback."""

    def __init__(self, config: GateConfig) -> None:
        self.config = config
        self.centroid: Optional[torch.Tensor] = None
        self.recent_gaps: Deque[float] = deque(maxlen=config.drift_window)
        self.consecutive_drift: int = 0
        self.served_count: int = 0

    def set_warm_start(self, model, tokenizer, queries):
        """Compute the warm-start centroid in encoder space."""
        embeddings = []
        for q in queries:
            embeddings.append(_encode(q, model, tokenizer, self.config.encoder_max_tokens))
        if embeddings:
            self.centroid = torch.stack(embeddings).mean(dim=0)

    def acceptance_check(self, original_responses, surrogate_responses, model, tokenizer) -> dict:
        """Decide whether to switch by averaging Gap over an acceptance batch."""
        if not original_responses:
            return {"accept": False, "mean_gap": float("nan"), "details": []}
        gaps = []
        for aF, aG in zip(original_responses, surrogate_responses):
            gaps.append(_gap(aF, aG, model, tokenizer, self.config.encoder_max_tokens))
        mean_gap = sum(gaps) / len(gaps)
        accept = mean_gap <= self.config.accept_threshold
        return {"accept": accept, "mean_gap": mean_gap, "details": gaps}

    def locality_ok(self, query: str, model, tokenizer) -> tuple:
        """Locality gate: is the current query close to the warm-start centroid?"""
        if self.centroid is None:
            return True, 0.0
        q_emb = _encode(query, model, tokenizer, self.config.encoder_max_tokens)
        cos = float(F.cosine_similarity(q_emb.unsqueeze(0), self.centroid.unsqueeze(0)).item())
        distance = 1.0 - cos
        return distance <= self.config.locality_threshold, distance

    def observe_post_switch(self, original_response, surrogate_response, model, tokenizer) -> dict:
        """Update drift state. Call when both responses are available (drift check
        sample). For inference-only operation, the user can periodically request a
        teacher response on, say, every k-th turn or whenever the locality gate
        is borderline.
        """
        gap = _gap(
            original_response, surrogate_response, model, tokenizer, self.config.encoder_max_tokens
        )
        self.recent_gaps.append(gap)
        self.served_count += 1
        moving = sum(self.recent_gaps) / len(self.recent_gaps)
        if moving > self.config.drift_threshold:
            self.consecutive_drift += 1
        else:
            self.consecutive_drift = max(0, self.consecutive_drift - 1)
        should_rollback = self.consecutive_drift >= self.config.consecutive_to_rollback
        return {"gap": gap, "moving": moving, "rollback": should_rollback}

    def reset_after_rollback(self) -> None:
        self.recent_gaps.clear()
        self.consecutive_drift = 0
