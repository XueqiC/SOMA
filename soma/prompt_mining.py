"""Stage 1: soft-prompt mining for weak-alignment directions.

Implements the objective from Section 3.2 of the paper:

    J(P) = E_t[ L_sem_exp(P; D_{t-1}, q_t) - beta * H_tail(P; t) ] + lambda_P ||P||_F^2

where L_sem_exp is the expectation-weighted semantic divergence (Eq. 2) and
H_tail is the entropy regularizer (anti-degeneration). The semantic
neighborhood N_k(u) is the top-k tokens by cosine similarity in the surrogate
embedding space. Optimization is AdamW with gradient clipping.

We run M independent candidates and keep the top-K by mining loss; later
stages take the best directions and form a localized training set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .soft_prompt import SoftPrompt


@dataclass
class SemanticDivergenceConfig:
    """Hyperparameters for the prompt-mining objective."""

    soft_len: int = 16
    n_candidates: int = 3
    iters: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    neighborhood_k: int = 32
    teacher_topm: int = 100
    temperature: float = 0.7
    lambda_exp: float = 1.0
    beta_anti_degen: float = 0.05
    lambda_P: float = 1e-4
    init_std: float = 0.02
    scale: float = 0.25
    max_prompt_tokens: int = 768
    max_teacher_tokens: int = 96


def _semantic_neighbors(
    embed_weight: torch.Tensor,
    token_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Top-k nearest tokens (cosine) for each token id. Returns [N, k+1].

    The +1 is the token itself. We use this to define N_k(u) ∪ {u} (Def. 3.1).
    """
    E = embed_weight
    E_n = F.normalize(E.float(), dim=-1)
    q = E_n[token_ids]  # [N, d]
    sims = q @ E_n.T  # [N, V]
    sims.scatter_(1, token_ids.unsqueeze(1), float("inf"))
    _, idx = sims.topk(k + 1, dim=-1)
    return idx  # [N, k+1]


def _build_aligned_inputs(
    tokenizer,
    embed_layer,
    context: str,
    query: str,
    soft_prompt: SoftPrompt,
    teacher_text: str,
    max_prompt_tokens: int,
    max_teacher_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the surrogate forward pass over S_t || a_t^F under prefix P.

    Returns:
        inputs_embeds: [1, L+T_S+T_y, d] -- prepended soft prompt + S_t + teacher answer
        labels: [1, L+T_S+T_y]            -- -100 except over teacher tokens
        teacher_token_ids: [T_y]
    """
    if context:
        full_context = f"{context}\nUser: {query}\nAssistant:"
    else:
        full_context = f"User: {query}\nAssistant:"

    ctx_ids = tokenizer(
        full_context,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    ).input_ids.to(device)

    y_ids = tokenizer(
        " " + teacher_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_teacher_tokens,
        add_special_tokens=False,
    ).input_ids.to(device)

    x_ctx = embed_layer(ctx_ids).to(dtype)
    x_with_sp = soft_prompt.prepend(x_ctx)  # [1, L+T_S, d]
    x_y = embed_layer(y_ids).to(dtype)  # [1, T_y, d]
    x_all = torch.cat([x_with_sp, x_y], dim=1)

    prefix_len = x_with_sp.size(1)
    labels = torch.full(
        (1, x_all.size(1)), -100, dtype=torch.long, device=device
    )
    labels[:, prefix_len:] = y_ids
    return x_all, labels, y_ids[0]


def _expectation_weighted_divergence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    neighbors: torch.Tensor,
    temperature: float,
    lambda_exp: float,
    teacher_topm: int,
) -> torch.Tensor:
    """Eq. (2): expectation-weighted semantic divergence loss.

    Args:
        logits: [1, L+T_S+T_y, V] surrogate logits
        labels: [1, L+T_S+T_y] -100 except over teacher tokens
        teacher_ids: [T_y] teacher token ids (in surrogate vocab)
        embed_weight: [V, d] surrogate input embedding matrix
        neighbors: [T_y, k+1] precomputed semantic neighborhood indices per teacher token
    """
    # Slice positions that predict teacher tokens (shift-by-one CLM).
    # logits[:, t-1] predicts labels[:, t].
    label_mask = labels.ne(-100)  # [1, S]
    pred_positions = label_mask[0].nonzero(as_tuple=True)[0] - 1
    pred_positions = pred_positions[pred_positions >= 0]
    if pred_positions.numel() == 0:
        return logits.new_zeros(())

    teacher_used = teacher_ids[: pred_positions.numel()]
    nbrs_used = neighbors[: pred_positions.numel()]  # [T_eff, k+1]

    log_probs = F.log_softmax(logits[0, pred_positions].float(), dim=-1)  # [T_eff, V]
    probs = log_probs.exp()

    # Cheap top-m expectation (Section 3.2): keeps cost linear in m, not |V|.
    top_vals, top_idx = probs.topk(teacher_topm, dim=-1)  # [T_eff, m]
    top_emb = embed_weight[top_idx]  # [T_eff, m, d]
    exp_emb = (top_vals.unsqueeze(-1).to(top_emb.dtype) * top_emb).sum(dim=1)  # [T_eff, d]
    teacher_emb = embed_weight[teacher_used]  # [T_eff, d]
    cos_exp = F.cosine_similarity(exp_emb.float(), teacher_emb.float(), dim=-1).clamp(0.0, 1.0)
    w_exp = 1.0 + lambda_exp * cos_exp  # [T_eff]

    # Token-level semantic divergence: unlikelihood over teacher token + neighbors.
    # s_tau(v | y) ∝ exp(cos(e_v, e_y) / tau), normalized over the neighborhood.
    teacher_e_norm = F.normalize(teacher_emb.float(), dim=-1)  # [T_eff, d]
    nbr_emb_norm = F.normalize(embed_weight[nbrs_used].float(), dim=-1)  # [T_eff, k+1, d]
    nbr_cos = (nbr_emb_norm * teacher_e_norm.unsqueeze(1)).sum(-1)  # [T_eff, k+1]
    s_tau = F.softmax(nbr_cos / max(temperature, 1e-3), dim=-1)  # [T_eff, k+1]

    # Gather log(1 - p_v) for v in neighborhood.
    # Use log(1 - p) via numerically stable form using log_probs.
    nbr_log_probs = torch.gather(log_probs, 1, nbrs_used)  # [T_eff, k+1]
    one_minus_p = (1.0 - nbr_log_probs.exp()).clamp(min=1e-6)
    log_one_minus = one_minus_p.log()
    per_pos = -(s_tau * log_one_minus).sum(-1)  # [T_eff]
    loss = (w_exp * per_pos).mean()
    return loss


def _tail_entropy(
    logits: torch.Tensor,
    tail_size: int = 32,
) -> torch.Tensor:
    """H_tail entropy regularizer on the last `tail_size` positions (Section 3.2)."""
    tail = logits[0, -min(tail_size, logits.size(1)):, :].float()
    log_p = F.log_softmax(tail, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1).mean()


def mine_soft_prompts(
    model,
    tokenizer,
    warm_start_turns: List[Dict[str, str]],
    config: SemanticDivergenceConfig,
    on_step: Optional[Callable[[int, float, int], None]] = None,
) -> Tuple[List[SoftPrompt], List[float], List[float]]:
    """Mine n_candidates soft prompts and keep all by best mining loss.

    Each warm-start turn is a dict with keys: "context", "query", "teacher".
    "teacher" is the original-model response a_t^F at that turn.

    Returns:
        prompts: list of SoftPrompt (best-loss first)
        losses:  list of final mining loss per candidate (lower = better)
        hardness: per-turn hardness score r_t = max_P L_sem_exp under the best P
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach()
    emb_dim = embed_weight.size(1)

    # Precompute teacher token ids + semantic neighbors per turn so we don't
    # redo it for every candidate / iteration.
    turn_cache: List[Dict] = []
    for turn in warm_start_turns:
        teacher = turn.get("teacher", "")
        if not teacher.strip():
            continue
        y_ids = tokenizer(
            " " + teacher,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_teacher_tokens,
            add_special_tokens=False,
        ).input_ids[0].to(device)
        if y_ids.numel() == 0:
            continue
        nbrs = _semantic_neighbors(embed_weight, y_ids, config.neighborhood_k)
        turn_cache.append({
            "context": turn.get("context", ""),
            "query": turn.get("query", ""),
            "teacher": teacher,
            "teacher_ids": y_ids,
            "neighbors": nbrs,
        })

    if not turn_cache:
        return [], [], []

    best_candidates: List[Tuple[float, SoftPrompt]] = []
    per_turn_max = [0.0 for _ in turn_cache]

    for cand_idx in range(config.n_candidates):
        sp = SoftPrompt(
            emb_dim=emb_dim,
            length=config.soft_len,
            dtype=dtype,
            device=device,
            init_std=config.init_std,
            scale=config.scale,
        )
        opt = torch.optim.AdamW(
            sp.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        last_loss = float("nan")
        last_per_turn: List[float] = [0.0 for _ in turn_cache]

        for it in range(config.iters):
            losses = []
            per_turn_now: List[float] = []
            for ti, turn in enumerate(turn_cache):
                x_all, labels, _ = _build_aligned_inputs(
                    tokenizer=tokenizer,
                    embed_layer=embed_layer,
                    context=turn["context"],
                    query=turn["query"],
                    soft_prompt=sp,
                    teacher_text=turn["teacher"],
                    max_prompt_tokens=config.max_prompt_tokens,
                    max_teacher_tokens=config.max_teacher_tokens,
                    device=device,
                    dtype=dtype,
                )
                out = model(inputs_embeds=x_all, use_cache=False)
                logits = out.logits
                sem = _expectation_weighted_divergence_loss(
                    logits=logits,
                    labels=labels,
                    teacher_ids=turn["teacher_ids"],
                    embed_weight=embed_weight,
                    neighbors=turn["neighbors"],
                    temperature=config.temperature,
                    lambda_exp=config.lambda_exp,
                    teacher_topm=config.teacher_topm,
                )
                ent = _tail_entropy(logits)
                obj = sem - config.beta_anti_degen * ent
                losses.append(obj)
                per_turn_now.append(sem.detach().item())

            total = torch.stack(losses).mean() + config.lambda_P * sp.frob_sq()
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(sp.parameters(), config.grad_clip)
            opt.step()

            last_loss = float(total.detach().item())
            last_per_turn = per_turn_now
            if on_step is not None:
                on_step(cand_idx, last_loss, it)

        best_candidates.append((last_loss, sp))
        for i, r in enumerate(last_per_turn):
            if r > per_turn_max[i]:
                per_turn_max[i] = r

    best_candidates.sort(key=lambda x: x[0])
    prompts = [sp for _, sp in best_candidates]
    losses = [loss for loss, _ in best_candidates]
    return prompts, losses, per_turn_max
