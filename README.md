# SOMA: Efficient Multi-turn LLM Serving via Small Language Model

Reference implementation of **SOMA** (**S**oft-prompts for l**O**cal **M**anifold **A**pproximation), a three-stage framework for serving multi-turn dialogues with a small surrogate language model adapted to the *session-local* behavior of a large model.

> Multi-turn LLM serving typically re-feeds the full dialogue history at every turn, which is expensive in latency, memory, and API cost. SOMA exploits a structural property of real-world dialogues — *early turns carry most of the context, while later turns are short and locally dependent* — and asks: can we use a large model **only** for the early turns and then hand the remainder to a small model that has been adapted to the session's local response manifold? SOMA answers yes, by (i) mining soft prompts that expose where the small model disagrees with the large one, (ii) distilling those hard cases into a lightweight LoRA adapter, and (iii) switching to the adapted surrogate only when a cosine gate confirms local alignment, with automatic rollback on drift.

---

## Method Overview

SOMA operates per dialogue session and has three stages, mirroring the paper:

1. **Soft-prompt mining (Section 3.2).** During a short *warm-start* window of the first $W$ turns the large model $F$ serves the user, and we collect $(\text{context}, \text{query}, a_t^F)$ triples. We then learn a soft prompt $\mathbf{P} \in \mathbb{R}^{L\times d}$ on the surrogate $G$ that maximizes an expectation-weighted *semantic divergence* loss (Eq. 2):
   - an unlikelihood-style penalty over the teacher token and its $k$-nearest neighbors in $G$'s embedding space (Definition 3.1);
   - an expectation weight $w_{t,i}(\mathbf{P}) = 1 + \lambda_\text{exp}\,\mathrm{clip}\bigl(\cos(\bar e_{t,i}, e_{y_{t,i}^F}), 0, 1\bigr)$ that detects distribution-level shadowing of the teacher token;
   - an anti-degeneration entropy regularizer $H_\text{tail}$ (Eq. 3) so the optimizer does not collapse onto a few high-frequency tokens.

2. **Localized LoRA fine-tuning (Section 3.3, Eq. 4).** The soft prompts give us a hardness score $r_t$ per warm-start turn — the turns where $G$ disagrees with $F$ the most. We freeze the surrogate base weights, attach LoRA on attention + MLP projections, and minimize a weighted token-NLL on the original-model responses, with per-example weight $\omega_t \propto r_t$.

3. **Switching and rollback (Section 3.3, Section 4.1).** A cosine gate computes a bounded discrepancy score $\mathrm{Gap}(S_t) = 1 - \cos(\phi(a_t^F), \phi(a_t^G))$ on an acceptance batch (output-fidelity gate) and on the warm-start centroid (locality gate). The session switches to $G$ only when both gates pass with the margins prescribed by Corollary 4.3. While the adapted $G$ serves later turns, a drift monitor watches a moving window of $\mathrm{Gap}$ values and rolls back to $F$ when it exceeds threshold for $m_\text{cons}$ consecutive checks.

After switching, the surrogate is served with a *compressed* prefix $\widetilde{S}_t$ — a one-line summary of older turns plus the last $K$ turns verbatim — avoiding repeated full-history processing.

### Why does this work?

Real-world multi-turn dialogues exhibit a long-tail token pattern: the first few turns carry most of the context (task, topic, constraints, roles) while later turns are short, locally-dependent updates. Once $G$ has been adapted to the local manifold induced by the warm-start prefix, it can serve those later turns at a fraction of the cost while preserving the response style and the dialogue state. The paper formalizes this through a local manifold approximation problem (Problem 2.1):
$$\min_{G \in \mathcal{G}} \mathrm{dist}\bigl(\mathcal{M}_k^G(\mathcal{D}_k), \mathcal{M}_k^F(\mathcal{D}_k)\bigr).$$

---

## Repository Organization

```
SOMA/
├── soma/                       # Core method (importable Python package)
│   ├── __init__.py
│   ├── soft_prompt.py          # SoftPrompt + V(P) nearest-token verbalization
│   ├── prompt_mining.py        # Stage 1: semantic divergence + anti-degeneration
│   ├── lora_ft.py              # Stage 2: localized LoRA fine-tuning (weighted NLL)
│   ├── gate.py                 # Stage 3: cosine gate + drift-aware rollback
│   ├── pipeline.py             # SOMASession controller (end-to-end per session)
│   ├── models.py               # HF model loading + generation helpers
│   └── data.py                 # Dialogue dataset utilities
├── scripts/
│   ├── run_soma.py             # End-to-end runner (loads two LMs + a dataset)
│   └── smoke_test.py           # CPU-only sanity check, no LLM download needed
├── configs/
│   ├── default.yaml            # Default hyperparameters (matches Appendix C.2)
│   └── small_demo.yaml         # Tiny budget for a quick smoke run
├── data/
│   └── demo/
│       └── demo_dialogues.json # 4 hand-picked context-dependent dialogues
├── requirements.txt
└── README.md
```

The repository ships only the SOMA method. Baselines (Original, Surrogate, History-Prefix, History-FT, LLMLingua-2, RouteLLM) and the full evaluation datasets are not included; the paper documents how each baseline is configured (Appendix C.3) so it can be reproduced from public sources.

---

## Requirements

Tested with Python 3.10 and CUDA-enabled PyTorch. The minimum runtime dependencies are listed in `requirements.txt`:

```
torch>=2.1
transformers>=4.40
peft>=0.10
accelerate>=0.30
sentencepiece>=0.2
tokenizers>=0.19
safetensors>=0.4
numpy>=1.24
pyyaml>=6.0
tqdm>=4.66
```

Install with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Hardware:
- **Smoke test (`scripts/smoke_test.py`)**: CPU only, no GPU needed.
- **Demo (`configs/small_demo.yaml`)**: one consumer GPU is enough (≈ 8–12 GB VRAM with the Qwen2.5 0.5B/1.5B pair, fp16).
- **Paper setting**: 1×80 GB A100 is enough for the Qwen3-0.6B/8B family with FlashAttention. The LLaMA setting uses LLaMA-3.1-70B as $F$ and is served via a separate inference endpoint — only its text outputs feed back into SOMA.

---

## Quickstart

### 1. CPU-only smoke test

Verifies the imports, soft-prompt mining loss, LoRA batching, and gate logic without downloading any model:

```bash
python scripts/smoke_test.py
```

Expected output ends with `[smoke] OK`.

### 2. End-to-end demo on a small dialogue set

The demo uses two small Qwen2.5 instruct checkpoints and the 4-dialogue sample shipped in `data/demo/`:

```bash
python scripts/run_soma.py \
    --config configs/small_demo.yaml \
    --dataset data/demo/demo_dialogues.json \
    --out_dir out/demo
```

Per-turn logs are written to `out/demo/soma_log.json`. Each turn records who served it (`F` = original, `G` = adapted surrogate), whether the switch happened, and any locality / drift annotations.

### 3. Paper-scale evaluation

Point the runner at a real dataset (e.g., a ShareGPT subset) and set the original / surrogate IDs to a paper-style pair:

```bash
python scripts/run_soma.py \
    --original_model Qwen/Qwen3-8B \
    --surrogate_model Qwen/Qwen3-0.6B \
    --dataset data/your_sharegpt_subset.json \
    --out_dir out/qwen_full \
    --config configs/default.yaml
```

The dataset must follow the schema

```json
{
  "<dialogue_id>": ["<turn_1_user_query>", "<turn_2_user_query>", ...]
}
```

Following the paper's filtering protocol (Appendix C.4), dialogues should be pre-filtered to context-dependent conversations before evaluation.

---

## Configuration

`configs/default.yaml` mirrors the hyperparameter ranges in Appendix C.2 of the paper. The most relevant knobs:

| Field                                 | Section / Eq. | Notes                                                 |
|---------------------------------------|---------------|-------------------------------------------------------|
| `warm_window`                         | 3.3, 4.3      | $W$ in turns. Paper: $W \in [3, 12]$.                |
| `mining.soft_len`                     | 3.1           | Prompt length $L \in \{4, 8, 16, 32, 64\}$.           |
| `mining.n_candidates`                 | 4.3           | $M$ in Corollary 4.6 (paper: $M \in \{3, 4, 5\}$).    |
| `mining.neighborhood_k`               | Def. 3.1      | Semantic neighborhood size.                           |
| `mining.teacher_topm`                 | 3.2           | Top-$m$ truncation of $\Pi_{t,i}(\mathbf{P})$.        |
| `mining.beta_anti_degen`              | Eq. 3         | $\beta \in [0.02, 0.15]$.                             |
| `lora.rank` / `lora.alpha`            | 3.3           | LoRA rank $r$ and scale $\alpha_\text{lora}$.         |
| `gate.accept_threshold`               | Cor. 4.3      | $\varepsilon$ in $[0.05, 0.12]$ for switching.        |
| `gate.consecutive_to_rollback`        | 3.3           | $m_\text{cons} \in \{2, 3\}$.                         |

---

## Implementation Notes

A few faithful but implementation-pragmatic choices worth flagging:

- **Encoder $\phi$ for the gate.** We mean-pool the surrogate's last hidden state. Any sentence encoder satisfying the bounded-discrepancy assumption in Lemma 4.1 can be swapped in by replacing `_encode` in `soma/gate.py`.
- **Top-$m$ expectation.** The expected embedding $\bar e_{t,i} = \mathbb{E}\,\Pi_{t,i}(\mathbf{P})$ is computed over the top-$m$ tokens of the surrogate distribution (Section 3.2). This keeps the per-step cost linear in $m$ rather than $|\mathcal{V}|$.
- **LoRA targets.** Default targets are `{q,k,v,o,gate,up,down}_proj`, matching standard LLaMA / Qwen attention + MLP names. Adjust `lora.target_modules` for other architectures.
- **Compressed prefix $\widetilde{S}_t$.** A one-line summary of older turns plus the last $K$ turns (`compressed_prefix_turns`) verbatim. The summary heuristic is intentionally simple; richer summarization can be plugged in.

---

## License

This repository is released under the MIT License. See `LICENSE` for details.
