# SOMA: Efficient Multi-turn LLM Serving via Small Language Model

**TL;DR.** Later turns in a dialogue are shorter but highly dependent on early context. **SOMA** replaces a large model with a small surrogate **within** a session by:
1) mining *least-aligned local steps* via soft-prompt tuning,
2) adapting the small model with localized LoRA on those mined cases, and
3) switching using a fast cosine-based gate with rollback on drift.

This yields high fidelity to the original model with far fewer tokens and competitive throughput.

---

## What’s in this repo
- `src/` — implementation of SOMA (soft-prompt tuning, expectation-weighted divergence, anti-degeneration loss, localized LoRA, switch/rollback).
- `configs/` — example YAMLs (models, optimizer, losses, thresholds).
- `scripts/` — experiment runners (mining, LoRA adapt, evaluate, ablations).
- `data/` — dataset loaders (ShareGPT, ReMeDi, Craigslist, Multi-Char, MATH, MT-Bench).
- `figures/` — plotting utilities for paper figures.
- `README.md` — this file.
- `LICENSE` — license.

> Note: Technical details, proofs, and all results are in the paper.

---

## Installation

We use PyTorch + HuggingFace + vLLM. Recommended environment:

```bash
conda create -n soma python=3.10 -y
conda activate soma
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.10.2 transformers==4.56.1 accelerate==1.9.0 \
            sentencepiece==0.2.0 tiktoken==0.11.0 einops==0.8.1 \
            datasets==4.0.0 huggingface-hub==0.34.2 safetensors==0.5.3 \
            scikit-learn==1.7.1 seaborn==0.13.2 ray==2.49.1
# optional (quantization / serving API)
pip install bitsandbytes==0.46.1 fastapi==0.116.2 uvicorn==0.35.0
