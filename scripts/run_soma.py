"""End-to-end SOMA runner.

Loads two HuggingFace causal LMs (original F + surrogate G), iterates over a
dialogue dataset, and serves each session through `SOMASession`. Per-turn logs
go to a JSON file under --out_dir.

Example (small models, single GPU):

    python scripts/run_soma.py \\
        --original_model Qwen/Qwen2.5-1.5B-Instruct \\
        --surrogate_model Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset data/demo/demo_dialogues.json \\
        --out_dir out/demo \\
        --warm_window 3

To match the paper's reported families, use:
    F = meta-llama/Llama-3.1-70B-Instruct (or Qwen/Qwen3-8B)
    G = meta-llama/Llama-2-7b-chat-hf      (or Qwen/Qwen3-0.6B)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from soma import (
    GateConfig,
    LoRAFTConfig,
    SemanticDivergenceConfig,
    SOMAConfig,
    SOMASession,
)
from soma.data import load_dialogues
from soma.models import load_lm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args, yaml_cfg: dict) -> SOMAConfig:
    mining_kwargs = yaml_cfg.get("mining", {})
    lora_kwargs = yaml_cfg.get("lora", {})
    gate_kwargs = yaml_cfg.get("gate", {})
    return SOMAConfig(
        warm_window=args.warm_window or yaml_cfg.get("warm_window", 4),
        compressed_prefix_turns=yaml_cfg.get("compressed_prefix_turns", 2),
        max_new_tokens=args.max_new_tokens or yaml_cfg.get("max_new_tokens", 200),
        acceptance_min_turns=yaml_cfg.get("acceptance_min_turns", 2),
        mining=SemanticDivergenceConfig(**mining_kwargs),
        lora=LoRAFTConfig(**lora_kwargs),
        gate=GateConfig(**gate_kwargs),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--original_model", type=str, default=None)
    p.add_argument("--surrogate_model", type=str, default=None)
    p.add_argument("--dataset", type=str, default="data/demo/demo_dialogues.json")
    p.add_argument("--out_dir", type=str, default="out/run")
    p.add_argument("--warm_window", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--limit_dialogues", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg_path = Path(args.config)
    yaml_cfg: dict = {}
    if cfg_path.exists():
        with cfg_path.open() as f:
            yaml_cfg = yaml.safe_load(f) or {}

    original_id = args.original_model or yaml_cfg.get("original_model")
    surrogate_id = args.surrogate_model or yaml_cfg.get("surrogate_model")
    if not original_id or not surrogate_id:
        raise SystemExit(
            "Both --original_model and --surrogate_model (or the yaml config) are required."
        )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"[load] original = {original_id}")
    original = load_lm(original_id, device_map=args.device, dtype=dtype)
    print(f"[load] surrogate = {surrogate_id}")
    surrogate = load_lm(surrogate_id, device_map=args.device, dtype=dtype)

    config = build_config(args, yaml_cfg)
    dialogues = load_dialogues(args.dataset)
    if args.limit_dialogues is not None:
        keys = list(dialogues.keys())[: args.limit_dialogues]
        dialogues = {k: dialogues[k] for k in keys}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for dlg_id, turns in dialogues.items():
        print(f"\n[dialogue {dlg_id}] turns={len(turns)}")
        t0 = time.time()
        session = SOMASession(original=original, surrogate=surrogate, config=config)
        dialogue_log = []
        for q in turns:
            log = session.serve(q)
            dialogue_log.append(
                {
                    "turn": log.turn_idx,
                    "served_by": log.served_by,
                    "response": log.response,
                    "note": log.note,
                }
            )
            print(
                f"  turn {log.turn_idx:>2} [{log.served_by}] "
                f"{log.response[:120].replace(chr(10), ' ')}"
            )
        results[dlg_id] = {
            "turns": dialogue_log,
            "elapsed_s": round(time.time() - t0, 2),
            "switched": session.switched,
        }
        # Save after every dialogue so a crash doesn't lose progress.
        (out_dir / "soma_log.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n[done] wrote {out_dir / 'soma_log.json'}")


if __name__ == "__main__":
    main()
