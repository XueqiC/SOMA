"""Dataset loading and per-turn iteration utilities.

The repo ships a tiny demo dataset under data/demo/. Real benchmarks
(ShareGPT, ReMeDi, Craigslist, Multi-Char, MATH, MT-Bench) follow the same
schema:

    {
      "<dialogue_id>": [
        "<user_turn_1>",
        "<user_turn_2>",
        ...
      ]
    }

Per the paper's filtering protocol (Appendix C.4), each dialogue should be
context-dependent: later answers should rely on earlier turns. The demo file
hand-picks small examples that exhibit this property.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_dialogues(path: str | Path) -> Dict[str, List[str]]:
    p = Path(path)
    data = json.loads(p.read_text())
    return {str(k): list(v) for k, v in data.items()}


def iter_dialogues(dialogues: Dict[str, List[str]]) -> Iterable[Tuple[str, List[str]]]:
    for dlg_id, turns in dialogues.items():
        yield dlg_id, turns


def split_warm_remainder(turns: List[str], warm_window: int) -> Tuple[List[str], List[str]]:
    """Split a dialogue into warm-start turns (first W) and serve turns (rest)."""
    w = min(max(warm_window, 1), len(turns))
    return turns[:w], turns[w:]
