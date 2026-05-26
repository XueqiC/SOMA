"""End-to-end SOMA serving pipeline.

Operates on one dialogue at a time. For each session:

  1. Warm-start phase (turns 1..W): serve with the original model F and
     collect (context, query, teacher_response) tuples.
  2. Soft-prompt mining: optimize P over warm-start turns to maximize
     L_sem_exp (Section 3.2). Keep the candidate with the lowest mining loss
     and record per-turn hardness scores r_t.
  3. Localized LoRA fine-tuning on the mined hard turns, weighted by r_t.
  4. Acceptance gate: probe F and G on the warm-start turns under the same
     dialogue state; switch to G only if mean Gap <= eps.
  5. Post-switch serving: G handles later turns with a compressed prefix
     (last K turns + a fixed summary), and a drift monitor rolls back to F
     when the moving Gap exceeds the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .gate import CosineGate, GateConfig
from .lora_ft import LoRAFTConfig, attach_lora, localized_lora_finetune
from .models import LMHandle, format_dialogue_context, generate_text
from .prompt_mining import SemanticDivergenceConfig, mine_soft_prompts


@dataclass
class SOMAConfig:
    warm_window: int = 4
    compressed_prefix_turns: int = 2
    max_new_tokens: int = 200
    acceptance_min_turns: int = 2
    mining: SemanticDivergenceConfig = field(default_factory=SemanticDivergenceConfig)
    lora: LoRAFTConfig = field(default_factory=LoRAFTConfig)
    gate: GateConfig = field(default_factory=GateConfig)


@dataclass
class TurnLog:
    turn_idx: int
    served_by: str
    response: str
    note: str = ""


def _summarize_context(history: List[Dict[str, str]], keep_last: int) -> str:
    """Compressed input ~S_t: keep a one-line summary of older turns + the
    last `keep_last` turns verbatim."""
    if len(history) <= keep_last:
        return format_dialogue_context(history)
    older = history[:-keep_last] if keep_last > 0 else history
    recent = history[-keep_last:] if keep_last > 0 else []
    if older:
        topics = []
        for turn in older:
            q = turn.get("user", "")
            if q:
                topics.append(q.strip().split("\n")[0][:80])
        summary = "Earlier topics: " + " | ".join(topics)
    else:
        summary = ""
    body = format_dialogue_context(recent)
    return (summary + "\n" + body).strip() if summary else body


class SOMASession:
    """Per-session SOMA controller. Keep one instance per dialogue."""

    def __init__(
        self,
        original: LMHandle,
        surrogate: LMHandle,
        config: SOMAConfig,
    ) -> None:
        self.original = original
        self.surrogate = surrogate
        self.config = config
        self.gate = CosineGate(config.gate)

        # Attach a fresh LoRA adapter onto the surrogate. The adapter is
        # zero-initialized so the surrogate behaves identically until we run
        # the localized FT stage.
        self.surrogate.model = attach_lora(self.surrogate.model, config.lora)

        self.history: List[Dict[str, str]] = []   # full (user, assistant) log
        self.warm_records: List[Dict[str, str]] = []
        self.switched: bool = False
        self.logs: List[TurnLog] = []

    # ----- Stage helpers -----

    def _serve_original(self, query: str) -> str:
        prompt = format_dialogue_context(self.history)
        prompt = (prompt + "\n" if prompt else "") + f"User: {query}\nAssistant:"
        return generate_text(
            self.original, prompt, max_new_tokens=self.config.max_new_tokens
        )

    def _serve_surrogate(self, query: str) -> str:
        prompt = _summarize_context(self.history, self.config.compressed_prefix_turns)
        prompt = (prompt + "\n" if prompt else "") + f"User: {query}\nAssistant:"
        return generate_text(
            self.surrogate, prompt, max_new_tokens=self.config.max_new_tokens
        )

    def _try_switch(self) -> Dict[str, object]:
        """Run prompt mining, localized FT, and the acceptance gate."""
        warm_turns = [
            {
                "context": _summarize_context(self.history[: rec["turn_idx"]], self.config.compressed_prefix_turns),
                "query": rec["query"],
                "teacher": rec["teacher"],
            }
            for rec in self.warm_records
        ]
        # Stage 1: mine soft prompts on the warm window.
        prompts, mining_losses, hardness = mine_soft_prompts(
            model=self.surrogate.model,
            tokenizer=self.surrogate.tokenizer,
            warm_start_turns=warm_turns,
            config=self.config.mining,
        )

        # Build localized training set: every warm-start example weighted by r_t.
        # Pure random-FT samples uniformly; SOMA up-weights weak-alignment cases.
        total_hardness = sum(hardness) + 1e-9
        training_set: List[Dict[str, object]] = []
        for rec, r in zip(warm_turns, hardness):
            training_set.append(
                {
                    "context": rec["context"],
                    "query": rec["query"],
                    "response": rec["teacher"],
                    "weight": r / total_hardness,
                }
            )

        # Stage 2: LoRA FT on those hard cases.
        ft_stats = localized_lora_finetune(
            model=self.surrogate.model,
            tokenizer=self.surrogate.tokenizer,
            training_set=training_set,
            config=self.config.lora,
        )

        # Stage 3: acceptance gate on the warm-start window.
        self.surrogate.model.eval()
        warm_surrogate_responses: List[str] = []
        warm_original_responses: List[str] = []
        for rec in self.warm_records:
            cand = generate_text(
                self.surrogate,
                _summarize_context(self.history[: rec["turn_idx"]], self.config.compressed_prefix_turns)
                + (f"\nUser: {rec['query']}\nAssistant:"),
                max_new_tokens=self.config.max_new_tokens,
            )
            warm_surrogate_responses.append(cand)
            warm_original_responses.append(rec["teacher"])

        self.gate.set_warm_start(
            model=self.surrogate.model,
            tokenizer=self.surrogate.tokenizer,
            queries=[rec["query"] for rec in self.warm_records],
        )
        accept = self.gate.acceptance_check(
            original_responses=warm_original_responses,
            surrogate_responses=warm_surrogate_responses,
            model=self.surrogate.model,
            tokenizer=self.surrogate.tokenizer,
        )
        self.switched = bool(accept.get("accept"))
        return {
            "mining_losses": mining_losses,
            "hardness": hardness,
            "ft_stats": ft_stats,
            "accept": accept,
            "switched": self.switched,
        }

    # ----- Public API -----

    def serve(self, query: str) -> TurnLog:
        """Serve one user turn, applying SOMA's switching/rollback logic."""
        turn_idx = len(self.history) + 1

        # 1) Warm-start phase: always serve with original F.
        if turn_idx <= self.config.warm_window:
            answer = self._serve_original(query)
            self.history.append({"user": query, "assistant": answer})
            self.warm_records.append(
                {
                    "turn_idx": turn_idx - 1,
                    "query": query,
                    "teacher": answer,
                }
            )
            log = TurnLog(turn_idx, "F", answer, note="warm-start")
            self.logs.append(log)

            # If we just finished the warm-start window, run mining + FT + gate.
            if turn_idx == self.config.warm_window:
                stats = self._try_switch()
                tail = "switch=YES" if self.switched else "switch=NO"
                if self.logs:
                    self.logs[-1].note = f"warm-start; {tail}; gap={stats['accept'].get('mean_gap'):.3f}"
            return log

        # 2) Post-warm-start serving.
        if not self.switched:
            # Gate rejected the switch -- continue with F.
            answer = self._serve_original(query)
            self.history.append({"user": query, "assistant": answer})
            log = TurnLog(turn_idx, "F", answer, note="post-warm, gate rejected")
            self.logs.append(log)
            return log

        # Locality gate: is the current query still close to the warm region?
        ok, dist = self.gate.locality_ok(
            query=query, model=self.surrogate.model, tokenizer=self.surrogate.tokenizer
        )
        if not ok:
            self.switched = False
            self.gate.reset_after_rollback()
            answer = self._serve_original(query)
            self.history.append({"user": query, "assistant": answer})
            log = TurnLog(turn_idx, "F", answer, note=f"rollback (locality dist={dist:.2f})")
            self.logs.append(log)
            return log

        # Serve with the adapted surrogate.
        answer = self._serve_surrogate(query)
        self.history.append({"user": query, "assistant": answer})
        log = TurnLog(turn_idx, "G", answer, note=f"locality dist={dist:.2f}")
        self.logs.append(log)
        return log
