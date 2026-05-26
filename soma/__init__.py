"""SOMA: Soft-prompts for lOcal Manifold Approximation.

Efficient multi-turn LLM serving by adapting a small surrogate model to the
local response manifold of a large model, conditioned on the early dialogue
turns. Three stages: soft-prompt mining of weak-alignment directions,
localized LoRA fine-tuning, and a cosine gate that switches between models
with drift-aware rollback.
"""

from .soft_prompt import SoftPrompt, verbalize_soft_prompt
from .prompt_mining import mine_soft_prompts, SemanticDivergenceConfig
from .lora_ft import localized_lora_finetune, LoRAFTConfig
from .gate import CosineGate, GateConfig
from .pipeline import SOMASession, SOMAConfig

__all__ = [
    "SoftPrompt",
    "verbalize_soft_prompt",
    "mine_soft_prompts",
    "SemanticDivergenceConfig",
    "localized_lora_finetune",
    "LoRAFTConfig",
    "CosineGate",
    "GateConfig",
    "SOMASession",
    "SOMAConfig",
]
