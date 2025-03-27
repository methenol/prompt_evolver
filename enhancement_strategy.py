"""
Enhancement Strategy module defining the structure of prompt enhancement strategies.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class EnhancementStrategy:
    """Represents a strategy for enhancing prompts."""
    name: str
    temperature: float
    chain_of_thought: bool
    semantic_check: bool
    context_preservation: bool
    system_prompt: str
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None