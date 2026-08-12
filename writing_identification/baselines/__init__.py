"""Baseline models for authorship verification evaluation."""

from .base import BaselineModel
from .luar import LUARBaseline
from .modernbert import ModernBERTBaseline
from .llm import LLMBaseline, create_haiku_baseline, create_sonnet_baseline, create_opus_baseline
from .custom_model import CustomModelBaseline

__all__ = [
    "BaselineModel",
    "LUARBaseline",
    "ModernBERTBaseline",
    "LLMBaseline",
    "create_haiku_baseline",
    "create_sonnet_baseline",
    "create_opus_baseline",
    "CustomModelBaseline"
]
