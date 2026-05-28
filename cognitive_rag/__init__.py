"""Cognitive RAG agent — vector memory, MCP tools, four-role loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["CognitiveAgent"]

if TYPE_CHECKING:
    from cognitive_rag.agent import CognitiveAgent


def __getattr__(name: str):
    if name == "CognitiveAgent":
        from cognitive_rag.agent import CognitiveAgent

        return CognitiveAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
