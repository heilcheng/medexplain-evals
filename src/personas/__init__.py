# src/personas/__init__.py
"""Audience persona definitions and prompt templates."""

from .audience_personas import PersonaFactory, AudiencePersona
from .prompt_templates import PromptTemplates

__all__ = ["PersonaFactory", "AudiencePersona", "PromptTemplates"]
