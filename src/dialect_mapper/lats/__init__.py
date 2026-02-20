"""
LATS (Language Agent Tree Search) core module for SQL dialect conversion.

This module implements a Monte-Carlo Tree Search approach to converting SQL
from various dialects into e6data-compatible SQL. Instead of a single-shot
LLM call, it explores multiple fix strategies and validates each one using
the e6data planner (no-execute mode).
"""

from .state import Node, Reflection, TreeState
from .graph import create_lats_graph
from .pipeline import run_coverage_test

__all__ = ["Node", "Reflection", "TreeState", "create_lats_graph", "run_coverage_test"]
