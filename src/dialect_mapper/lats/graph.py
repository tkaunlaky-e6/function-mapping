"""
LangGraph StateGraph definition for the LATS dialect agent.

This file assembles the graph from the node functions defined in nodes.py.
The graph looks like this:

    START → generate_initial_response → [should_loop?] ─── yes ──→ expand → [should_loop?] ─┐
                                              │                                                │
                                              └── no (solved or max depth) ──→ END            │
                                                                                               │
                                              ┌────────────────────────────────────────────────┘
                                              │
                                              └── no (solved or max depth) ──→ END

Usage:
    from lats_core.graph import create_lats_graph

    graph = create_lats_graph(llm=my_llm)
    result = graph.invoke({
        "input": "Fix this SQL...",
        "original_sql": "SELECT month FROM sales",
        "error_message": "reserved keyword 'month'",
        "catalog": "",
        "database": "",
    })

    best = result["root"].get_best_solution()
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph, START

from .state import TreeState
from .nodes import generate_initial_response, expand, should_loop
from .tools import PlannerValidator, DocSearcher, SemanticChecker

logger = logging.getLogger(__name__)


def create_lats_graph(
    llm: BaseChatModel,
    planner: Optional[PlannerValidator] = None,
    doc_searcher: Optional[DocSearcher] = None,
    semantic_checker: Optional[SemanticChecker] = None,
    num_candidates: int = 3,
):
    """
    Create and compile the LATS StateGraph for SQL dialect conversion.

    The graph has two nodes:
      1. "start": Generates the first candidate fix (root of the tree)
      2. "expand": Expands the tree with N new candidates from the best leaf

    And one conditional edge:
      - should_loop: After each node, check if we found a solution or hit max depth

    Args:
        llm: The language model to use for generation + reflection
        planner: Planner validator (defaults to stub that always passes)
        doc_searcher: Doc searcher (defaults to stub with basic e6data tips)
        semantic_checker: Semantic checker (defaults to basic structural check)
        num_candidates: How many candidates to generate per expansion (default 3)

    Returns:
        A compiled LangGraph that can be invoked with TreeState
    """
    # Use defaults for any tools not provided
    planner = planner or PlannerValidator()
    doc_searcher = doc_searcher or DocSearcher()
    semantic_checker = semantic_checker or SemanticChecker(llm=llm)

    # We inject the tools and LLM into the state as private keys.
    # The node functions read these from state to access the tools.
    # This keeps the node functions pure (no global state).
    def _inject_tools(state: TreeState) -> TreeState:
        """Add tool instances and config to the state."""
        state["_planner"] = planner
        state["_doc_searcher"] = doc_searcher
        state["_semantic_checker"] = semantic_checker
        state["_num_candidates"] = num_candidates
        return state

    # Wrap node functions to inject the LLM parameter
    # (LangGraph nodes receive only the state dict, so we use partial)
    def start_node(state: TreeState) -> TreeState:
        state = _inject_tools(state)
        return generate_initial_response(state, llm=llm)

    def expand_node(state: TreeState) -> TreeState:
        state = _inject_tools(state)
        return expand(state, llm=llm)

    # Build the graph
    builder = StateGraph(TreeState)

    # Add the two nodes
    builder.add_node("start", start_node)
    builder.add_node("expand", expand_node)

    # Wire up the edges:
    #   START → start (always)
    builder.add_edge(START, "start")

    #   start → should_loop → expand or END
    builder.add_conditional_edges(
        "start",
        should_loop,
        {"expand": "expand", "__end__": END},
    )

    #   expand → should_loop → expand or END
    builder.add_conditional_edges(
        "expand",
        should_loop,
        {"expand": "expand", "__end__": END},
    )

    # Compile and return
    graph = builder.compile()

    logger.info(
        f"[LATS] Graph compiled: num_candidates={num_candidates}, "
        f"planner={planner.__class__.__name__}, "
        f"doc_searcher={doc_searcher.__class__.__name__}"
    )

    return graph


def run_lats(
    llm: BaseChatModel,
    original_sql: str,
    error_message: str,
    catalog: str = "",
    database: str = "",
    calcite_idioms: str = "",
    planner: Optional[PlannerValidator] = None,
    doc_searcher: Optional[DocSearcher] = None,
    num_candidates: int = 3,
    total_queries: int = 1,
) -> dict:
    """
    Convenience function to run the full LATS pipeline.

    Creates the graph, invokes it, and returns the result with the best solution
    extracted. This is the simplest way to use the LATS dialect agent.

    Args:
        llm: The language model
        original_sql: The failing SQL query
        error_message: Why it failed
        catalog: E6Data catalog (for planner validation)
        database: E6Data database (for planner validation)
        calcite_idioms: Reserved keywords text (optional)
        planner: Custom planner validator (optional)
        doc_searcher: Custom doc searcher (optional)
        num_candidates: Candidates per expansion (default 3)

    Returns:
        {
            "best_sql": str,          # The best SQL fix found
            "is_solved": bool,        # Whether the planner verified it
            "score": float,           # Best node's score (0-10)
            "iterations": int,        # How many tree levels were explored
            "root": Node,             # The full search tree (for inspection)
        }
    """
    # Create the graph
    graph = create_lats_graph(
        llm=llm,
        planner=planner,
        doc_searcher=doc_searcher,
        semantic_checker=SemanticChecker(llm=llm),
        num_candidates=num_candidates,
    )

    # Build the initial state
    initial_state = {
        "input": f"Translate this Spark SQL to e6data:\n{original_sql}\n\nError:\n{error_message}",
        "original_sql": original_sql,
        "error_message": error_message,
        "docs": [],
        "calcite_idioms": calcite_idioms,
        "catalog": catalog,
        "database": database,
        "_total_queries": total_queries,
    }

    # Run the graph
    result = graph.invoke(initial_state)

    # Extract the best solution
    root = result["root"]
    best_node = root.get_best_solution()
    best_trajectory = best_node.get_trajectory(include_reflections=False)

    # Extract SQL from the best node's AIMessage
    best_sql = ""
    for msg in reversed(best_trajectory):
        if hasattr(msg, 'content') and msg.content:
            # Try to parse JSON to get sql_query
            try:
                import json
                parsed = json.loads(msg.content)
                if "sql_query" in parsed:
                    best_sql = parsed["sql_query"]
                    break
            except (json.JSONDecodeError, TypeError):
                pass

    # Fallback: use the raw content of the last AIMessage
    if not best_sql:
        from langchain_core.messages import AIMessage
        for msg in reversed(best_trajectory):
            if isinstance(msg, AIMessage) and msg.content:
                best_sql = msg.content
                break

    return {
        "best_sql": best_sql,
        "is_solved": root.is_solved,
        "score": best_node.reflection.score if best_node.reflection else 0,
        "iterations": root.height,
        "root": root,
    }