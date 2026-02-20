"""
Graph node functions for the LATS dialect agent.

These are the functions that run at each step of the LangGraph StateGraph.
There are three:

  1. generate_initial_response(): Creates the root node (first candidate fix)
  2. expand(): Selects the best leaf, generates N new candidates, validates them
  3. should_loop(): Decides whether to keep searching or stop

The overall flow:
  START → generate_initial_response → [should_loop] → expand → [should_loop] → expand → ... → END
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from .state import Node, Reflection, TreeState
from .tools import PlannerValidator, DocSearcher, SemanticChecker
from .prompts import GENERATION_PROMPT, REFLECTION_PROMPT, format_trajectory_context

logger = logging.getLogger(__name__)

# How many candidate fixes to generate per expansion step
DEFAULT_NUM_CANDIDATES = 3

# Maximum tree depth before we stop searching
MAX_TREE_HEIGHT = 3


# ---------------------------------------------------------------------------
# Helper: Parse JSON from LLM response (handles common formatting issues)
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling common issues like
    markdown code fences, trailing text, etc.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed dict, or empty dict if parsing fails
    """
    import re

    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON within markdown code fences
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object anywhere in the text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON from LLM response: {text[:200]}...")
    return {}


# ---------------------------------------------------------------------------
# Helper: Run reflection on a single candidate
# ---------------------------------------------------------------------------

def _reflect_on_candidate(
    llm: BaseChatModel,
    original_sql: str,
    error_message: str,
    candidate_sql: str,
    error_analysis: str,
    fix_strategy: str,
    planner_result: dict,
    semantic_result: dict,
    state: TreeState = None,
) -> Reflection:
    """
    Ask the LLM to evaluate a candidate fix and return a Reflection.

    This combines all the signals:
      - The original error
      - The candidate fix + its reasoning
      - Whether the planner said it's valid
      - Whether it preserves the original query's intent

    Args:
        llm: The language model to use
        original_sql: The original failing SQL
        error_message: The original error message
        candidate_sql: The proposed fix
        error_analysis: The fixer's analysis of the error
        fix_strategy: The fixer's description of what it changed
        planner_result: {"success": bool, "error": str} from the planner
        semantic_result: {"preserves_intent": bool, "differences": str}

    Returns:
        A Reflection object with score, analysis, and found_solution flag
    """
    # Format planner result as readable text
    if planner_result["success"]:
        planner_text = "SUCCESS — the SQL compiles and is valid in e6data"
    else:
        planner_text = f"FAILED — {planner_result['error']}"

    # Format semantic check as readable text
    if semantic_result["preserves_intent"]:
        semantic_text = "Preserves intent — the candidate is semantically equivalent to the original"
    else:
        semantic_text = f"DIFFERS — {semantic_result['differences']}"

    # Build the reflection prompt
    prompt = REFLECTION_PROMPT.format(
        original_sql=original_sql,
        error_message=error_message,
        candidate_sql=candidate_sql,
        error_analysis=error_analysis,
        fix_strategy=fix_strategy,
        planner_result=planner_text,
        semantic_result=semantic_text,
    )

    # ── Deterministic scoring based on execution result ──────────────
    # The LLM only provides reasoning text ("reflections").
    # Score and found_solution are set here based on actual execution:
    #   - Output matched  → score=1.0, found_solution=True  → STOP
    #   - Output mismatch → score=0.5, found_solution=False → keep searching
    #   - Execution error  → score=0.2, found_solution=False → keep searching
    # normalized_score = score / total_queries (set by pipeline)
    if planner_result["success"]:
        deterministic_score = 1.0
        deterministic_solved = True
    elif "mismatch" in planner_result.get("error", "").lower():
        deterministic_score = 0.5
        deterministic_solved = False
    else:
        deterministic_score = 0.2
        deterministic_solved = False

    total_queries = (state or {}).get("_total_queries", 1)

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        parsed = _parse_json_response(content)

        return Reflection(
            error_analysis=error_analysis,
            reflections=parsed.get("reflections", ""),
            score=deterministic_score,
            total_queries=total_queries,
            found_solution=deterministic_solved,
            planner_success=planner_result["success"],
            planner_error=planner_result.get("error", ""),
            preserves_intent=semantic_result["preserves_intent"],
        )
    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        return Reflection(
            error_analysis=error_analysis,
            reflections=f"Reflection failed: {e}",
            score=deterministic_score,
            total_queries=total_queries,
            found_solution=deterministic_solved,
            planner_success=planner_result.get("success", False),
            planner_error=planner_result.get("error", ""),
            preserves_intent=semantic_result.get("preserves_intent", False),
        )


# ---------------------------------------------------------------------------
# Helper: Generate a candidate fix and validate it
# ---------------------------------------------------------------------------

def _generate_and_validate_candidate(
    llm: BaseChatModel,
    state: TreeState,
    trajectory_messages: list = None,
) -> tuple[list, dict, Reflection]:
    """
    Generate one candidate SQL fix, validate it, and reflect on it.

    This is the core "simulate" step: generate → validate → reflect.

    Args:
        llm: The language model
        state: Current tree state (has docs, original_sql, etc.)
        trajectory_messages: Previous attempts for context (None for first attempt)

    Returns:
        (messages, parsed_response, reflection) tuple
    """
    planner: PlannerValidator = state.get("_planner", PlannerValidator())
    semantic_checker: SemanticChecker = state.get("_semantic_checker", SemanticChecker())

    # Build the trajectory context string (empty for first attempt)
    trajectory_context = format_trajectory_context(trajectory_messages or [])

    # Format docs as readable text
    docs_text = ""
    for i, doc in enumerate(state.get("docs", [])[:5], 1):
        docs_text += f"\n{i}. {doc.get('title', f'Doc {i}')}:\n{doc.get('content', '')}\n"

    # Build the generation prompt
    prompt = GENERATION_PROMPT.format(
        original_sql=state["original_sql"],
        error_message=state["error_message"],
        trajectory_context=trajectory_context,
        docs=docs_text or "(No documentation available)",
        calcite_idioms=state.get("calcite_idioms", "(Not available)"),
    )

    # Call the LLM to generate a candidate fix
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    parsed = _parse_json_response(content)

    candidate_sql = parsed.get("sql_query", "").strip()
    error_analysis = parsed.get("error_analysis", "")
    fix_strategy = parsed.get("fix_strategy", "")

    # If parsing failed, use the raw response as the SQL (best effort)
    if not candidate_sql:
        candidate_sql = content.strip()
        error_analysis = "Could not parse structured response"
        fix_strategy = "Raw LLM output used as SQL"

    # Create the messages for this node
    # AIMessage: the candidate fix
    # HumanMessage: the planner validation result (this is the feedback)
    messages = [AIMessage(content=content)]

    # Validate with the planner (no-execute)
    planner_result = planner.validate(
        candidate_sql,
        catalog=state.get("catalog", ""),
        database=state.get("database", ""),
    )

    # Add planner result as feedback message
    if planner_result["success"]:
        planner_msg = "Planner validation: SUCCESS — SQL is valid in e6data."
    else:
        planner_msg = f"Planner validation: FAILED — {planner_result['error']}"
    messages.append(HumanMessage(content=planner_msg))

    # Check semantic equivalence
    semantic_result = semantic_checker.check(state["original_sql"], candidate_sql)

    # Reflect on this candidate
    reflection = _reflect_on_candidate(
        llm=llm,
        original_sql=state["original_sql"],
        error_message=state["error_message"],
        candidate_sql=candidate_sql,
        error_analysis=error_analysis,
        fix_strategy=fix_strategy,
        planner_result=planner_result,
        semantic_result=semantic_result,
        state=state,
    )

    return messages, parsed, reflection


# ---------------------------------------------------------------------------
# Node 1: Generate Initial Response (creates the root node)
# ---------------------------------------------------------------------------

def generate_initial_response(state: TreeState, llm: BaseChatModel, **kwargs) -> TreeState:
    """
    Generate the first candidate fix and create the root of the search tree.

    This is the "start" node in the graph. It:
      1. Fetches e6data docs (once, cached in state)
      2. Generates the first candidate SQL fix
      3. Validates it with the planner
      4. Checks semantic equivalence
      5. Reflects and scores it
      6. Creates the root Node

    Args:
        state: The initial TreeState (has input, original_sql, error_message)
        llm: The language model to use

    Returns:
        Updated TreeState with root node set
    """
    # Fetch docs if not already in state
    if not state.get("docs"):
        doc_searcher: DocSearcher = state.get("_doc_searcher", DocSearcher())
        # Extract search terms from the error message and SQL
        search_terms = state["error_message"].split()[:10]  # Simple: first 10 words
        state["docs"] = doc_searcher.search(search_terms)

    # Generate, validate, and reflect on the first candidate
    messages, parsed, reflection = _generate_and_validate_candidate(
        llm=llm,
        state=state,
        trajectory_messages=None,  # No history yet — this is the first attempt
    )

    # Create the root node of the search tree
    root = Node(messages=messages, reflection=reflection)

    logger.info(
        f"[LATS] Initial candidate: solved={root.is_solved} "
        f"score={reflection.score}/10 "
        f"planner={'PASS' if reflection.planner_success else 'FAIL'}"
    )

    # Return updated state with the root node
    return {
        **state,
        "root": root,
    }


# ---------------------------------------------------------------------------
# Select: Pick the best leaf node to expand
# ---------------------------------------------------------------------------

def select(root: Node) -> Node:
    """
    Select the best leaf node to expand using UCB1.

    Starting from the root, walk down the tree always picking the child
    with the highest UCB score. Stop when we reach a leaf (no children).

    This is the "Select" phase of MCTS — it decides where to focus
    the next round of candidate generation.
    """
    if not root.children:
        return root

    node = root
    while node.children:
        # Pick the child with the highest UCB score
        max_child = max(
            node.children,
            key=lambda child: child.upper_confidence_bound()
        )
        node = max_child

    return node


# ---------------------------------------------------------------------------
# Node 2: Expand (generates N new candidates from the best leaf)
# ---------------------------------------------------------------------------

def expand(state: TreeState, llm: BaseChatModel, **kwargs) -> TreeState:
    """
    Expand the search tree by generating N new candidate fixes.

    This is the "expand" node in the graph. It:
      1. Selects the best leaf node (via UCB)
      2. Gets that node's trajectory (all previous attempts + feedback)
      3. Generates N new candidates (each one sees the full history)
      4. Validates each with the planner + semantic check
      5. Creates child nodes, backpropagates scores

    The key insight: each candidate sees the FULL trajectory, including
    planner errors from previous attempts. This is how the agent
    learns from its mistakes across iterations.

    Args:
        state: Current TreeState with root node
        llm: The language model to use

    Returns:
        Updated TreeState (tree is modified in-place via child nodes)
    """
    root = state["root"]
    num_candidates = state.get("_num_candidates", DEFAULT_NUM_CANDIDATES)

    # Step 1: Select the best leaf to expand
    best_candidate = select(root)

    # Step 2: Get the full trajectory from root to this leaf
    # This includes all previous attempts, planner errors, and reflections
    trajectory = best_candidate.get_trajectory()

    logger.info(
        f"[LATS] Expanding node at depth={best_candidate.depth}, "
        f"generating {num_candidates} candidates"
    )

    # Step 3: Generate N candidates, each seeing the full history
    child_nodes = []
    for i in range(num_candidates):
        try:
            messages, parsed, reflection = _generate_and_validate_candidate(
                llm=llm,
                state=state,
                trajectory_messages=trajectory,  # Full history for context
            )

            # Create child node (backpropagation happens in Node.__init__)
            child = Node(
                messages=messages,
                reflection=reflection,
                parent=best_candidate,
            )
            child_nodes.append(child)

            logger.info(
                f"[LATS]   Candidate {i + 1}/{num_candidates}: "
                f"solved={child.is_solved} score={reflection.score}/10 "
                f"planner={'PASS' if reflection.planner_success else 'FAIL'}"
            )

            # Early exit: if we found a solution, no need to generate more
            if child.is_solved:
                logger.info("[LATS]   Solution found! Stopping expansion early.")
                break

        except Exception as e:
            logger.error(f"[LATS]   Candidate {i + 1} generation failed: {e}")
            continue

    # Step 4: Add children to the tree
    best_candidate.children.extend(child_nodes)

    # State is returned as-is because the tree is modified in-place
    # (child nodes are added directly to best_candidate.children)
    return state


# ---------------------------------------------------------------------------
# Conditional Edge: Should we keep searching?
# ---------------------------------------------------------------------------

def should_loop(state: TreeState) -> str:
    """
    Decide whether to continue the tree search or stop.

    Returns:
        "expand" to keep searching, or "__end__" to stop

    Stop conditions:
      1. A solution was found (planner passed + intent preserved)
      2. Maximum tree height reached (prevent infinite loops)
    """
    root = state["root"]

    # Stop if we found a verified solution anywhere in the tree
    if root.is_solved:
        logger.info(f"[LATS] Solution found! Tree height={root.height}")
        return "__end__"

    # Stop if the tree is too deep (max depth = 3)
    if root.height >= MAX_TREE_HEIGHT:
        logger.info(
            f"[LATS] Max height {MAX_TREE_HEIGHT} reached. "
            f"Returning best-effort result."
        )
        return "__end__"

    # Keep searching
    logger.info(f"[LATS] No solution yet. Tree height={root.height}. Expanding...")
    return "expand"