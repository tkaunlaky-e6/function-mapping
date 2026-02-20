"""
State definitions for the LATS (Language Agent Tree Search) dialect agent.

This file defines three core classes:
  - Reflection: The LLM's evaluation of a candidate SQL fix
  - Node: A single node in the Monte-Carlo search tree
  - TreeState: The top-level state passed through the LangGraph StateGraph

The tree search works like this:
  1. Each Node holds a candidate SQL fix + its reflection (score + planner result)
  2. Nodes form a tree — children are refinements of their parent's approach
  3. UCB (Upper Confidence Bound) balances exploring new branches vs exploiting good ones
  4. Backpropagation updates scores up the tree when we learn how good a branch is
"""

import math
from collections import deque
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Reflection: How the LLM evaluates a candidate SQL fix
# ---------------------------------------------------------------------------

class Reflection(BaseModel):
    """
    The LLM's evaluation of a single candidate SQL fix.

    After generating a candidate fix, we:
      1. Run it through the planner (no-execute validation)
      2. Check if it preserves the original query's intent
      3. Ask the LLM to reflect on the quality of the fix

    All of that is captured here.
    """

    error_analysis: str = Field(
        default="",
        description="What error was identified in the original SQL"
    )
    reflections: str = Field(
        default="",
        description="The LLM's critique of this fix attempt — what worked, what didn't"
    )
    score: float = Field(
        default=0.0,
        description="Per-query score: 1.0 = output matched, 0.5 = mismatch, 0.2 = error.",
        ge=0.0,
        le=1.0,
    )
    total_queries: int = Field(
        default=1,
        description="Total basis queries for this function. Used for normalization.",
        ge=1,
    )
    found_solution: bool = Field(
        default=False,
        description="True when output matches expected — set deterministically, not by LLM"
    )
    planner_success: bool = Field(
        default=False,
        description="Did the e6data execution succeed?"
    )
    planner_error: str = Field(
        default="",
        description="Error from execution (empty if success). Fed into next LATS iteration."
    )
    preserves_intent: bool = Field(
        default=False,
        description="Does the candidate SQL preserve the original query's semantics?"
    )

    def as_message(self) -> HumanMessage:
        """
        Convert this reflection into a HumanMessage for the LLM's trajectory.

        When we build the trajectory for the next iteration, we include
        reflections so the LLM can see what it tried before and how it went.
        """
        # Include the planner error if there is one — this is the key feedback signal
        planner_info = "Planner: PASSED" if self.planner_success else f"Planner: FAILED — {self.planner_error}"
        intent_info = "Intent: preserved" if self.preserves_intent else "Intent: NOT preserved"

        return HumanMessage(
            content=(
                f"Reflection on previous attempt:\n"
                f"  {planner_info}\n"
                f"  {intent_info}\n"
                f"  Score: {self.score} (normalized: {self.normalized_score:.3f})\n"
                f"  Analysis: {self.reflections}"
            )
        )

    @property
    def normalized_score(self) -> float:
        """Normalized score = score / total_queries. Used for UCB."""
        return self.score / self.total_queries


# ---------------------------------------------------------------------------
# Node: A single node in the Monte-Carlo search tree
# ---------------------------------------------------------------------------

class Node:
    """
    A node in the LATS search tree.

    Each node represents one candidate SQL fix attempt. The tree structure:
      - Root: the first candidate generated from the original error
      - Children: refinements that try to fix what the parent got wrong
      - Leaves: the most recent attempts (not yet expanded)

    The search uses UCB1 (Upper Confidence Bound) to decide which leaf
    to expand next. This balances:
      - Exploitation: expanding nodes with high scores (good fixes)
      - Exploration: expanding nodes that haven't been visited much

    When a node gets a score, that score is "backpropagated" up to the root,
    updating every ancestor's average reward.
    """

    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        # The messages for this node: typically [AIMessage(candidate SQL), ...]
        self.messages = messages

        # The LLM's reflection on this candidate
        self.reflection = reflection

        # Tree structure
        self.parent = parent
        self.children: list["Node"] = []

        # MCTS values — updated by backpropagation
        self.value = 0.0       # Running average reward
        self.visits = 0        # How many times this node has been visited

        # Depth in the tree (root = 1)
        self.depth = parent.depth + 1 if parent is not None else 1

        # Is this candidate a solution? (planner passed + preserves intent)
        self._is_solved = reflection.found_solution if reflection else False

        # If we found a solution, mark the entire branch as solved
        if self._is_solved:
            self._mark_tree_as_solved()

        # Backpropagate this node's score up to the root
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        solved_str = "SOLVED" if self._is_solved else "unsolved"
        return (
            f"<Node depth={self.depth} value={self.value:.2f} "
            f"visits={self.visits} {solved_str}>"
        )

    # --- Properties ---

    @property
    def is_solved(self) -> bool:
        """True if this node or any descendant has a planner-verified solution."""
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        """A terminal node has no children (it's a leaf)."""
        return not self.children

    @property
    def best_child_score(self) -> Optional["Node"]:
        """Return the child with the highest value, preferring solved ones."""
        if not self.children:
            return None
        return max(
            self.children,
            key=lambda child: int(child.is_solved) * child.value
        )

    @property
    def height(self) -> int:
        """How deep the tree extends below this node."""
        if self.children:
            return 1 + max(child.height for child in self.children)
        return 1

    # --- Core MCTS Methods ---

    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> float:
        """
        Calculate UCB1 score for this node.

        UCB1 = average_reward + C * sqrt(ln(parent_visits) / visits)

        The first term (average_reward) favors nodes with high scores
        → exploitation: keep expanding what's working.

        The second term (exploration_term) favors nodes with few visits
        → exploration: try branches we haven't explored much.

        The exploration_weight (C) controls the balance between the two.
        Higher C = more exploration. Lower C = more exploitation.
        """
        if self.parent is None:
            raise ValueError("Cannot compute UCB for root node")

        if self.visits == 0:
            # Unvisited nodes get their raw value (encourage first visit)
            return self.value

        # Exploitation: how good is this branch on average?
        average_reward = self.value / self.visits

        # Exploration: how under-explored is this branch?
        exploration_term = math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """
        Update this node's score and propagate up to the root.

        After scoring a new candidate, we walk up the tree and update
        every ancestor's running average. This is how the tree "learns"
        which branches are promising.

        The running average formula:
          new_value = (old_value * (visits - 1) + reward) / visits
        """
        node = self
        while node is not None:
            node.visits += 1
            # Running average: incorporate the new reward
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    # --- Trajectory Methods ---

    def get_messages(self, include_reflections: bool = True) -> list[BaseMessage]:
        """
        Get this node's messages, optionally including the reflection.

        Messages typically look like:
          [AIMessage("candidate SQL + analysis"), HumanMessage("planner result")]

        If include_reflections=True, we also append the reflection as a message
        so the LLM can see how this attempt was evaluated.
        """
        if include_reflections and self.reflection:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """
        Get the full message history from root to this node.

        This is the KEY method for the LATS feedback loop:
        - Walk from this node up to the root
        - Collect all messages + reflections along the way
        - Reverse to get chronological order

        The resulting trajectory shows the LLM:
          1. Original error + first attempt
          2. Planner feedback on first attempt
          3. Reflection on what went wrong
          4. Second attempt (informed by the feedback)
          5. ... and so on

        This is how the agent "learns" from previous errors across iterations.
        """
        messages = []
        node = self

        # Walk up to the root, collecting messages in reverse
        while node is not None:
            node_messages = node.get_messages(include_reflections=include_reflections)
            messages.extend(node_messages[::-1])  # Reverse each node's messages
            node = node.parent

        # Reverse the full list to get chronological order
        # Result: [root messages, child1 messages, child2 messages, ...]
        return messages[::-1]

    # --- Solution Finding ---

    def get_best_solution(self) -> "Node":
        """
        Find the best solution in this subtree.

        Searches all descendants and returns the node with the highest
        value that is both terminal (a leaf) and solved (planner passed).

        If no solved node exists, returns the highest-scoring terminal node
        as a best-effort result.
        """
        all_nodes = [self] + self._get_all_children()

        # First, try to find a solved terminal node
        best_node = max(
            all_nodes,
            key=lambda n: int(n.is_terminal and n.is_solved) * n.value,
        )

        # If best_node has value 0 and isn't solved, fall back to best terminal
        if not best_node.is_solved:
            terminal_nodes = [n for n in all_nodes if n.is_terminal]
            if terminal_nodes:
                best_node = max(terminal_nodes, key=lambda n: n.value)

        return best_node

    # --- Internal Helpers ---

    def _get_all_children(self) -> list["Node"]:
        """BFS to collect all descendants of this node."""
        all_nodes = []
        queue = deque()
        queue.append(self)

        while queue:
            node = queue.popleft()
            all_nodes.extend(node.children)
            for child in node.children:
                queue.append(child)

        return all_nodes

    def _mark_tree_as_solved(self):
        """When a solution is found, mark all ancestors as solved too."""
        parent = self.parent
        while parent is not None:
            parent._is_solved = True
            parent = parent.parent


# ---------------------------------------------------------------------------
# TreeState: The top-level state for the LangGraph StateGraph
# ---------------------------------------------------------------------------

class TreeState(TypedDict):
    """
    The state object that flows through the LangGraph StateGraph.

    This is what gets passed between the "start" and "expand" nodes
    in the graph. It holds everything the LATS algorithm needs:

      - root: The root of the search tree (all nodes are reachable from here)
      - input: The full conversion prompt (SQL + error + context)
      - original_sql: The failing SQL (used for semantic equivalence checks)
      - error_message: The execution error (used in prompts)
      - docs: Retrieved e6data documentation (fetched once, shared across nodes)
      - calcite_idioms: Apache Calcite reserved keywords text
      - catalog: E6Data catalog name (for planner validation)
      - database: E6Data database/schema name (for planner validation)
    """
    root: Node
    input: str
    original_sql: str
    error_message: str
    docs: list
    calcite_idioms: str
    catalog: str
    database: str