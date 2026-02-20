"""
Prompt templates for the LATS dialect mapper.

This is a DIALECT MAPPER: Spark SQL → e6data SQL.
The ground truth is EXECUTION — does the e6data SQL produce the same output?

Two prompts:
  1. GENERATION_PROMPT: Translate Spark SQL → e6data SQL
  2. REFLECTION_PROMPT: Analyze why a translation attempt failed

Scoring is deterministic (set in nodes.py, NOT by LLM):
  - Output matched  → score=10, normalized=1.0, found_solution=True → STOP
  - Output mismatch → score=5
  - Execution error  → score=2
The LLM reflection only provides the reasoning text, not the score.
"""

from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# 1. Generation Prompt — Spark → e6data translation
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are a SQL dialect mapper. Translate the Spark SQL below
into e6data SQL that produces the EXACT SAME output.

== SPARK SQL ==
{original_sql}

== PREVIOUS EXECUTION RESULT ==
{error_message}

{trajectory_context}

== E6DATA DOCUMENTATION ==
{docs}

== TRANSLATION CONTEXT ==
{calcite_idioms}

== RULES ==
1. Your e6data SQL must produce EXACTLY the same output as the Spark query.
2. e6data uses Apache Calcite SQL. Use e6data-compatible functions only.
3. If the Spark function does not exist in e6data, find an equivalent
   (CASE expressions, CAST, arithmetic, other built-ins).
4. If the previous attempt returned the wrong value, analyze WHY and adjust.
5. If the previous attempt threw an error, read the error and fix it.
6. Keep it simple — SELECT <expression> is fine.

== OUTPUT FORMAT ==
Return ONLY valid JSON:
{{
  "error_analysis": "Why the previous attempt failed (or 'First attempt')",
  "fix_strategy": "What e6data function/expression you chose and why",
  "sql_query": "The e6data SQL query"
}}"""


# ---------------------------------------------------------------------------
# 2. Reflection Prompt — analyze why translation failed
# ---------------------------------------------------------------------------

REFLECTION_PROMPT = """A Spark → e6data SQL translation was attempted. Analyze the result.

== SPARK QUERY ==
{original_sql}

== EXECUTION RESULT ==
{error_message}

== E6DATA TRANSLATION ==
{candidate_sql}

== STRATEGY ==
{error_analysis}
{fix_strategy}

== E6DATA EXECUTION RESULT ==
{planner_result}

{semantic_result}

== TASK ==
Explain why this translation did or didn't produce the correct output.
If it failed, suggest what to try differently next time.

Return ONLY valid JSON:
{{
  "reflections": "Why this worked or didn't work, and what to try next",
  "score": 0,
  "found_solution": false
}}

NOTE: score and found_solution are overridden by the system based on actual
execution results. Just provide your reasoning in "reflections"."""


# ---------------------------------------------------------------------------
# 3. Semantic Check Prompt (kept for compatibility)
# ---------------------------------------------------------------------------

SEMANTIC_CHECK_PROMPT = """Does this e6data query produce the same result as the Spark query?

== SPARK SQL ==
{original_sql}

== E6DATA SQL ==
{candidate_sql}

Return ONLY valid JSON:
{{
  "preserves_intent": true,
  "differences": "None"
}}"""


# ---------------------------------------------------------------------------
# Helper: Format trajectory context for the generation prompt
# ---------------------------------------------------------------------------

def format_trajectory_context(trajectory_messages: list) -> str:
    """
    Format previous attempts into a string for the generation prompt.
    """
    if not trajectory_messages:
        return ""

    lines = ["== PREVIOUS ATTEMPTS (learn from these) =="]

    for i, msg in enumerate(trajectory_messages):
        role = "Assistant" if isinstance(msg, AIMessage) else "Feedback"
        content = msg.content[:1500] if len(msg.content) > 1500 else msg.content
        lines.append(f"\n[{role} - Step {i + 1}]")
        lines.append(content)

    lines.append("\n== END PREVIOUS ATTEMPTS ==")
    lines.append("Use the errors and feedback above to avoid repeating mistakes.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: Format translation context for basis query mapping
# ---------------------------------------------------------------------------

def format_translation_context(
    analysis: dict,
    query_row: dict,
    partition_queries: list[dict] = None,
) -> str:
    """
    Format domain analysis + partition context for LATS translation.
    """
    func_name = analysis.get("function", analysis.get("function_name", "UNKNOWN"))
    partition = query_row.get("input_partition", "")
    expected = query_row.get("actual_output", "")

    lines = [
        f"== SPARK → E6DATA FUNCTION TRANSLATION ==",
        f"Spark function: {func_name}",
        f"",
        f"Query being translated: {query_row.get('query', '')}",
        f"Input partition: {partition}",
        f"Expected output: {expected}",
        f"Codomain class: {query_row.get('codomain_class', '')}",
        f"Partition reasoning: {query_row.get('reasoning', '')}",
        f"",
        f"CRITICAL: Your e6data SQL MUST produce exactly this output: {expected}",
        f"The query will be EXECUTED on e6data and the output compared.",
    ]

    if partition_queries and len(partition_queries) > 1:
        lines.append(f"")
        lines.append(f"== ALL QUERIES IN PARTITION '{partition}' ==")
        lines.append(f"Your translation strategy must work for ALL of these:")
        for q in partition_queries:
            lines.append(f"  {q.get('query', '')}  →  expected: {q.get('actual_output', '')}")

    lines.append(f"")
    lines.append(f"== FUNCTION DOMAIN ==")

    params = analysis.get("parameter_analysis", {})
    for param_name, param_info in params.items():
        lines.append(f"Parameter '{param_name}': {param_info.get('real_domain', '')}")
        for p in param_info.get("partitions", []):
            name = p.get("name", p.get("partition", ""))
            pred = p.get("predicate", "")
            reason = p.get("reasoning", p.get("behavior", ""))
            marker = " ← THIS PARTITION" if name == partition else ""
            lines.append(f"  {name}: {pred} → {reason}{marker}")

    output = analysis.get("output_analysis", analysis.get("output_range", {}))
    if output:
        lines.append(f"")
        range_str = output.get("range", output.get("type", ""))
        lines.append(f"Output range: {range_str}")
        for prop in output.get("special_properties", output.get("properties", [])):
            lines.append(f"  - {prop}")

    codomain = output.get("codomain_classes", analysis.get("codomain_classes", []))
    if codomain:
        lines.append(f"")
        lines.append(f"Codomain classes:")
        for cls in codomain:
            name = cls.get("name", cls.get("class", ""))
            desc = cls.get("description", cls.get("condition", ""))
            marker = " ← THIS QUERY" if name == query_row.get("codomain_class", "") else ""
            lines.append(f"  {name}: {desc}{marker}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: Format cumulative context across partitions (CSV-like)
# ---------------------------------------------------------------------------

def format_cumulative_context(results_so_far: list[dict]) -> str:
    """
    Build a CSV-like table of results from all partitions processed so far.
    """
    if not results_so_far:
        return ""

    lines = [
        "== TRANSLATION RESULTS SO FAR ==",
        "id | input_partition | spark_query | expected | e6_query | actual | status | e6_function",
        "---|----------------|-------------|----------|---------|--------|--------|------------",
    ]

    for r in results_so_far:
        status = "PASS" if r.get("covered") else "FAIL"
        e6_q = r.get("e6_query", "") or r.get("spark_query", "")
        actual = r.get("actual_output", "")
        error = r.get("direct_error", "")

        if not r.get("covered") and error:
            actual = error[:60]

        lines.append(
            f"{r.get('id', '')} | "
            f"{r.get('input_partition', '')} | "
            f"{r.get('spark_query', '')[:45]} | "
            f"{r.get('expected_output', '')} | "
            f"{e6_q[:45]} | "
            f"{actual} | "
            f"{status} | "
            f"{r.get('e6_function', '')}"
        )

    total = len(results_so_far)
    passed = sum(1 for r in results_so_far if r.get("covered"))
    lines.append(f"")
    lines.append(f"Coverage so far: {passed}/{total} queries pass")
    lines.append("== END TRANSLATION RESULTS ==")

    return "\n".join(lines)