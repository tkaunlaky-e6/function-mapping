# LATS Dialect Mapper — Architecture

## What It Does

Maps Spark SQL functions to e6data SQL by finding the minimal set of e6 functions that produce identical output for every basis query.

**Input**: `basis_table.csv` (Spark queries + expected outputs) + `basis_analysis.json` (domain partitions, codomain classes, spanning proofs).

**Output**: `e6_mapping.json` with the mapped SQL, coverage matrix, and minimal function set.

## End-to-End Data Flow

```
  +----------------------+     +----------------------+
  | basis_table.csv      |     | basis_analysis.json  |
  | id,query,partition,  |     | partitions, codomain |
  |   actual_output,...  |     | spanning proofs      |
  +---------+------------+     +---------+------------+
            |                            |
            +----------+-----------------+
                       |
                       v
  +-------------------------------------------------+
  |            run_coverage_test()                   |
  |            (pipeline.py)                         |
  |                                                  |
  |  read_basis() -> rows[], analysis{}              |
  |  group_by_partition() -> {partition: [rows]}     |
  |                                                  |
  |  +-----------  Phase 1  ----------------------+  |
  |  |  DIRECT EXECUTION                         |  |
  |  |                                            |  |
  |  |  for partition in partitions:              |  |
  |  |    for query in partition:                 |  |
  |  |                                            |  |
  |  |    +----------------------------------+    |  |
  |  |    | QueryExecutor                    |    |  |
  |  |    | .execute_and_compare(            |    |  |
  |  |    |    sql="SELECT ABS(-5)",         |    |  |
  |  |    |    expected="5"                  |    |  |
  |  |    | )                                |    |  |
  |  |    +--------+-------+--------+--------+    |  |
  |  |             |       |        |             |  |
  |  |          MATCH   MISMATCH  ERROR           |  |
  |  |             |       |        |             |  |
  |  |             v       v        v             |  |
  |  |         covered  +-----------+             |  |
  |  |         = true   | FAILED    |             |  |
  |  |                  | queue for |             |  |
  |  |                  | Phase 2   |             |  |
  |  |                  +-----------+             |  |
  |  +--------------------------------------------+  |
  |                       |                          |
  |                 failed queries                   |
  |                       |                          |
  |                       v                          |
  |  +-----------  Phase 2  ----------------------+  |
  |  |  LATS TREE SEARCH (if --lats)              |  |
  |  |                                            |  |
  |  |  for each failed query:                    |  |
  |  |                                            |  |
  |  |  format_translation_context()              |  |
  |  |    -> domain analysis for THIS partition   |  |
  |  |    -> expected output                      |  |
  |  |    -> all partitions (with marker)         |  |
  |  |    -> codomain classes                     |  |
  |  |                                            |  |
  |  |  format_cumulative_context()               |  |
  |  |    -> table of all results so far          |  |
  |  |    -> PASS/FAIL status per query           |  |
  |  |                                            |  |
  |  |  run_lats()                                |  |
  |  |    |                                       |  |
  |  |    v  (see MCTS detail below)              |  |
  |  |    -> {best_sql, is_solved, score}         |  |
  |  |                                            |  |
  |  |  update entry:                             |  |
  |  |    e6_query = best_sql                     |  |
  |  |    e6_function = extract_e6_function(sql)  |  |
  |  |    covered = is_solved                     |  |
  |  +--------------------------------------------+  |
  |                       |                          |
  |                 all results                      |
  |                       |                          |
  |                       v                          |
  |  +-----------  Phase 3  ----------------------+  |
  |  |  COVERAGE ANALYSIS                        |  |
  |  |                                            |  |
  |  |  build_coverage_matrix(results)            |  |
  |  |    -> {                                    |  |
  |  |         "ABS":  {ids: [1,2,3,5,6], n: 5}, |  |
  |  |         "CASE": {ids: [4,7,8],     n: 3}  |  |
  |  |       }                                    |  |
  |  |                                            |  |
  |  |  greedy_set_cover(matrix, all_ids)         |  |
  |  |    1. Pick func with most coverage: ABS(5) |  |
  |  |    2. Remove covered IDs                   |  |
  |  |    3. Pick next: CASE(3)                   |  |
  |  |    4. All covered -> done                  |  |
  |  |    -> ["ABS", "CASE_EXPRESSION"]           |  |
  |  |                                            |  |
  |  |  overall_coverage = covered / total        |  |
  |  |  partition_scores per partition             |  |
  |  +--------------------------------------------+  |
  |                       |                          |
  |                       v                          |
  |  _write_report()                                 |
  |    -> e6_coverage.csv  (13 columns per query)    |
  |    -> e6_mapping.json  (translations + matrix)   |
  +-------------------------------------------------+
```

## MCTS Tree Search — Detailed

```
  run_lats() -> create_lats_graph() -> graph.invoke()
  =====================================================

  StateGraph:
    START --> "start" --> should_loop --> "expand" --> should_loop --> ...
                                    |                            |
                                    +--> END                     +--> END

  "start" node: generate_initial_response()
  ==========================================

  +--------------------------------------------------+
  |  1. DocSearcher.search(error_terms)               |
  |     -> e6data dialect tips (reserved keywords,    |
  |        date functions, unsupported features)      |
  |                                                   |
  |  2. Build GENERATION_PROMPT:                      |
  |     - Original Spark SQL                          |
  |     - Previous execution error                    |
  |     - e6data documentation                        |
  |     - Domain context (from basis_analysis.json)   |
  |                                                   |
  |  3. LLM generates candidate:                     |
  |     {                                             |
  |       "error_analysis": "ABS(NULL) returns 0...", |
  |       "fix_strategy": "Wrap in CASE WHEN...",     |
  |       "sql_query": "SELECT CASE WHEN..."          |
  |     }                                             |
  |                                                   |
  |  4. Validate candidate:                           |
  |     executor.validate(candidate_sql)              |
  |       -> execute on e6data                        |
  |       -> compare output with expected             |
  |       -> {success: bool, error: str}              |
  |                                                   |
  |  5. Semantic check:                               |
  |     checker.check(original_sql, candidate_sql)    |
  |       -> {preserves_intent: bool, differences}    |
  |                                                   |
  |  6. Deterministic scoring:                        |
  |     +---------------------+-------+----------+   |
  |     | Condition           | Score | Solved?  |   |
  |     +---------------------+-------+----------+   |
  |     | Output matches      |  1.0  |  true    |   |
  |     | Output mismatch     |  0.5  |  false   |   |
  |     | Execution error     |  0.2  |  false   |   |
  |     +---------------------+-------+----------+   |
  |                                                   |
  |  7. LLM reflection (reasoning text only):         |
  |     REFLECTION_PROMPT -> {reflections: "..."}     |
  |     (score + found_solution overridden by system) |
  |                                                   |
  |  8. Create root Node:                             |
  |     Node(messages=[AI+Human], reflection)         |
  |     backpropagate(normalized_score)               |
  +--------------------------------------------------+

  "expand" node: expand()
  ========================

  +--------------------------------------------------+
  |  1. UCB1 LEAF SELECTION                           |
  |                                                   |
  |     select(root):                                 |
  |       current = root                              |
  |       while current has children:                 |
  |         current = argmax(child.ucb())             |
  |       return current  (best leaf)                 |
  |                                                   |
  |     UCB formula:                                  |
  |       ucb = avg_value + C * sqrt(ln(P.visits)     |
  |                                  / visits)        |
  |                                                   |
  |  2. GET FULL TRAJECTORY                           |
  |                                                   |
  |     best_leaf.get_trajectory():                   |
  |       walk from leaf -> root                      |
  |       collect all messages                        |
  |       reverse to chronological order              |
  |       -> [root_msgs, child1_msgs, leaf_msgs]      |
  |                                                   |
  |     This is how the LLM sees its full history:    |
  |       "I tried X (score 0.2), then Y (0.5),      |
  |        now try something better"                  |
  |                                                   |
  |  3. GENERATE num_candidates NEW CHILDREN          |
  |                                                   |
  |     for i in range(3):  [default]                 |
  |       LLM(GENERATION_PROMPT + trajectory)         |
  |       -> candidate_sql                            |
  |       executor.validate(candidate_sql)            |
  |       semantic_check()                            |
  |       deterministic_score()                       |
  |       reflection()                                |
  |       child = Node(parent=best_leaf)              |
  |       backpropagate(child.normalized_score)       |
  |                                                   |
  |       if child.is_solved: break early             |
  |                                                   |
  |  Tree after expand (depth 1):                     |
  |                                                   |
  |         [Root] depth=0                            |
  |        /  |  \                                    |
  |     [C1] [C2] [C3]  depth=1                      |
  |                                                   |
  +--------------------------------------------------+

  should_loop() termination check
  ================================

  +--------------------------------------------------+
  |  root.is_solved?                                  |
  |    -> YES: return "__end__"                       |
  |       (found a candidate with matching output)    |
  |                                                   |
  |  root.height >= 3?                                |
  |    -> YES: return "__end__"                       |
  |       (max depth reached, return best effort)     |
  |                                                   |
  |  Otherwise:                                       |
  |    -> return "expand" (keep searching)            |
  +--------------------------------------------------+

  WORST CASE TREE (depth 3, 3 candidates per expand):
  ===================================================

  Depth 0:   [Root]             1 node,  2 LLM calls
                                         (generate + reflect)
  Depth 1:   [C1] [C2] [C3]    3 nodes, 6 LLM calls

  Depth 2:   [C4] [C5] [C6]    3 nodes, 6 LLM calls

  Total: 7 nodes, 14 LLM calls per failed query
```

## Scoring System

```
  Scoring Flow
  =============

  executor.validate(candidate_sql)
       |
       +-- success=true (output matches expected)
       |     score = 1.0
       |     found_solution = true
       |     -> STOP SEARCHING
       |
       +-- success=false, "mismatch" in error
       |     score = 0.5
       |     found_solution = false
       |     -> KEEP SEARCHING (output was wrong but SQL ran)
       |
       +-- success=false, execution error
             score = 0.2
             found_solution = false
             -> KEEP SEARCHING (SQL didn't even run)

  Normalization for UCB:
    normalized_score = score / total_queries

    Example: ABS has 8 basis queries
      matching query:  1.0 / 8 = 0.125
      mismatch query:  0.5 / 8 = 0.0625
      error query:     0.2 / 8 = 0.025
```

## Greedy Set Cover Algorithm

```
  Input: coverage_matrix, all_query_ids
  =======================================

  coverage_matrix = {
    "ABS":            {ids: {1, 2, 3, 4, 5, 6, 8}, count: 7},
    "CASE_EXPRESSION": {ids: {4, 7},                 count: 2},
    "DIRECT":         {ids: {8},                     count: 1},
  }
  all_query_ids = {1, 2, 3, 4, 5, 6, 7, 8}

  Iteration 1:
    ABS covers 7 remaining -> pick ABS
    remaining = {7}

  Iteration 2:
    CASE covers 1 remaining (id 7) -> pick CASE
    remaining = {}

  Result: minimal_function_set = ["ABS", "CASE_EXPRESSION"]
  overall_coverage = 8/8 = 1.0
```

## File Dependency Chain

```
  How files call each other
  ==========================

  pipeline.py
    |
    |-- read_basis()         reads output/<func>/ CSV + JSON
    |-- group_by_partition() pure data grouping
    |-- extract_e6_function() regex extraction
    |
    |-- tools.py
    |   |-- QueryExecutor    executes SQL on e6data
    |   |-- PlannerValidator stub (for --stub mode)
    |
    |-- prompts.py
    |   |-- format_translation_context()   domain context for LLM
    |   |-- format_cumulative_context()    results-so-far table
    |
    |-- graph.py
    |   |-- run_lats()       convenience wrapper
    |   |-- create_lats_graph()
    |   |   |
    |   |   |-- nodes.py
    |   |   |   |-- generate_initial_response()
    |   |   |   |-- expand()
    |   |   |   |-- should_loop()
    |   |   |   |-- _generate_and_validate_candidate()
    |   |   |   |-- _reflect_on_candidate()
    |   |   |   |-- _parse_json_response()
    |   |   |   |-- select()           UCB1 leaf selection
    |   |   |   |
    |   |   |   |-- state.py
    |   |   |   |   |-- Node           MCTS tree node
    |   |   |   |   |-- Reflection     per-candidate evaluation
    |   |   |   |   |-- TreeState      LangGraph state dict
    |   |   |   |
    |   |   |   |-- prompts.py
    |   |   |   |   |-- GENERATION_PROMPT
    |   |   |   |   |-- REFLECTION_PROMPT
    |   |   |   |   |-- format_trajectory_context()
    |   |   |   |
    |   |   |   |-- tools.py
    |   |   |       |-- PlannerValidator.validate()
    |   |   |       |-- DocSearcher.search()
    |   |   |       |-- SemanticChecker.check()
    |   |   |
    |   |   |-- langgraph.graph.StateGraph
    |   |
    |   |-- langchain_anthropic.ChatAnthropic
    |
    |-- _write_report()      writes e6_coverage.csv + e6_mapping.json
```

## Handling Dialect Differences

The system discovers dialect differences through execution, not hardcoded rules:

```
  Example: FACTORIAL(-5)
  =======================

  Spark:  SELECT FACTORIAL(-5)  ->  NULL
  e6data: SELECT FACTORIAL(-5)  ->  IllegalArgumentException

  Phase 1 (Direct):
    executor.execute_and_compare("SELECT FACTORIAL(-5)", expected="NULL")
    -> e6data throws error
    -> result: {success: false, error: "Execution error: IllegalArgumentException"}

  Phase 2 (LATS):
    LLM sees: "Spark returns NULL for negative, e6data throws"
    LLM generates: "SELECT CASE WHEN -5 < 0 THEN NULL ELSE FACTORIAL(-5) END"
    executor.validate(candidate)
    -> e6data returns NULL
    -> matches expected "NULL"
    -> score = 1.0, solved!

  Result:
    e6_function = "CASE_EXPRESSION"  (extracted from CASE WHEN ...)
    e6_query = "SELECT CASE WHEN -5 < 0 THEN NULL ELSE FACTORIAL(-5) END"
    covered = true
```

```
  Example: DATEDIFF parameter order
  ===================================

  Spark:  SELECT DATEDIFF('2024-01-10', '2024-01-01')  ->  9
  e6data: SELECT DATEDIFF('2024-01-10', '2024-01-01')  ->  -9  (reversed!)

  Phase 1 (Direct):
    executor.execute_and_compare(..., expected="9")
    -> e6data returns "-9"
    -> result: {success: false, error: "Output mismatch: expected '9', got '-9'"}

  Phase 2 (LATS):
    LLM sees: "expected 9, got -9" -> realizes parameters are reversed
    LLM generates: "SELECT DATEDIFF('2024-01-01', '2024-01-10')"
    executor.validate(candidate)
    -> e6data returns 9
    -> matches!
    -> score = 1.0, solved!
```

## Output Format

### e6_mapping.json

```json
{
  "spark_function": "FACTORIAL",
  "total_basis_queries": 13,
  "overall_coverage": 1.0,
  "minimal_function_set": ["FACTORIAL", "DIRECT"],
  "uncovered_partitions": [],
  "coverage_matrix": {
    "FACTORIAL": {"covered_queries": [1,3,4,5,6,7,8,9,10], "count": 9, "coverage": 0.69},
    "DIRECT":    {"covered_queries": [2,11,12,13],          "count": 4, "coverage": 0.31}
  },
  "translations": [
    {
      "id": 2,
      "spark_query": "SELECT FACTORIAL(-5)",
      "input_partition": "P_NEGATIVE",
      "e6_query": "SELECT CASE WHEN -5 < 0 THEN NULL ELSE FACTORIAL(-5) END",
      "e6_function": "DIRECT",
      "covered": true,
      "reasoning": "e6data throws on negative input; Spark returns NULL. CASE wrapper needed."
    }
  ]
}
```

### e6_coverage.csv

13 columns per query:
```
id, spark_query, input_partition, expected_output, actual_output,
codomain_class, direct_pass, direct_error, e6_function, e6_query,
lats_pass, lats_score, covered
```