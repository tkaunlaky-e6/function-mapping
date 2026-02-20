# LATS Dialect Mapper

Maps Spark SQL functions to e6data by executing basis queries and using MCTS tree search for translations that produce identical output.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical design with algorithm details.

## Overview

```
  LATS Dialect Mapping Pipeline
  ==============================

  Input                              Output
  -----                              ------
  basis_table.csv                    e6_coverage.csv
  basis_analysis.json                e6_mapping.json
       |                                  ^
       v                                  |
  +------------------------------------------+
  |           run_coverage_test()            |
  |                                          |
  |  +------------------------------------+  |
  |  | Phase 1: DIRECT EXECUTION         |  |
  |  |                                    |  |
  |  | for each basis query:             |  |
  |  |   run Spark SQL on e6data as-is   |  |
  |  |   compare output with expected    |  |
  |  |                                    |  |
  |  |   MATCH  -> covered=true, done    |  |
  |  |   FAIL   -> send to Phase 2       |  |
  |  +------------------------------------+  |
  |                  |                       |
  |            failed queries                |
  |                  |                       |
  |                  v                       |
  |  +------------------------------------+  |
  |  | Phase 2: LATS TREE SEARCH         |  |
  |  |          (only if --lats flag)     |  |
  |  |                                    |  |
  |  | for each failed query:            |  |
  |  |   build domain context            |  |
  |  |   build cumulative context        |  |
  |  |   run_lats() -> MCTS search       |  |
  |  |     generate candidate SQL        |  |
  |  |     execute on e6data             |  |
  |  |     score deterministically       |  |
  |  |     expand tree (UCB + reflect)   |  |
  |  |   -> best_sql, is_solved          |  |
  |  +------------------------------------+  |
  |                  |                       |
  |            all results                   |
  |                  |                       |
  |                  v                       |
  |  +------------------------------------+  |
  |  | Phase 3: COVERAGE ANALYSIS        |  |
  |  |                                    |  |
  |  | build_coverage_matrix()           |  |
  |  |   group queries by e6 function    |  |
  |  |   -> {ABS: [1,2,3], CASE: [4]}   |  |
  |  |                                    |  |
  |  | greedy_set_cover()                |  |
  |  |   pick func with most coverage    |  |
  |  |   repeat until all covered        |  |
  |  |   -> minimal function set         |  |
  |  |                                    |  |
  |  | compute partition scores          |  |
  |  | compute overall coverage %        |  |
  |  +------------------------------------+  |
  |                  |                       |
  |                  v                       |
  |         _write_report()                  |
  +------------------------------------------+
```

## Files

### `state.py` — MCTS Data Model

Pure data classes. No I/O, no LLM calls.

```
  +------------------+         +------------------+
  |   Reflection     |         |     Node         |
  |------------------|         |------------------|
  | error_analysis   |    +--->| messages         |
  | reflections      |    |    | reflection ------+---> Reflection
  | score (0.0-1.0)  |    |    | parent           |
  | total_queries    |    |    | children[]       |
  | found_solution   |    |    | value            |
  | planner_success  |    |    | visits           |
  | planner_error    |    |    | depth            |
  | preserves_intent |    |    | _is_solved       |
  |------------------|    |    |------------------|
  | normalized_score |    |    | upper_confidence_bound()
  |  = score /       |    |    | backpropagate()  |
  |    total_queries |    |    | get_trajectory() |
  | as_message()     |    |    | get_best_solution()
  +------------------+    |    | height           |
                          |    | is_solved        |
  +------------------+    |    +------------------+
  |   TreeState      |    |
  |  (TypedDict)     |    |
  |------------------|    |
  | root  -----------+----+
  | input            |
  | original_sql     |
  | error_message    |
  | docs             |
  | calcite_idioms   |
  | catalog          |
  | database         |
  |                  |
  | _planner         |  (injected at runtime)
  | _doc_searcher    |
  | _semantic_checker|
  | _num_candidates  |
  | _total_queries   |
  +------------------+
```

**Reflection fields**:
- `score`: float 0.0-1.0. Set deterministically: 1.0 (output match), 0.5 (mismatch), 0.2 (error)
- `total_queries`: total basis queries for this function. Used for normalization.
- `normalized_score`: `score / total_queries`. Used in UCB calculation.
- `found_solution`: True only when output matches expected.

**Node methods**:
- `upper_confidence_bound(C=1.0)`: `avg_reward + C * sqrt(ln(parent.visits) / visits)`
- `backpropagate(reward)`: walks up to root, updates running average
- `get_trajectory()`: collects messages from leaf to root, reverses to chronological order
- `get_best_solution()`: BFS over all nodes, prefers solved+terminal with highest value

### `prompts.py` — Prompt Templates and Context Formatters

```
  Prompt Templates
  =================

  GENERATION_PROMPT                    REFLECTION_PROMPT
  (Spark -> e6data translation)        (Analyze a translation attempt)
  +---------------------------+        +---------------------------+
  | SPARK SQL: {original_sql} |        | ORIGINAL: {original_sql}  |
  | PREV RESULT: {error_msg}  |        | CANDIDATE: {candidate}    |
  | TRAJECTORY: {history}     |        | PLANNER: {result}         |
  | E6 DOCS: {docs}           |        | SEMANTIC: {check}         |
  | DOMAIN CONTEXT: {idioms}  |        +---------------------------+
  +---------------------------+        Output: {reflections, score,
  Output: {error_analysis,                      found_solution}
           fix_strategy,               (score overridden by system)
           sql_query}

  SEMANTIC_CHECK_PROMPT
  (Does candidate preserve intent?)
  +---------------------------+
  | ORIGINAL: {original_sql}  |
  | CANDIDATE: {candidate}    |
  +---------------------------+
  Output: {preserves_intent,
           differences}

  Context Formatters
  ==================

  format_trajectory_context(messages)
    -> "== PREVIOUS ATTEMPTS =="
    -> Step 1: AI: ... | Human: ...
    -> Step 2: AI: ... | Human: ...
    (truncated at 1500 chars each)

  format_translation_context(analysis, query_row, partition_queries)
    -> Function name, partition being translated
    -> Expected output
    -> All partitions with THIS PARTITION marker
    -> Output range, codomain classes
    -> Other queries in same partition

  format_cumulative_context(results_so_far)
    -> Pipe-delimited table:
    -> id | partition | spark_query | expected | e6_query | actual | status | e6_function
    -> Shows PASS/FAIL for all queries processed so far
```

### `tools.py` — Pluggable Tool Interfaces

```
  Tool Class Hierarchy
  =====================

  PlannerValidator (STUB)
    |  validate(sql) -> {success: True, error: ""}
    |  Always passes. For testing.
    |
    +-- RealPlannerValidator
    |     validate(sql) -> {success: bool, error: str}
    |     Connects to e6data get_logical_plan() (no-execute validation)
    |     Reads credentials from .env via os.getenv()
    |
    +-- QueryExecutor
          validate(sql) -> {success: bool, error: str}
          EXECUTES SQL on e6data, compares output with expected
          execute(sql) -> raw rows
          execute_and_compare(sql, expected) -> {success, error, actual, expected, match}
          Reads credentials from .env via os.getenv()

  DocSearcher (STUB)
    search(terms) -> [{content, title}, ...]
    Returns 3 hardcoded e6data tips (reserved keywords, date functions, unsupported features)

  SemanticChecker
    check(original, candidate) -> {preserves_intent, differences}
    With LLM: uses SEMANTIC_CHECK_PROMPT
    Without LLM: basic length + FROM-table comparison
```

### `nodes.py` — LangGraph Node Functions

```
  LATS Graph Nodes
  ==================

  generate_initial_response(state, llm) -> TreeState
    |
    |  1. Search docs if needed (DocSearcher)
    |  2. Generate candidate SQL (LLM + GENERATION_PROMPT)
    |  3. Parse JSON response: {error_analysis, fix_strategy, sql_query}
    |  4. Validate: planner.validate(candidate_sql)
    |  5. Semantic check: checker.check(original, candidate)
    |  6. Score deterministically:
    |       planner success -> 1.0, solved
    |       "mismatch" in error -> 0.5
    |       execution error -> 0.2
    |  7. Create root Node(messages, reflection)
    |
    +-> returns state with root

  expand(state, llm) -> TreeState
    |
    |  1. select(root) -> best leaf via UCB1
    |  2. best_leaf.get_trajectory() -> full message history
    |  3. for i in range(num_candidates):  [default 3]
    |       generate candidate (sees full trajectory)
    |       validate on e6data
    |       semantic check
    |       score deterministically
    |       create child Node(parent=best_leaf)
    |       backpropagate score up the tree
    |  4. Early exit if any child is_solved
    |
    +-> returns state (tree modified in-place)

  should_loop(state) -> "expand" | "__end__"
    |
    |  root.is_solved? -> "__end__"
    |  root.height >= 3? -> "__end__"
    |  otherwise -> "expand"
```

### `graph.py` — LangGraph StateGraph Wiring

```
  StateGraph Wiring
  ==================

  create_lats_graph(llm, planner, doc_searcher, semantic_checker, num_candidates)
    |
    |  Builds:
    |
    |  +-------+     +-----------+     +---------+
    |  | START |---->|  "start"  |---->| should  |
    |  +-------+     | (generate)|     | _loop() |
    |                +-----------+     +----+----+
    |                                       |
    |                      +----------------+----------------+
    |                      |                                 |
    |                      v                                 v
    |                +----------+                       +--------+
    |                | "expand" |-----> should_loop ---->|  END   |
    |                +----+-----+          |            +--------+
    |                     |                |
    |                     +---<--"expand"--+
    |
    |  _inject_tools(state): adds _planner, _doc_searcher,
    |    _semantic_checker, _num_candidates to state dict

  run_lats(llm, original_sql, error_message, ..., total_queries=1) -> dict
    |
    |  1. Creates graph
    |  2. Builds initial_state with _total_queries
    |  3. graph.invoke(initial_state)
    |  4. root.get_best_solution() -> best node
    |  5. Parse best SQL from node messages
    |  6. Returns {best_sql, is_solved, score, iterations, root}
```

### `pipeline.py` — End-to-End Orchestration

```
  Pipeline Functions
  ===================

  read_basis(func_name)
    -> reads output/<func>/basis_table.csv (csv.DictReader)
    -> reads basis_analysis_new.json or basis_analysis.json
    -> returns (rows, analysis)

  group_by_partition(rows)
    -> groups by input_partition field
    -> returns defaultdict(list)

  extract_e6_function(sql)
    -> "SELECT " stripped
    -> "CASE " -> "CASE_EXPRESSION"
    -> regex /([A-Za-z_]\w*)\s*\(/ -> uppercase
    -> "CAST(" -> "CAST"
    -> else "DIRECT"

  build_coverage_matrix(results)
    -> groups covered queries by e6_function
    -> {func: {covered_ids: [int], count: int}}

  greedy_set_cover(matrix, total_ids)
    -> repeatedly pick func covering most remaining IDs
    -> returns ordered list of function names

  run_coverage_test(func_name, catalog, database, use_stub, use_lats, model_name)
    -> Phase 1: direct execution
    -> Phase 2: LATS for failures (if --lats)
    -> Phase 3: coverage matrix + set cover
    -> _write_report() -> e6_coverage.csv + e6_mapping.json
    -> returns full report dict
```

## Scoring

Deterministic, based on execution result — not LLM judgment:

| Condition | Score | found_solution | Action |
|---|---|---|---|
| Output matches expected | 1.0 | True | Stop searching |
| Output mismatch | 0.5 | False | Keep searching |
| Execution error | 0.2 | False | Keep searching |

`normalized_score = score / total_queries` for UCB. A function with 20 basis queries normalizes each matching query to 0.05.

Max tree depth: 3. Max candidates per expansion: 3. Max nodes per query: 7.

## Usage

```bash
# Stub mode (no cluster, all queries pass):
python -m dialect_mapper.lats.pipeline ABS --stub

# Real e6data execution:
python -m dialect_mapper.lats.pipeline ABS

# With LATS translation for failures:
python -m dialect_mapper.lats.pipeline ABS --lats

# Specify model:
python -m dialect_mapper.lats.pipeline ABS --lats --model claude-sonnet-4-5-20250929

# Verbose logging:
python -m dialect_mapper.lats.pipeline ABS --lats -v
```