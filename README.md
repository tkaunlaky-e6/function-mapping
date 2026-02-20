# SQL Dialect Function Mapper

Maps SQL functions between dialects (Spark SQL to e6data) using mathematical basis synthesis and execution-validated translation. The system discovers behavioral differences through actual execution, not hardcoded rules.

## System Overview

```
                         SQL Dialect Function Mapper
  ========================================================================

  Phase 1: BASIS SYNTHESIS                Phase 2: DIALECT MAPPING
  (What does the function do?)            (How to express it in e6data?)

  +--------------------------+            +----------------------------+
  |     Spark SQL Docs       |            |    basis_table.csv         |
  |  (fetched from Apache)   |            |    basis_analysis.json     |
  +-----------+--------------+            +-------------+--------------+
              |                                         |
              v                                         v
  +--------------------------+            +----------------------------+
  |    LLM Agent             |            |  Direct Execution          |
  |  (Claude Sonnet)         |            |  Run Spark SQL on e6data   |
  |                          |            |  Compare output            |
  |  1. Partition domain     |            +-------+----------+---------+
  |  2. Write spanning proof |                    |          |
  |  3. Execute on PySpark   |              PASS  |          | FAIL
  |  4. Derive codomain      |                    v          v
  |  5. Check surjectivity   |            +-------+--+  +----+----------+
  |  6. Back-solve gaps      |            | Record   |  | LATS Tree     |
  +-----------+--------------+            | as-is    |  | Search (MCTS) |
              |                           +-------+--+  |               |
              v                                   |     | UCB selection |
  +--------------------------+                    |     | LLM generate  |
  |  output/<func>/          |                    |     | Execute + cmp |
  |   basis_table.csv        |                    |     | Reflect       |
  |   basis_analysis.json    |                    |     +----+----------+
  +--------------------------+                    |          |
                                                  v          v
                                          +----------------------------+
                                          |  Coverage Analysis         |
                                          |  Build coverage matrix     |
                                          |  Greedy set cover          |
                                          |  Minimal e6 function set   |
                                          +-------------+--------------+
                                                        |
                                                        v
                                          +----------------------------+
                                          |  output/<func>/            |
                                          |   e6_coverage.csv          |
                                          |   e6_mapping.json          |
                                          +----------------------------+
```

## How It Works

### Phase 1: Basis Synthesis (`basis/`)

An LLM agent generates a mathematically complete set of test queries for any Spark SQL function. "Complete" means two things:

**Input spanning** — the input domain is partitioned into equivalence classes where each class represents a distinct behavior. For `ABS`: null, negative integer, zero, positive integer, negative decimal, positive decimal, min bigint (boundary), string cast. The union of all partitions equals the full domain (proven via a spanning proof).

**Output surjectivity** — every category of output the function can produce is hit by at least one query. For `ABS`: null output, zero, positive integer, positive decimal, large integer. If a codomain class is uncovered, the agent back-solves to find an input that produces it.

The agent executes each query on a local PySpark session to get ground-truth outputs, then writes:
- `basis_table.csv` — queries + expected outputs (6 columns)
- `basis_analysis.json` — domain analysis with formal proofs

```
  Basis Synthesis Data Flow
  =========================

  User: "python -m dialect_mapper.basis.agent ABS"
                        |
                        v
  +--------------------------------------------+
  |  run_basis_agent("ABS")                    |
  |                                            |
  |  1. _fetch_docs("ABS")                     |
  |     HTTP GET spark.apache.org/docs/...     |
  |     -> raw HTML (up to 15,000 chars)       |
  |                                            |
  |  2. create_basis_agent()                   |
  |     LangChain create_agent(                |
  |       model = Claude Sonnet,               |
  |       tools = [execute_spark_sql],         |
  |       middleware = [FilesystemMiddleware]   |
  |     )                                      |
  |     FilesystemMiddleware adds:             |
  |       read_file, write_file                |
  |                                            |
  |  3. agent.invoke(HUMAN_PROMPT)             |
  +-----+--------------------------------------+
        |
        v  (LLM agent loop)
  +--------------------------------------------+
  |  LLM reasons about input partitions        |
  |    |                                       |
  |    +-> execute_spark_sql("SELECT ABS(-5)") |
  |    |   -> PySpark -> "5"                   |
  |    |                                       |
  |    +-> execute_spark_sql("SELECT ABS(0)")  |
  |    |   -> PySpark -> "0"                   |
  |    |                                       |
  |    +-> ... (one query per partition)        |
  |    |                                       |
  |    +-> write_file("basis_table.csv", ...)  |
  |    +-> write_file("basis_analysis.json")   |
  +--------------------------------------------+
```

### Phase 2: Dialect Mapping (`lats/`)

The LATS pipeline reads the basis and maps each query to e6data in three steps:

**Step 1 — Direct execution**: Run each Spark query as-is on e6data. If the output matches the expected value from `basis_table.csv`, the query passes. No translation needed.

**Step 2 — LATS translation**: For failures, use Language Agent Tree Search (MCTS + LLM) to find an e6 SQL query that produces the same output. The search generates 3 candidates per expansion, up to depth 3 (max 7 nodes, 14 LLM calls per query).

**Step 3 — Coverage analysis**: Build a coverage matrix (which e6 functions cover which basis queries), then run greedy set cover to find the minimal set of e6 functions needed.

```
  LATS Pipeline Data Flow
  ========================

  basis_table.csv + basis_analysis.json
              |
              v
  +---------------------------------------------------+
  |  run_coverage_test("ABS")                          |
  |                                                    |
  |  Phase 1: Direct Execution                         |
  |  +-----------------------------------------------+ |
  |  | for each basis query:                         | |
  |  |   executor.execute_and_compare(               | |
  |  |     "SELECT ABS(-5)", expected="5"            | |
  |  |   )                                           | |
  |  |   -> e6data returns "5" -> MATCH -> PASS      | |
  |  |                                               | |
  |  |   executor.execute_and_compare(               | |
  |  |     "SELECT ABS(-9223372036854775808)",       | |
  |  |     expected="9223372036854775808"             | |
  |  |   )                                           | |
  |  |   -> e6data throws overflow -> FAIL           | |
  |  +-----------------------------------------------+ |
  |                                                    |
  |  Phase 2: LATS (for failures only, if --lats)      |
  |  +-----------------------------------------------+ |
  |  | for each failed query:                        | |
  |  |   format_translation_context(analysis, row)   | |
  |  |   format_cumulative_context(results_so_far)   | |
  |  |   run_lats(                                   | |
  |  |     original_sql, error, planner=executor     | |
  |  |   )                                           | |
  |  |   -> MCTS tree search (see below)             | |
  |  |   -> best_sql, is_solved, score               | |
  |  +-----------------------------------------------+ |
  |                                                    |
  |  Phase 3: Coverage                                 |
  |  +-----------------------------------------------+ |
  |  | build_coverage_matrix(results)                | |
  |  |   -> {ABS: [1,2,3,4,5,6,8], CASE: [7]}       | |
  |  | greedy_set_cover(matrix, all_ids)             | |
  |  |   -> ["ABS", "CASE_EXPRESSION"]               | |
  |  +-----------------------------------------------+ |
  |                                                    |
  |  _write_report()                                   |
  |   -> e6_coverage.csv                               |
  |   -> e6_mapping.json                               |
  +---------------------------------------------------+
```

### LATS Tree Search (MCTS)

```
  MCTS Tree Structure (per failed query)
  =======================================

  Depth 0:  [Root]  generate_initial_response()
             |      LLM generates candidate SQL
             |      Execute on e6data, compare output
             |      Score: 1.0 (match) / 0.5 (mismatch) / 0.2 (error)
             |
             |-- should_loop() -> solved? or depth >= 3? -> END
             |                    otherwise -> expand
             |
  Depth 1:  [C1]   [C2]   [C3]   expand()
             |      UCB1 selects best leaf
             |      LLM sees full trajectory (root -> leaf)
             |      Generates 3 new candidates
             |      Each executed + scored
             |
  Depth 2:  [C4]   [C5]   [C6]   expand() again
             |      UCB1 selects new best leaf
             |      3 more candidates with full history
             |
  Max 7 nodes per query, 14 LLM calls worst case

  UCB1 formula:
    ucb = (avg_reward) + C * sqrt(ln(parent_visits) / visits)

  Backpropagation:
    On each new node, walk up to root updating running averages:
    value = (value * (visits-1) + reward) / visits
```

### Scoring

Deterministic, based on execution result — not LLM judgment:

| Condition | Score | found_solution | Action |
|---|---|---|---|
| Output matches expected | 1.0 | True | Stop searching |
| Output mismatch | 0.5 | False | Keep searching |
| Execution error | 0.2 | False | Keep searching |

The LLM provides reasoning text only. Score and `found_solution` are set by the system.

`normalized_score = score / total_queries` — used for UCB calculation. For a function with 20 basis queries, a single matching query contributes 1.0/20 = 0.05 to the tree's UCB values.

## Project Structure

```
function-mapping/
  pyproject.toml                    Project metadata, dependencies
  verify_basis.py                   Re-run basis queries against PySpark to verify outputs
  .env                              e6data + Anthropic credentials (gitignored)

  src/dialect_mapper/
    __init__.py                     Package root (version 0.2.0)

    basis/                          Phase 1: Basis synthesis agent
      __init__.py                   Exports run_basis_agent
      prompt.py                     System prompt — 7 mathematical definitions + worked examples
      agent.py                      Agent runner (LangChain + filesystem middleware)
      tools.py                      execute_spark_sql tool (local PySpark singleton)

    lats/                           Phase 2: LATS dialect mapper
      __init__.py                   Exports Node, Reflection, TreeState, create_lats_graph, run_coverage_test
      state.py                      MCTS data model (Node, Reflection, TreeState)
      prompts.py                    Generation + reflection prompts, context formatters
      tools.py                      QueryExecutor, PlannerValidator, DocSearcher, SemanticChecker
      nodes.py                      LangGraph nodes (generate, expand, should_loop)
      graph.py                      StateGraph wiring, run_lats convenience function
      pipeline.py                   End-to-end orchestration (read basis -> execute -> LATS -> report)

  output/                           Generated data per function
    abs/                            8 queries, 8 partitions, 5 codomain classes
    concat_ws/                      25 queries, variadic parameter handling
    factorial/                      13 queries — discovers e6 throws on negatives (Spark returns NULL)
    datediff/                       26 queries — discovers e6 has reversed parameter order vs Spark
    web_sales.csv                   TPC-DS sample table for SQL queries

  tests/                            70 tests (all offline)
    test_lats_pipeline.py           65 tests — pipeline, MCTS model, mock executor, coverage
    test_agent.py                   5 tests — doc fetching, config
    test_prompt.py                  Prompt template content verification
    test_tools.py                   Spark SQL execution (requires local JVM)
```

## Module Dependency Graph

```
  basis/                              lats/
  ======                              =====

  agent.py                            pipeline.py
    |                                   |
    +-> prompt.py (SYSTEM_PROMPT,       +-> graph.py (run_lats)
    |              HUMAN_PROMPT)        |     |
    +-> tools.py (execute_spark_sql)    |     +-> nodes.py (generate, expand, should_loop)
    |     |                             |     |     |
    |     +-> pyspark.sql.SparkSession  |     |     +-> state.py (Node, Reflection)
    |                                   |     |     +-> prompts.py (GENERATION_PROMPT, ...)
    +-> langchain.agents.create_agent   |     |     +-> tools.py (PlannerValidator, ...)
    +-> deepagents (FilesystemMiddleware)|     |
                                        |     +-> state.py (TreeState)
                                        |     +-> tools.py (PlannerValidator, DocSearcher, ...)
                                        |     +-> langgraph (StateGraph)
                                        |
                                        +-> tools.py (QueryExecutor)
                                        +-> prompts.py (format_translation_context, ...)
                                        +-> langchain_anthropic (ChatAnthropic)
```

## Quick Start

```bash
# Install dependencies:
pip install -e ".[dev]"

# Generate basis for a function (requires local PySpark + JVM):
python -m dialect_mapper.basis.agent ABS

# Run dialect mapping (stub mode, no cluster needed):
python -m dialect_mapper.lats.pipeline ABS --stub

# Run with real e6data execution:
python -m dialect_mapper.lats.pipeline ABS

# Run with LATS translation for failures:
python -m dialect_mapper.lats.pipeline ABS --lats

# Verbose logging:
python -m dialect_mapper.lats.pipeline ABS --lats -v

# Verify basis outputs against PySpark:
python verify_basis.py output/abs/basis_table.csv

# Run tests (no cluster or API keys needed):
.venv/bin/python -m pytest tests/ -v
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
      "id": 1,
      "spark_query": "SELECT FACTORIAL(NULL)",
      "input_partition": "P_NULL",
      "e6_query": "SELECT FACTORIAL(NULL)",
      "e6_function": "FACTORIAL",
      "covered": true,
      "reasoning": "NULL propagation — both Spark and e6data return NULL"
    },
    {
      "id": 2,
      "spark_query": "SELECT FACTORIAL(-5)",
      "input_partition": "P_NEGATIVE",
      "e6_query": "SELECT CASE WHEN -5 < 0 THEN NULL ELSE FACTORIAL(-5) END",
      "e6_function": "DIRECT",
      "covered": true,
      "reasoning": "e6data throws on negative input; Spark returns NULL. CASE wrapper handles this."
    }
  ]
}
```

### e6_coverage.csv

```
id,spark_query,input_partition,expected_output,actual_output,codomain_class,direct_pass,direct_error,e6_function,e6_query,lats_pass,lats_score,covered
```

## Dialect Differences Discovered

The system discovers differences through execution, not hardcoded rules:

| Function | Difference | LATS Solution |
|---|---|---|
| FACTORIAL | Spark returns NULL for negative inputs and n>20. e6data throws `IllegalArgumentException`. | `CASE WHEN n < 0 THEN NULL ELSE FACTORIAL(n) END` |
| DATEDIFF | e6data has reversed parameter order: `DATEDIFF(a,b) = b-a` vs Spark's `a-b`. | LATS discovers this from output mismatches (-1 vs 1) and swaps parameters. |
| ABS(NULL) | e6data's ABS treats NULL as 0 instead of propagating NULL. | `CASE WHEN x IS NULL THEN NULL ELSE ABS(x) END` |

## Environment

Set in `.env` (gitignored):
```
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

E6DATA_HOST=...
E6DATA_PORT=443
E6_USER=...
E6_TOKEN=...
E6DATA_CLUSTER_NAME=...
E6DATA_SECURE=True

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=function-mapper
```

## Dependencies

Core: `langchain>=1.0`, `langgraph>=1.0`, `langchain-anthropic>=0.3`, `deepagents>=0.1`, `pyspark>=3.5`, `pydantic>=2.0`, `python-dotenv>=1.0`, `httpx>=0.27`

Optional: `e6data-python-connector==2.3.12rc11` (for real e6data execution)

Dev: `pytest>=8.0`, `pytest-cov>=4.0`