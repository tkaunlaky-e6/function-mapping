# Tests

All tests run offline — no e6data cluster, no API keys, no LLM calls needed.

## Overview

```
  Test Coverage Map
  ==================

  tests/
    |
    |-- test_lats_pipeline.py  (65 tests)
    |   |
    |   |-- TestReadBasis (5)
    |   |   reads basis_table.csv + basis_analysis.json
    |   |
    |   |-- TestGroupByPartition (3)
    |   |   groups queries by input_partition
    |   |
    |   |-- TestExtractE6Function (7)
    |   |   regex extraction from SQL strings
    |   |
    |   |-- TestCoverageMatrix (7)
    |   |   build_coverage_matrix + greedy_set_cover
    |   |
    |   |-- TestNodeAndReflection (8)
    |   |   MCTS data model: scoring, backprop, UCB, trajectories
    |   |
    |   |-- TestPlannerStub (1)
    |   |   stub always returns success
    |   |
    |   |-- TestSemanticChecker (2)
    |   |   intent preservation check (no LLM)
    |   |
    |   |-- TestFormatTranslationContext (2)
    |   |   domain context formatter
    |   |
    |   |-- TestFormatCumulativeContext (4)
    |   |   results-so-far table formatter
    |   |
    |   |-- TestStubPipeline (5)
    |   |   full pipeline with stub executor (all pass)
    |   |
    |   |-- TestQueryExecutorMock (16)
    |   |   MockQueryExecutor simulating e6data
    |   |
    |   +-- TestMockPipeline (3)
    |       full pipeline with mock executor (7/8 pass)
    |
    |-- test_agent.py  (5 tests)
    |   |-- TestFetchDocs (4)  HTTP mocking
    |   +-- TestConfig (1)     OUTPUT_DIR check
    |
    |-- test_prompt.py  (varies)
    |   |-- TestSystemPrompt   key strings in SYSTEM_PROMPT
    |   +-- TestHumanPrompt    template formatting
    |
    +-- test_tools.py  (16 tests, requires local JVM + PySpark)
        |-- TestExecuteSparkSql (14)  SQL execution
        +-- TestSparkTableLoaded (2)  web_sales table
```

## What Each Test File Covers

### test_lats_pipeline.py — LATS Dialect Mapper (65 tests)

The main test file. Tests every layer of the LATS pipeline without any network calls.

```
  Test Data Flow
  ===============

  output/abs/basis_table.csv  ─────>  TestReadBasis
  (8 real ABS queries)                  |
                                        v
                                 TestGroupByPartition
                                        |
                                        v
                            TestExtractE6Function
                            (regex on SQL strings)
                                        |
                                        v
  +-- TestNodeAndReflection ──────────────────────────+
  |   (MCTS data model, pure in-memory)               |
  |                                                    |
  |   Reflection(score=1.0, total_queries=8)           |
  |     -> normalized_score = 0.125                    |
  |                                                    |
  |   Node(messages, reflection)                       |
  |     -> backpropagate, UCB, get_best_solution       |
  |                                                    |
  |   Tree with children:                              |
  |     root(0.2) -> child1(1.0, solved)               |
  |     -> root.is_solved = true (propagated)          |
  |     -> root.value = avg(0.2, 1.0) = 0.35          |
  +----------------------------------------------------+
                                        |
                                        v
  +-- TestStubPipeline ──────────────────────────────+
  |   (PlannerValidator stub: all queries pass)       |
  |                                                    |
  |   run_coverage_test("abs", use_stub=True)          |
  |     -> all 8 queries covered                       |
  |     -> coverage = 1.0                              |
  |     -> minimal_set = ["ABS"]                       |
  |     -> writes e6_coverage.csv + e6_mapping.json    |
  +----------------------------------------------------+
                                        |
                                        v
  +-- TestQueryExecutorMock ─────────────────────────+
  |   (MockQueryExecutor: lookup table + errors)      |
  |                                                    |
  |   KNOWN_OUTPUTS = {                                |
  |     "SELECT ABS(NULL)": "NULL",                    |
  |     "SELECT ABS(-42)": "42",                       |
  |     "SELECT ABS(0)": "0",                          |
  |     ...                                            |
  |   }                                                |
  |   ERROR_QUERIES = {                                |
  |     "SELECT ABS(-9223372036854775808)":            |
  |       "ArithmeticException: overflow"              |
  |   }                                                |
  |                                                    |
  |   Tests: SELECT 1, SELECT NULL, all 8 ABS queries, |
  |          output mismatch detection, validate()     |
  |          interface, CONCAT_WS, FACTORIAL,          |
  |          FACTORIAL(-5) error, unknown query error   |
  +----------------------------------------------------+
                                        |
                                        v
  +-- TestMockPipeline ──────────────────────────────+
  |   (Full pipeline with MockQueryExecutor)          |
  |                                                    |
  |   run_coverage_test("abs", executor=mock)          |
  |     -> 7/8 pass (P_MIN_BIGINT fails with overflow)|
  |     -> coverage = 0.875                            |
  |     -> cumulative context shows PASS/FAIL          |
  |     -> partition scores reflect failure             |
  +----------------------------------------------------+
```

**Key test classes**:

| Class | Tests | What it validates |
|---|---|---|
| TestReadBasis | 5 | CSV parsing, field validation, analysis JSON structure |
| TestGroupByPartition | 3 | Partition grouping, single-query partitions, empty input |
| TestExtractE6Function | 7 | ABS, CONCAT_WS, FACTORIAL, CASE, CAST, DIRECT, lowercase |
| TestCoverageMatrix | 7 | Matrix building, empty/no-covered edge cases, greedy set cover (single, two-func, overlap, uncoverable) |
| TestNodeAndReflection | 8 | Normalized score (score/total_queries), as_message content, node creation, tree+children, solved propagation, backprop averaging, get_best_solution, trajectory |
| TestPlannerStub | 1 | Stub always returns success |
| TestSemanticChecker | 2 | Similar queries pass, trivially short queries fail |
| TestFormatTranslationContext | 2 | THIS PARTITION marker, partition queries included |
| TestFormatCumulativeContext | 4 | Empty results, single PASS, mixed PASS/FAIL, headers |
| TestStubPipeline | 5 | Full run, all pass, partition scores, coverage matrix, output files |
| TestQueryExecutorMock | 16 | All mock executor scenarios |
| TestMockPipeline | 3 | Realistic pipeline (7/8 pass), cumulative context, partition scores |

### test_agent.py — Basis Agent (5 tests)

Tests doc fetching and configuration without invoking any LLM.

| Test | What it validates |
|---|---|
| test_unknown_dialect | Unknown dialect returns "Use your knowledge" fallback |
| test_successful_fetch | HTTP 200 returns doc content (mocked httpx) |
| test_http_error | HTTP 404 returns fallback string |
| test_network_error | Network exception returns fallback string |
| test_output_dir | OUTPUT_DIR.name == "output" |

### test_prompt.py — Prompt Templates

Structural content assertions — no LLM, just verifying key strings exist in prompt templates.

**TestSystemPrompt**: Checks for mathematical framework terms (Equivalence, COVERING, DISJOINT, Output Range, TYPE BOUNDS, VALUE BOUNDS, Codomain Classes, Surjectivity), worked examples (POWER), variadic handling, tool names, workflow steps (STEP 1-7), output file names.

**TestHumanPrompt**: Template formatting works, all placeholders present, no unfilled `{` in formatted output.

### test_tools.py — Spark SQL Execution (16 tests, requires JVM)

Tests `execute_spark_sql` against a real local PySpark session. **Requires Java/JVM installed.**

| Test | Query | Expected |
|---|---|---|
| test_select_one | SELECT 1 | "1" |
| test_select_null | SELECT NULL | "NULL" |
| test_string | SELECT 'hello' | "hello" |
| test_abs_positive | SELECT ABS(-5) | "5" |
| test_abs_null | SELECT ABS(NULL) | "NULL" |
| ... | various SQL functions | string results |
| test_invalid_sql | SELECT INVALID_FUNC() | "ERROR: ..." |
| test_decimal | SELECT 1.5 + 2.3 | "3.8" (float, not Decimal) |

**TestSparkTableLoaded**: Confirms `recent_web_sales` temp view exists and column queries work.

## MockQueryExecutor

`test_lats_pipeline.py` includes a `MockQueryExecutor` class that simulates e6data behavior using hardcoded lookup tables:

```python
KNOWN_OUTPUTS = {
    "SELECT 1":              "1",
    "SELECT NULL":           "NULL",
    "SELECT ABS(NULL)":      "NULL",
    "SELECT ABS(-42)":       "42",
    "SELECT ABS(0)":         "0",
    "SELECT ABS(99)":        "99",
    "SELECT ABS(-3.14)":     "3.14",
    "SELECT ABS(3.14)":      "3.14",
    "SELECT CAST(ABS(CAST('-7' AS INT)) AS STRING)": "7",
    ...
}
ERROR_QUERIES = {
    "SELECT ABS(-9223372036854775808)": "ArithmeticException: overflow",
    "SELECT FACTORIAL(-5)":             "IllegalArgumentException: negative",
}
```

This allows testing the full pipeline offline. For ABS: 7/8 queries pass, P_MIN_BIGINT fails with overflow (matches real e6data behavior).

## Running

```bash
# All tests (test_tools.py requires JVM):
.venv/bin/python -m pytest tests/ -v

# LATS pipeline + agent tests only (no JVM needed):
.venv/bin/python -m pytest tests/test_lats_pipeline.py tests/test_agent.py -v

# Single test class:
.venv/bin/python -m pytest tests/test_lats_pipeline.py::TestMockPipeline -v

# With coverage:
.venv/bin/python -m pytest tests/ --cov=dialect_mapper --cov-report=term-missing
```
