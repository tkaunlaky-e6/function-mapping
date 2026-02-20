# Basis Synthesis Agent

Generates the mathematical basis for a SQL function — the minimal complete set of test queries that covers every distinct behavior of the function.

## Overview

```
  Basis Synthesis Pipeline
  =========================

  Input: Function name (e.g., "ABS", "CONCAT_WS", "DATEDIFF")

  +---------------------+     +------------------------+
  |  Spark SQL Docs     |     |  SYSTEM_PROMPT         |
  |  (HTTP fetch from   |     |  7 mathematical defs:  |
  |   apache.org)       |     |  1. Equivalence        |
  +----------+----------+     |  2. Spanning           |
             |                |  3. Finding Partitions |
             v                |  4. Output Range       |
  +----------+----------+     |  5. Codomain Classes   |
  |  HUMAN_PROMPT       |     |  6. Surjectivity       |
  |  (formatted with    |     |  7. Completeness       |
  |   func name, docs,  |     +----------+-------------+
  |   output dir)       |                |
  +----------+----------+                |
             |                           |
             +----------+---------------+
                        |
                        v
  +--------------------------------------------+
  |         LangChain Agent                     |
  |  model: Claude Sonnet                       |
  |  tools: execute_spark_sql                   |
  |         read_file, write_file               |
  |         (from FilesystemMiddleware)          |
  +-----+--------------------------------------+
        |
        |  Agent Loop (LLM reasons + calls tools)
        |
        v
  +--------------------------------------------+
  |  Step 1: Read documentation                |
  |  Step 2: Probe the domain                  |
  |    execute_spark_sql("SELECT F(NULL)")     |
  |    execute_spark_sql("SELECT F(-1)")       |
  |    execute_spark_sql("SELECT F('abc')")    |
  |    -> discover what types/values work      |
  |                                            |
  |  Step 3: Partition the domain              |
  |    P_NULL: input is NULL                   |
  |    P_NEGATIVE: input < 0                   |
  |    P_ZERO: input = 0                       |
  |    ... etc.                                |
  |    Write spanning proof: Union(Pi) = D     |
  |                                            |
  |  Step 4: Generate one query per partition  |
  |    execute_spark_sql per query             |
  |    record actual output                    |
  |                                            |
  |  Step 5: Derive codomain classes           |
  |    O_NULL, O_ZERO, O_POSITIVE_INT, ...     |
  |                                            |
  |  Step 6: Check surjectivity                |
  |    Every codomain class covered?           |
  |    If not -> back-solve to fill gaps       |
  |                                            |
  |  Step 7: Write output files                |
  |    write_file(basis_table.csv)             |
  |    write_file(basis_analysis.json)         |
  +--------------------------------------------+
        |
        v
  +--------------------------------------------+
  |  output/<func>/                            |
  |    basis_table.csv      (queries + outputs)|
  |    basis_analysis.json  (formal analysis)  |
  +--------------------------------------------+
```

## Files

### `prompt.py` — Mathematical Framework

The system prompt that teaches the LLM how to perform rigorous basis synthesis. Contains:

- **7 formal definitions** with mathematical notation:
  1. **Equivalence and Partitions** — input domain D partitioned into k classes B
  2. **Spanning** — COVERING (union = Dp) and DISJOINT predicates, with formal proofs for INTEGER, STRING, COUPLED parameters
  3. **Finding Partitions** — behavioral reasoning: "what makes the function do something different?"
  4. **Output Range R(F)** — TYPE BOUNDS, VALUE BOUNDS, OUTPUT-INPUT RELATIONSHIP
  5. **Codomain Classes** — partition the output range into distinct categories (O_NULL, O_ZERO, etc.)
  6. **Surjectivity** — every codomain class is covered; back-solve for gaps
  7. **Completeness = Spanning + Surjectivity**

- **Worked example** for `POWER(base, exp)` — all partitions, output range, codomain classes, basis queries

- **Variadic function handling** — table of partition cases for functions like `CONCAT_WS` that take variable argument counts

- **7-step workflow** the agent must follow

- **CSV quoting rules** — any field containing a comma must be double-quoted

- **HUMAN_PROMPT** — template formatted with `func_name`, `dialect`, `documentation`, `output_dir`

### `agent.py` — Agent Runner

Creates and runs the LangChain agent:

```
  agent.py internals
  ==================

  DOC_URLS = {"spark": "https://spark.apache.org/docs/.../{func}"}

  _fetch_docs(func_name, dialect)
    |-> HTTP GET spark docs URL
    |-> returns up to 15,000 chars
    |-> on error: "Use your knowledge of {func_name}"

  create_basis_agent(model=None)
    |-> reads ANTHROPIC_MODEL from env (default: claude-sonnet-4-5-20250929)
    |-> langchain.agents.create_agent(
    |     model,
    |     tools=[execute_spark_sql],
    |     system_prompt=SYSTEM_PROMPT,
    |     middleware=[FilesystemMiddleware(FilesystemBackend())]
    |   )
    |-> FilesystemMiddleware injects read_file + write_file tools

  run_basis_agent(func_name, dialect="spark", model=None, output_dir=None)
    |-> creates output/<func>/basis_table.csv with header if missing
    |-> fetches docs
    |-> formats HUMAN_PROMPT
    |-> agent.invoke({messages: [{role: "user", content: human_msg}]})
    |-> returns full agent state dict
```

**CLI**: `python -m dialect_mapper.basis.agent ABS [dialect]`

### `tools.py` — PySpark Execution Tool

Single tool available to the agent: `execute_spark_sql(query: str) -> str`

```
  tools.py internals
  ==================

  _spark = None  (module-level singleton)

  _get_spark() -> SparkSession
    |-> creates SparkSession.builder
    |     .master("local[*]")
    |     .appName("basis-agent")
    |     .config("spark.ui.enabled", "false")
    |     .getOrCreate()
    |-> loads output/web_sales.csv as temp view "recent_web_sales" (if exists)
    |-> path overridable via BASIS_CSV_PATH env var

  execute_spark_sql(query: str) -> str
    |-> spark.sql(query).collect()
    |-> returns rows[0][0] as string
    |-> Decimal -> float conversion
    |-> None -> "NULL"
    |-> empty result -> "NULL"
    |-> exception -> "ERROR: <msg[:300]>"
```

## Output Format

### basis_table.csv

```
id,query,input_partition,actual_output,codomain_class,reasoning
1,SELECT ABS(NULL),P_NULL,NULL,O_NULL,NULL input propagates to NULL output
2,SELECT ABS(-42),P_NEGATIVE_INT,42,O_POSITIVE_INT,Negation of negative integer
3,"SELECT CONCAT_WS(',', 'a', 'b')",P_MULTI_STRING,a,b,O_JOINED,Comma-separated join
```

**CSV quoting**: any field containing a comma must be double-quoted. SQL queries with multiple arguments always contain commas (e.g., `CONCAT_WS(',', 'a', 'b')`), so the query column must always be quoted for multi-arg functions.

### basis_analysis.json

```json
{
  "function": "ABS",
  "parameters": [
    {
      "name": "n",
      "partitions": [
        {
          "id": "P_NULL",
          "predicate": "n IS NULL",
          "representative": "NULL",
          "reasoning": "NULL propagation"
        },
        {
          "id": "P_NEGATIVE_INT",
          "predicate": "n IN Z, n < 0",
          "representative": "-42",
          "reasoning": "Negation behavior"
        }
      ],
      "spanning_proof": {
        "covering": "P_NULL ∪ P_NEGATIVE_INT ∪ P_ZERO ∪ ... = D",
        "disjoint": "predicates are mutually exclusive"
      }
    }
  ],
  "output_analysis": {
    "range": "{NULL} ∪ {0} ∪ R+",
    "codomain_classes": [
      {"id": "O_NULL", "predicate": "output IS NULL"},
      {"id": "O_ZERO", "predicate": "output = 0"},
      {"id": "O_POSITIVE_INT", "predicate": "output IN Z, output > 0"}
    ],
    "surjectivity": {"coverage_ratio": 1.0}
  },
  "basis_statistics": {
    "total_queries": 8,
    "spanning_complete": true,
    "surjectivity_complete": true
  }
}
```

## Usage

```bash
# Generate basis for ABS:
python -m dialect_mapper.basis.agent ABS

# Generate basis for DATEDIFF:
python -m dialect_mapper.basis.agent DATEDIFF spark

# Verify basis outputs are still correct:
python verify_basis.py output/abs/basis_table.csv
```

## Date/Timestamp Edge Cases

For functions with date or timestamp parameters, the agent also tests:
- Leap year boundaries, year/month boundaries, extreme date ranges
- Timestamps vs dates, string-to-date casting, time component handling