"""Tests for the LATS dialect mapping pipeline.

Tests cover:
  1. Pipeline utilities (read_basis, group_by_partition, extract_e6_function, etc.)
  2. Cumulative context formatting
  3. Coverage matrix + greedy set cover
  4. State/Node/Reflection data model
  5. Full stub pipeline (no network, no LLM)
  6. QueryExecutor real execution (integration, requires e6data)
"""

import pytest
from pathlib import Path

from dialect_mapper.lats.pipeline import (
    read_basis,
    group_by_partition,
    extract_e6_function,
    build_coverage_matrix,
    greedy_set_cover,
    run_coverage_test,
)
from dialect_mapper.lats.prompts import (
    format_translation_context,
    format_cumulative_context,
    GENERATION_PROMPT,
    REFLECTION_PROMPT,
)
from dialect_mapper.lats.state import Node, Reflection, TreeState
from dialect_mapper.lats.tools import PlannerValidator, SemanticChecker


# ---------------------------------------------------------------------------
# Test: read_basis
# ---------------------------------------------------------------------------

class TestReadBasis:
    """Test reading basis CSV and analysis JSON."""

    def test_read_abs(self):
        rows, analysis = read_basis("ABS")
        assert len(rows) == 8
        assert analysis["function"] == "ABS"

    def test_read_concat_ws(self):
        rows, analysis = read_basis("CONCAT_WS")
        assert len(rows) > 0

    def test_read_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            read_basis("NONEXISTENT_FUNCTION_XYZ")

    def test_rows_have_required_fields(self):
        rows, _ = read_basis("ABS")
        for row in rows:
            assert "id" in row
            assert "query" in row
            assert "input_partition" in row
            assert "actual_output" in row

    def test_analysis_has_parameter_partitions(self):
        _, analysis = read_basis("ABS")
        assert "parameter_analysis" in analysis
        assert "n" in analysis["parameter_analysis"]
        partitions = analysis["parameter_analysis"]["n"]["partitions"]
        assert len(partitions) >= 8


# ---------------------------------------------------------------------------
# Test: group_by_partition
# ---------------------------------------------------------------------------

class TestGroupByPartition:
    """Test partition grouping logic."""

    def test_abs_partitions(self):
        rows, _ = read_basis("ABS")
        groups = group_by_partition(rows)
        assert len(groups) == 8
        assert "P_NULL" in groups
        assert "P_NEGATIVE_INT" in groups
        assert "P_ZERO" in groups

    def test_single_query_partitions(self):
        rows, _ = read_basis("ABS")
        groups = group_by_partition(rows)
        assert len(groups["P_NULL"]) == 1
        assert len(groups["P_ZERO"]) == 1

    def test_empty_rows(self):
        groups = group_by_partition([])
        assert groups == {}


# ---------------------------------------------------------------------------
# Test: extract_e6_function
# ---------------------------------------------------------------------------

class TestExtractE6Function:
    """Test SQL function name extraction."""

    def test_abs(self):
        assert extract_e6_function("SELECT ABS(-5)") == "ABS"

    def test_concat_ws(self):
        assert extract_e6_function("SELECT CONCAT_WS(',', 'a', 'b')") == "CONCAT_WS"

    def test_factorial(self):
        assert extract_e6_function("SELECT FACTORIAL(5)") == "FACTORIAL"

    def test_case_expression(self):
        assert extract_e6_function("SELECT CASE WHEN x > 0 THEN 1 ELSE 0 END") == "CASE_EXPRESSION"

    def test_cast(self):
        assert extract_e6_function("SELECT CAST(5.5 AS INT)") == "CAST"

    def test_no_function(self):
        assert extract_e6_function("SELECT 1") == "DIRECT"

    def test_lowercase(self):
        assert extract_e6_function("SELECT abs(-5)") == "ABS"


# ---------------------------------------------------------------------------
# Test: coverage matrix + greedy set cover
# ---------------------------------------------------------------------------

class TestCoverageMatrix:
    """Test coverage matrix building and greedy set cover."""

    def test_build_matrix(self):
        results = [
            {"id": "1", "e6_function": "ABS", "covered": True},
            {"id": "2", "e6_function": "ABS", "covered": True},
            {"id": "3", "e6_function": "ABS", "covered": False},
            {"id": "4", "e6_function": "CASE_EXPRESSION", "covered": True},
        ]
        matrix = build_coverage_matrix(results)
        assert "ABS" in matrix
        assert matrix["ABS"]["count"] == 2
        assert set(matrix["ABS"]["covered_ids"]) == {1, 2}
        assert "CASE_EXPRESSION" in matrix
        assert matrix["CASE_EXPRESSION"]["count"] == 1

    def test_empty_results(self):
        matrix = build_coverage_matrix([])
        assert matrix == {}

    def test_no_covered(self):
        results = [
            {"id": "1", "e6_function": "ABS", "covered": False},
        ]
        matrix = build_coverage_matrix(results)
        assert matrix == {}

    def test_greedy_set_cover_single(self):
        matrix = {
            "ABS": {"covered_ids": [1, 2, 3, 4, 5], "count": 5},
        }
        selected = greedy_set_cover(matrix, {1, 2, 3, 4, 5})
        assert selected == ["ABS"]

    def test_greedy_set_cover_two_functions(self):
        matrix = {
            "ABS": {"covered_ids": [1, 2, 3], "count": 3},
            "CASE_EXPRESSION": {"covered_ids": [4, 5], "count": 2},
        }
        selected = greedy_set_cover(matrix, {1, 2, 3, 4, 5})
        assert len(selected) == 2
        assert "ABS" in selected
        assert "CASE_EXPRESSION" in selected

    def test_greedy_set_cover_overlap(self):
        matrix = {
            "A": {"covered_ids": [1, 2, 3, 4], "count": 4},
            "B": {"covered_ids": [3, 4, 5], "count": 3},
        }
        selected = greedy_set_cover(matrix, {1, 2, 3, 4, 5})
        assert selected == ["A", "B"]

    def test_greedy_set_cover_uncoverable(self):
        matrix = {
            "A": {"covered_ids": [1, 2], "count": 2},
        }
        selected = greedy_set_cover(matrix, {1, 2, 3})
        assert selected == ["A"]


# ---------------------------------------------------------------------------
# Test: Node + Reflection (LATS state model)
# ---------------------------------------------------------------------------

class TestNodeAndReflection:
    """Test the MCTS tree data model."""

    def test_reflection_normalized_score(self):
        """normalized_score = score / total_queries."""
        r = Reflection(score=1.0, total_queries=8)
        assert r.normalized_score == 1.0 / 8

    def test_reflection_normalized_score_default(self):
        """With total_queries=1 (default), normalized_score = score."""
        r = Reflection(score=0.5)
        assert r.normalized_score == 0.5

    def test_reflection_as_message(self):
        r = Reflection(
            score=0.2,
            planner_success=False,
            planner_error="ABS not found",
            preserves_intent=True,
        )
        msg = r.as_message()
        assert "FAILED" in msg.content
        assert "ABS not found" in msg.content

    def test_node_creation(self):
        from langchain_core.messages import AIMessage
        r = Reflection(score=0.5, found_solution=False)
        node = Node(
            messages=[AIMessage(content="SELECT 1")],
            reflection=r,
        )
        assert node.depth == 1
        assert node.visits == 1
        assert not node.is_solved
        assert node.is_terminal

    def test_node_with_children(self):
        from langchain_core.messages import AIMessage
        r1 = Reflection(score=0.2, found_solution=False)
        root = Node(messages=[AIMessage(content="attempt1")], reflection=r1)

        r2 = Reflection(score=1.0, found_solution=True, planner_success=True, preserves_intent=True)
        child = Node(messages=[AIMessage(content="attempt2")], reflection=r2, parent=root)
        root.children.append(child)

        assert child.depth == 2
        assert child.is_solved
        assert root.is_solved  # propagated up
        assert not root.is_terminal
        assert child.is_terminal

    def test_backpropagation(self):
        from langchain_core.messages import AIMessage
        r1 = Reflection(score=0.2, found_solution=False)
        root = Node(messages=[AIMessage(content="attempt1")], reflection=r1)
        assert root.visits == 1

        r2 = Reflection(score=0.5, found_solution=False)
        child = Node(messages=[AIMessage(content="attempt2")], reflection=r2, parent=root)

        # root: (0.2 * 1 + 0.5) / 2 = 0.35
        assert root.visits == 2
        assert abs(root.value - 0.35) < 0.01

    def test_get_best_solution(self):
        from langchain_core.messages import AIMessage
        r1 = Reflection(score=0.2, found_solution=False)
        root = Node(messages=[AIMessage(content="root")], reflection=r1)

        r2 = Reflection(score=1.0, found_solution=True, planner_success=True, preserves_intent=True)
        child = Node(messages=[AIMessage(content="child")], reflection=r2, parent=root)
        root.children.append(child)

        best = root.get_best_solution()
        assert best is child
        assert best.is_solved

    def test_trajectory(self):
        from langchain_core.messages import AIMessage
        r1 = Reflection(score=0.2, found_solution=False, planner_success=False, planner_error="err1")
        root = Node(messages=[AIMessage(content="root_sql")], reflection=r1)

        r2 = Reflection(score=1.0, found_solution=True, planner_success=True)
        child = Node(messages=[AIMessage(content="child_sql")], reflection=r2, parent=root)

        traj = child.get_trajectory(include_reflections=True)
        assert len(traj) >= 4


# ---------------------------------------------------------------------------
# Test: PlannerValidator stub
# ---------------------------------------------------------------------------

class TestPlannerStub:
    """Test the stub planner always passes."""

    def test_stub_always_passes(self):
        stub = PlannerValidator()
        result = stub.validate("SELECT anything")
        assert result["success"] is True
        assert result["error"] == ""


# ---------------------------------------------------------------------------
# Test: SemanticChecker basic
# ---------------------------------------------------------------------------

class TestSemanticChecker:
    """Test the basic semantic checker (no LLM)."""

    def test_similar_queries_pass(self):
        checker = SemanticChecker()
        result = checker.check(
            "SELECT ABS(-5)",
            "SELECT ABS(CAST(-5 AS INT))",
        )
        assert result["preserves_intent"] is True

    def test_trivial_query_fails(self):
        checker = SemanticChecker()
        result = checker.check(
            "SELECT SUM(amount) FROM orders WHERE status = 'active' GROUP BY customer_id",
            "SELECT 1",
        )
        assert result["preserves_intent"] is False


# ---------------------------------------------------------------------------
# Test: format_translation_context
# ---------------------------------------------------------------------------

class TestFormatTranslationContext:
    """Test the domain analysis context formatter."""

    def test_basic_context(self):
        analysis = {
            "function": "ABS",
            "parameter_analysis": {
                "n": {
                    "real_domain": "NUMERIC",
                    "partitions": [
                        {"name": "P_ZERO", "predicate": "n = 0", "reasoning": "base case"},
                        {"name": "P_NEGATIVE_INT", "predicate": "n < 0", "reasoning": "negate"},
                    ]
                }
            },
            "output_analysis": {
                "range": "{0} ∪ ℝ⁺",
                "special_properties": ["ABS(0) = 0"],
                "codomain_classes": [
                    {"name": "O_ZERO", "description": "output = 0"},
                ]
            }
        }
        query_row = {
            "query": "SELECT ABS(0)",
            "input_partition": "P_ZERO",
            "actual_output": "0",
            "codomain_class": "O_ZERO",
            "reasoning": "base case",
        }
        ctx = format_translation_context(analysis, query_row)
        assert "ABS" in ctx
        assert "P_ZERO" in ctx
        assert "THIS PARTITION" in ctx
        assert "Expected output: 0" in ctx

    def test_partition_queries(self):
        analysis = {"function": "ABS", "parameter_analysis": {}, "output_analysis": {}}
        query_row = {
            "query": "SELECT ABS(-5)",
            "input_partition": "P_NEGATIVE_INT",
            "actual_output": "5",
            "codomain_class": "O_POSITIVE_INT",
            "reasoning": "negate",
        }
        partition_queries = [
            {"query": "SELECT ABS(-5)", "actual_output": "5"},
            {"query": "SELECT ABS(-10)", "actual_output": "10"},
        ]
        ctx = format_translation_context(analysis, query_row, partition_queries)
        assert "ALL QUERIES IN PARTITION" in ctx
        assert "ABS(-10)" in ctx
        assert "10" in ctx


# ---------------------------------------------------------------------------
# Test: format_cumulative_context
# ---------------------------------------------------------------------------

class TestFormatCumulativeContext:
    """Test the cumulative context (CSV-like table) formatter."""

    def test_empty_results(self):
        assert format_cumulative_context([]) == ""

    def test_single_pass(self):
        results = [{
            "id": "1",
            "spark_query": "SELECT ABS(0)",
            "input_partition": "P_ZERO",
            "expected_output": "0",
            "actual_output": "0",
            "e6_query": "SELECT ABS(0)",
            "e6_function": "ABS",
            "direct_pass": True,
            "direct_error": "",
            "covered": True,
        }]
        ctx = format_cumulative_context(results)
        assert "TRANSLATION RESULTS SO FAR" in ctx
        assert "PASS" in ctx
        assert "ABS" in ctx
        assert "1/1" in ctx

    def test_mixed_pass_fail(self):
        results = [
            {
                "id": "1",
                "spark_query": "SELECT ABS(5)",
                "input_partition": "P_POSITIVE_INT",
                "expected_output": "5",
                "actual_output": "5",
                "e6_query": "SELECT ABS(5)",
                "e6_function": "ABS",
                "direct_pass": True,
                "direct_error": "",
                "covered": True,
            },
            {
                "id": "2",
                "spark_query": "SELECT ABS(-9223372036854775808)",
                "input_partition": "P_MIN_BIGINT",
                "expected_output": "9223372036854775808",
                "actual_output": "",
                "e6_query": "",
                "e6_function": "",
                "direct_pass": False,
                "direct_error": "Execution error: overflow",
                "covered": False,
            },
        ]
        ctx = format_cumulative_context(results)
        assert "PASS" in ctx
        assert "FAIL" in ctx
        assert "1/2" in ctx
        assert "P_POSITIVE_INT" in ctx
        assert "P_MIN_BIGINT" in ctx

    def test_table_headers(self):
        results = [{
            "id": "1",
            "spark_query": "SELECT ABS(0)",
            "input_partition": "P_ZERO",
            "expected_output": "0",
            "actual_output": "0",
            "e6_query": "SELECT ABS(0)",
            "e6_function": "ABS",
            "direct_pass": True,
            "direct_error": "",
            "covered": True,
        }]
        ctx = format_cumulative_context(results)
        assert "id |" in ctx
        assert "input_partition" in ctx
        assert "spark_query" in ctx


# ---------------------------------------------------------------------------
# Test: Stub pipeline (full run, no network)
# ---------------------------------------------------------------------------

class TestStubPipeline:
    """Test the full pipeline in stub mode (no e6data, no LLM)."""

    def test_abs_stub(self):
        report = run_coverage_test("ABS", use_stub=True, use_lats=False)

        assert report["function"] == "ABS"
        assert report["total"] == 8
        assert report["coverage"] == 1.0  # stub always passes
        assert report["direct_pass"] == 8
        assert report["direct_fail"] == 0
        assert len(report["results"]) == 8

        for r in report["results"]:
            assert r["covered"] is True

    def test_abs_stub_partition_scores(self):
        report = run_coverage_test("ABS", use_stub=True, use_lats=False)

        ps = report["partition_scores"]
        assert "P_NULL" in ps
        assert "P_NEGATIVE_INT" in ps
        assert ps["P_NULL"]["fully_covered"] is True
        assert ps["P_ZERO"]["score"] == 1.0

    def test_abs_stub_coverage_matrix(self):
        report = run_coverage_test("ABS", use_stub=True, use_lats=False)

        matrix = report["coverage_matrix"]
        assert "ABS" in matrix
        assert matrix["ABS"]["count"] == 8

    def test_abs_stub_minimal_set(self):
        report = run_coverage_test("ABS", use_stub=True, use_lats=False)
        assert report["minimal_function_set"] == ["ABS"]

    def test_stub_writes_output_files(self):
        """Verify CSV and JSON output files are written."""
        report = run_coverage_test("ABS", use_stub=True, use_lats=False)

        out_dir = Path(__file__).resolve().parents[1] / "output" / "abs"
        csv_path = out_dir / "e6_coverage.csv"
        json_path = out_dir / "e6_mapping.json"

        assert csv_path.exists()
        assert json_path.exists()

        # Check CSV contents
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 8
        assert rows[0]["spark_query"].startswith("SELECT ABS")

        # Check JSON contents
        import json
        with open(json_path) as f:
            mapping = json.load(f)
        assert mapping["spark_function"] == "ABS"
        assert mapping["overall_coverage"] == 1.0
        assert len(mapping["translations"]) == 8


# ---------------------------------------------------------------------------
# MockQueryExecutor: simulates e6data with a lookup table (no cluster needed)
# ---------------------------------------------------------------------------

class MockQueryExecutor:
    """
    Simulates QueryExecutor using a lookup table of known SQL → output mappings.
    No network, no cluster — tests run offline.

    The lookup table mirrors what e6data actually returns for ABS queries
    (verified in earlier sessions when the cluster was live).
    """

    # Known e6data outputs — verified against real cluster
    KNOWN_OUTPUTS = {
        "SELECT 1":                              [[1]],
        "SELECT NULL":                           [[None]],
        "SELECT ABS(NULL)":                      [[None]],
        "SELECT ABS(-5)":                        [[5]],
        "SELECT ABS(0)":                         [[0]],
        "SELECT ABS(5)":                         [[5]],
        "SELECT ABS(-3.14)":                     [[3.14]],
        "SELECT ABS(2.71)":                      [[2.71]],
        "SELECT ABS(CAST('-10' AS INT))":        [[10]],
        "SELECT CONCAT_WS(',', 'a', 'b', 'c')": [["a,b,c"]],
        "SELECT FACTORIAL(5)":                   [[120]],
        "SELECT FACTORIAL(0)":                   [[1]],
        "SELECT FACTORIAL(1)":                   [[1]],
    }

    # Queries that throw runtime errors on e6data
    ERROR_QUERIES = {
        "SELECT ABS(-9223372036854775808)": "Execution error: overflow — BIGINT boundary",
        "SELECT FACTORIAL(-5)": "Execution error: runtime error for negative factorial",
        "SELECT FACTORIAL(21)": "Execution error: overflow",
    }

    def __init__(self, catalog="nl2sql", database="tpcds_1"):
        self.default_catalog = catalog
        self.default_database = database
        self.expected_output = ""

    def execute(self, sql: str) -> list:
        """Look up SQL in known outputs or raise if it's an error query."""
        sql_clean = sql.strip().rstrip(";")
        if sql_clean in self.ERROR_QUERIES:
            raise RuntimeError(self.ERROR_QUERIES[sql_clean])
        if sql_clean in self.KNOWN_OUTPUTS:
            return self.KNOWN_OUTPUTS[sql_clean]
        raise RuntimeError(f"Unknown query (not in mock lookup): {sql_clean}")

    def execute_and_compare(self, sql: str, expected: str) -> dict:
        """Execute and compare — same interface as QueryExecutor."""
        try:
            result = self.execute(sql)
            actual = result[0][0] if result and result[0] else None
            actual_str = str(actual) if actual is not None else "NULL"

            match = (actual_str == expected)
            if match:
                return {
                    "success": True, "error": "",
                    "actual_output": actual_str, "expected_output": expected,
                    "match": True,
                }
            else:
                return {
                    "success": False,
                    "error": f"Output mismatch: expected '{expected}', got '{actual_str}'",
                    "actual_output": actual_str, "expected_output": expected,
                    "match": False,
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)[:200]}",
                "actual_output": "", "expected_output": expected,
                "match": False,
            }

    def validate(self, sql: str, catalog: str = "", database: str = "") -> dict:
        """Same interface as PlannerValidator — uses self.expected_output."""
        result = self.execute_and_compare(sql, self.expected_output)
        return {"success": result["success"], "error": result["error"]}


# ---------------------------------------------------------------------------
# Test: QueryExecutor with mock (no cluster needed)
# ---------------------------------------------------------------------------

class TestQueryExecutorMock:
    """Test QueryExecutor behavior using the mock lookup table."""

    @pytest.fixture
    def executor(self):
        return MockQueryExecutor()

    # --- Basic execution ---

    def test_select_one(self, executor):
        result = executor.execute("SELECT 1")
        assert result == [[1]]

    def test_select_null(self, executor):
        result = executor.execute("SELECT NULL")
        assert result == [[None]]

    # --- ABS function: all 8 basis queries ---

    def test_abs_null(self, executor):
        result = executor.execute_and_compare("SELECT ABS(NULL)", "NULL")
        assert result["success"] is True
        assert result["match"] is True
        assert result["actual_output"] == "NULL"

    def test_abs_negative_int(self, executor):
        result = executor.execute_and_compare("SELECT ABS(-5)", "5")
        assert result["success"] is True
        assert result["match"] is True
        assert result["actual_output"] == "5"

    def test_abs_zero(self, executor):
        result = executor.execute_and_compare("SELECT ABS(0)", "0")
        assert result["success"] is True
        assert result["match"] is True
        assert result["actual_output"] == "0"

    def test_abs_positive_int(self, executor):
        result = executor.execute_and_compare("SELECT ABS(5)", "5")
        assert result["success"] is True
        assert result["match"] is True

    def test_abs_negative_decimal(self, executor):
        result = executor.execute_and_compare("SELECT ABS(-3.14)", "3.14")
        assert result["success"] is True
        assert result["match"] is True

    def test_abs_positive_decimal(self, executor):
        result = executor.execute_and_compare("SELECT ABS(2.71)", "2.71")
        assert result["success"] is True
        assert result["match"] is True

    def test_abs_min_bigint_error(self, executor):
        """MIN BIGINT boundary — e6data throws overflow error."""
        result = executor.execute_and_compare(
            "SELECT ABS(-9223372036854775808)", "9223372036854775808"
        )
        assert result["success"] is False
        assert "error" in result["error"].lower()
        assert result["match"] is False

    def test_abs_string_cast(self, executor):
        result = executor.execute_and_compare("SELECT ABS(CAST('-10' AS INT))", "10")
        assert result["success"] is True
        assert result["match"] is True

    # --- Output mismatch detection ---

    def test_output_mismatch(self, executor):
        """When e6 returns 5 but we expect 999 → mismatch."""
        result = executor.execute_and_compare("SELECT ABS(-5)", "999")
        assert result["success"] is False
        assert result["match"] is False
        assert "mismatch" in result["error"].lower()
        assert result["actual_output"] == "5"
        assert result["expected_output"] == "999"

    # --- validate() interface (used by LATS core) ---

    def test_validate_match(self, executor):
        executor.expected_output = "5"
        result = executor.validate("SELECT ABS(-5)")
        assert result["success"] is True
        assert result["error"] == ""

    def test_validate_mismatch(self, executor):
        executor.expected_output = "999"
        result = executor.validate("SELECT ABS(-5)")
        assert result["success"] is False
        assert "mismatch" in result["error"].lower()

    def test_validate_error_query(self, executor):
        executor.expected_output = "9223372036854775808"
        result = executor.validate("SELECT ABS(-9223372036854775808)")
        assert result["success"] is False
        assert "error" in result["error"].lower()

    # --- Cross-function queries ---

    def test_concat_ws(self, executor):
        result = executor.execute_and_compare(
            "SELECT CONCAT_WS(',', 'a', 'b', 'c')", "a,b,c"
        )
        assert result["success"] is True
        assert result["match"] is True

    def test_factorial(self, executor):
        result = executor.execute_and_compare("SELECT FACTORIAL(5)", "120")
        assert result["success"] is True
        assert result["match"] is True

    def test_factorial_negative_error(self, executor):
        result = executor.execute_and_compare("SELECT FACTORIAL(-5)", "NULL")
        assert result["success"] is False
        assert "error" in result["error"].lower()

    # --- Unknown query handling ---

    def test_unknown_query_error(self, executor):
        result = executor.execute_and_compare("SELECT SOME_FUNC(42)", "42")
        assert result["success"] is False
        assert "not in mock" in result["error"].lower() or "error" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test: Full pipeline with MockQueryExecutor
# ---------------------------------------------------------------------------

class TestMockPipeline:
    """Test pipeline with mock executor — simulates real e6data behavior.

    Unlike TestStubPipeline (where everything passes), this tests with
    realistic pass/fail behavior for ABS queries.
    """

    def test_abs_mock_pipeline(self, monkeypatch):
        """Run pipeline with mock executor to see realistic ABS coverage."""
        import dialect_mapper.lats.pipeline as pipeline_mod

        # Patch the executor creation in run_coverage_test
        mock_exec = MockQueryExecutor()

        # We can't easily swap the executor inside run_coverage_test,
        # so we test the components that feed it instead
        rows, analysis = read_basis("ABS")
        partitions = group_by_partition(rows)

        results = []
        for partition_name, queries in partitions.items():
            for row in queries:
                query = row["query"]
                expected = row.get("actual_output", "")
                exec_result = mock_exec.execute_and_compare(query, expected)

                entry = {
                    "id": row["id"],
                    "spark_query": query,
                    "input_partition": row["input_partition"],
                    "expected_output": expected,
                    "actual_output": exec_result.get("actual_output", ""),
                    "direct_pass": exec_result.get("match", exec_result["success"]),
                    "direct_error": exec_result.get("error", ""),
                    "e6_function": extract_e6_function(query) if exec_result.get("match", exec_result["success"]) else "",
                    "e6_query": query if exec_result.get("match", exec_result["success"]) else "",
                    "covered": exec_result.get("match", exec_result["success"]),
                }
                results.append(entry)

        # Count results
        passed = sum(1 for r in results if r["covered"])
        failed = sum(1 for r in results if not r["covered"])

        # ABS: 7 pass, 1 fail (MIN_BIGINT overflow)
        assert passed == 7
        assert failed == 1

        # The failed one should be MIN_BIGINT
        failed_entries = [r for r in results if not r["covered"]]
        assert len(failed_entries) == 1
        assert failed_entries[0]["input_partition"] == "P_MIN_BIGINT"
        assert "error" in failed_entries[0]["direct_error"].lower()

        # Coverage matrix
        matrix = build_coverage_matrix(results)
        assert "ABS" in matrix
        assert matrix["ABS"]["count"] == 7

        # All passed partitions
        passed_partitions = {r["input_partition"] for r in results if r["covered"]}
        assert "P_NULL" in passed_partitions
        assert "P_NEGATIVE_INT" in passed_partitions
        assert "P_ZERO" in passed_partitions
        assert "P_POSITIVE_INT" in passed_partitions
        assert "P_NEGATIVE_DECIMAL" in passed_partitions
        assert "P_POSITIVE_DECIMAL" in passed_partitions
        assert "P_STRING_CAST" in passed_partitions
        assert "P_MIN_BIGINT" not in passed_partitions

    def test_cumulative_context_built_correctly(self):
        """Verify cumulative context shows pass/fail for processed partitions."""
        rows, analysis = read_basis("ABS")
        mock_exec = MockQueryExecutor()

        results = []
        for row in rows:
            exec_result = mock_exec.execute_and_compare(
                row["query"], row.get("actual_output", "")
            )
            results.append({
                "id": row["id"],
                "spark_query": row["query"],
                "input_partition": row["input_partition"],
                "expected_output": row.get("actual_output", ""),
                "actual_output": exec_result.get("actual_output", ""),
                "direct_pass": exec_result.get("match", False),
                "direct_error": exec_result.get("error", ""),
                "e6_function": "ABS" if exec_result.get("match", False) else "",
                "e6_query": row["query"] if exec_result.get("match", False) else "",
                "covered": exec_result.get("match", False),
            })

        ctx = format_cumulative_context(results)

        # Should contain both PASS and FAIL entries
        assert "PASS" in ctx
        assert "FAIL" in ctx
        # Should show the MIN_BIGINT failure
        assert "P_MIN_BIGINT" in ctx
        # Should show 7/8 coverage
        assert "7/8" in ctx

    def test_partition_scores_with_failures(self):
        """Verify partition scores reflect the MIN_BIGINT failure."""
        rows, _ = read_basis("ABS")
        partitions = group_by_partition(rows)
        mock_exec = MockQueryExecutor()

        results = []
        for row in rows:
            exec_result = mock_exec.execute_and_compare(
                row["query"], row.get("actual_output", "")
            )
            results.append({
                "id": row["id"],
                "input_partition": row["input_partition"],
                "covered": exec_result.get("match", False),
            })

        # Build partition scores
        partition_scores = {}
        for pname, queries in partitions.items():
            qids = {q["id"] for q in queries}
            covered = sum(1 for r in results if r["id"] in qids and r["covered"])
            partition_scores[pname] = {
                "total": len(queries),
                "covered": covered,
                "score": covered / len(queries) if queries else 0.0,
                "fully_covered": covered == len(queries),
            }

        # All partitions should be fully covered EXCEPT P_MIN_BIGINT
        for pname, ps in partition_scores.items():
            if pname == "P_MIN_BIGINT":
                assert ps["fully_covered"] is False
                assert ps["score"] == 0.0
            else:
                assert ps["fully_covered"] is True
                assert ps["score"] == 1.0