"""Tests for the prompt templates."""

from dialect_mapper.basis.prompt import SYSTEM_PROMPT, HUMAN_PROMPT


class TestSystemPrompt:
    """Test the system prompt has all key mathematical concepts."""

    def test_has_equivalence_partitioning(self):
        assert "Equivalence" in SYSTEM_PROMPT
        assert "equivalence classes" in SYSTEM_PROMPT

    def test_has_spanning_definition(self):
        assert "Spanning" in SYSTEM_PROMPT
        assert "COVERING" in SYSTEM_PROMPT
        assert "DISJOINT" in SYSTEM_PROMPT

    def test_has_spanning_proof_requirement(self):
        assert "spanning proof" in SYSTEM_PROMPT.lower() or "Spanning Proof" in SYSTEM_PROMPT

    def test_has_output_range(self):
        assert "Output Range" in SYSTEM_PROMPT or "output range" in SYSTEM_PROMPT
        assert "TYPE BOUNDS" in SYSTEM_PROMPT
        assert "VALUE BOUNDS" in SYSTEM_PROMPT
        assert "OUTPUT-INPUT RELATIONSHIP" in SYSTEM_PROMPT

    def test_has_codomain_classes(self):
        assert "Codomain Classes" in SYSTEM_PROMPT
        assert "O_NULL" in SYSTEM_PROMPT
        assert "O_ZERO" in SYSTEM_PROMPT

    def test_has_surjectivity(self):
        assert "Surjectivity" in SYSTEM_PROMPT
        assert "coverage" in SYSTEM_PROMPT

    def test_has_completeness_definition(self):
        assert "Spanning + Surjectivity" in SYSTEM_PROMPT

    def test_has_worked_example(self):
        assert "POWER" in SYSTEM_PROMPT
        assert "POWER(2, 3)" in SYSTEM_PROMPT

    def test_has_variadic_partitioning(self):
        assert "VARIADIC" in SYSTEM_PROMPT
        assert "array" in SYSTEM_PROMPT.lower()

    def test_has_tool_references(self):
        assert "execute_spark_sql" in SYSTEM_PROMPT
        assert "read_file" in SYSTEM_PROMPT
        assert "write_file" in SYSTEM_PROMPT

    def test_has_workflow_steps(self):
        assert "STEP 1" in SYSTEM_PROMPT
        assert "STEP 2" in SYSTEM_PROMPT
        assert "STEP 3" in SYSTEM_PROMPT
        assert "STEP 4" in SYSTEM_PROMPT
        assert "STEP 5" in SYSTEM_PROMPT
        assert "STEP 6" in SYSTEM_PROMPT
        assert "STEP 7" in SYSTEM_PROMPT

    def test_has_output_format(self):
        assert "basis_table.csv" in SYSTEM_PROMPT
        assert "basis_analysis.json" in SYSTEM_PROMPT

    def test_has_back_solve(self):
        assert "back-solve" in SYSTEM_PROMPT.lower() or "BACK-SOLVE" in SYSTEM_PROMPT


class TestHumanPrompt:
    """Test the human prompt template."""

    def test_format_works(self):
        result = HUMAN_PROMPT.format(
            func_name="ABS",
            func_name_lower="abs",
            dialect="spark",
            documentation="ABS(x) returns absolute value",
            output_dir="/tmp/output",
        )
        assert "ABS" in result
        assert "spark" in result
        assert "/tmp/output/abs/basis_table.csv" in result

    def test_has_all_placeholders(self):
        assert "{func_name}" in HUMAN_PROMPT
        assert "{func_name_lower}" in HUMAN_PROMPT
        assert "{dialect}" in HUMAN_PROMPT
        assert "{documentation}" in HUMAN_PROMPT
        assert "{output_dir}" in HUMAN_PROMPT

    def test_no_unfilled_placeholders(self):
        result = HUMAN_PROMPT.format(
            func_name="NVL",
            func_name_lower="nvl",
            dialect="spark",
            documentation="docs",
            output_dir="/out",
        )
        assert "{" not in result