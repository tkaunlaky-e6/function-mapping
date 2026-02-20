"""Tests for the Spark SQL execution tool."""

import pytest
from dialect_mapper.basis.tools import execute_spark_sql


class TestExecuteSparkSql:
    """Test the execute_spark_sql tool on various SQL functions."""

    def test_simple_select(self):
        assert execute_spark_sql.invoke({"query": "SELECT 1"}) == "1"

    def test_null_result(self):
        assert execute_spark_sql.invoke({"query": "SELECT NULL"}) == "NULL"

    def test_string_result(self):
        assert execute_spark_sql.invoke({"query": "SELECT 'hello'"}) == "hello"

    def test_abs_positive(self):
        assert execute_spark_sql.invoke({"query": "SELECT ABS(-5)"}) == "5"

    def test_abs_zero(self):
        assert execute_spark_sql.invoke({"query": "SELECT ABS(0)"}) == "0"

    def test_abs_null(self):
        assert execute_spark_sql.invoke({"query": "SELECT ABS(NULL)"}) == "NULL"

    def test_concat_ws(self):
        assert execute_spark_sql.invoke({"query": "SELECT CONCAT_WS(',', 'a', 'b', 'c')"}) == "a,b,c"

    def test_concat_ws_null_sep(self):
        assert execute_spark_sql.invoke({"query": "SELECT CONCAT_WS(NULL, 'a', 'b')"}) == "NULL"

    def test_nvl_null(self):
        assert execute_spark_sql.invoke({"query": "SELECT NVL(NULL, 42)"}) == "42"

    def test_nvl_non_null(self):
        assert execute_spark_sql.invoke({"query": "SELECT NVL(10, 42)"}) == "10"

    def test_power(self):
        assert float(execute_spark_sql.invoke({"query": "SELECT POWER(2, 3)"})) == 8.0

    def test_substring(self):
        assert execute_spark_sql.invoke({"query": "SELECT SUBSTRING('hello', 2, 3)"}) == "ell"

    def test_invalid_sql_returns_error(self):
        result = execute_spark_sql.invoke({"query": "SELECT FROM WHERE"})
        assert result.startswith("ERROR:")

    def test_decimal_result(self):
        result = execute_spark_sql.invoke({"query": "SELECT CAST(1.5 AS DECIMAL(10,2))"})
        assert float(result) == 1.5


class TestSparkTableLoaded:
    """Test that the web_sales CSV is loaded as a table."""

    def test_table_exists(self):
        result = execute_spark_sql.invoke({"query": "SELECT COUNT(*) FROM recent_web_sales"})
        assert not result.startswith("ERROR:"), f"Table not loaded: {result}"
        assert int(result) > 0

    def test_column_query(self):
        result = execute_spark_sql.invoke({
            "query": "SELECT ws_sales_price FROM recent_web_sales LIMIT 1"
        })
        assert not result.startswith("ERROR:")
        assert float(result) > 0
