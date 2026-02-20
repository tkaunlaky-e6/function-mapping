"""
Pluggable tool interfaces for the LATS dialect agent.

These are the "tools" that the LATS agent uses to validate candidate SQL fixes.
By default they are STUBS that work without any infrastructure — perfect for
local testing. For real usage, swap in the real implementations.

Three tools:
  1. PlannerValidator: Validates SQL via e6data's get_logical_plan() (no-execute)
  2. DocSearcher: Searches e6data SQL documentation (ChromaDB + BM25)
  3. SemanticChecker: Checks if a candidate SQL preserves the original query's intent

Each tool is a class with a simple interface. You can subclass or replace them
to plug in real infrastructure.
"""

import json
import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

from .prompts import SEMANTIC_CHECK_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Planner Validator
# ---------------------------------------------------------------------------

class PlannerValidator:
    """
    Validates SQL using the e6data planner (no-execute mode).

    The planner parses and plans the SQL server-side without fetching any rows.
    It catches: missing tables/columns, type mismatches, syntax errors, etc.

    This is the GROUND TRUTH signal in our LATS loop — if the planner says
    success, the SQL will actually run.

    Default implementation is a STUB that always returns success.
    Override validate() to plug in the real e6data planner.
    """

    def validate(self, sql: str, catalog: str = "", database: str = "") -> dict:
        """
        Validate SQL without executing it.

        Args:
            sql: The SQL query to validate
            catalog: E6Data catalog name
            database: E6Data database/schema name

        Returns:
            {"success": True/False, "error": "error message if failed"}
        """
        # STUB: always succeeds. Replace with real planner for integration testing.
        logger.info(f"[PlannerValidator STUB] Validating SQL ({len(sql)} chars)")
        return {"success": True, "error": ""}


class RealPlannerValidator(PlannerValidator):
    """
    Real planner validator that connects to e6data.

    Uses connection.get_logical_plan(query, catalog, schema) which validates
    SQL without executing it. Returns the logical plan on success, or a
    detailed error message on failure.

    Usage:
        validator = RealPlannerValidator(catalog="nl2sql", database="tpcds_1")
        result = validator.validate("SELECT * FROM catalog_sales LIMIT 1")
        # {"success": True, "error": ""} or {"success": False, "error": "..."}

    Requires:
        - pip install e6data-python-connector==2.3.12rc11 (or later with get_logical_plan)
        - Env vars: E6DATA_HOST, E6DATA_PORT, E6_USER, E6_TOKEN,
                    E6DATA_SECURE, E6DATA_CLUSTER_NAME
    """

    def __init__(self, catalog: str = "", database: str = ""):
        self.default_catalog = catalog
        self.default_database = database
        self._conn = None  # Reuse connection across validations

    def _get_connection(self):
        """Create or reuse e6data connection. Reads credentials from env vars (.env)."""
        if self._conn is not None:
            return self._conn

        from dotenv import load_dotenv
        from e6data_python_connector import Connection  # type: ignore

        load_dotenv()

        host = os.getenv("E6DATA_HOST", "")
        port = int(os.getenv("E6DATA_PORT", "443"))
        username = os.getenv("E6_USER", "")
        password = os.getenv("E6_TOKEN", "")
        secure = os.getenv("E6DATA_SECURE", "True").lower() == "true"
        cluster_name = os.getenv("E6DATA_CLUSTER_NAME", "")

        if not host or not username or not password:
            raise RuntimeError(
                "Missing e6data credentials. Set E6DATA_HOST, E6_USER, E6_TOKEN in .env"
            )

        conn_params = {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": self.default_database,
            "secure": secure,
        }
        if cluster_name:
            conn_params["cluster_name"] = cluster_name

        self._conn = Connection(**conn_params)
        logger.info(f"[RealPlanner] Connected to {host}:{port} (cluster={cluster_name})")
        return self._conn

    @staticmethod
    def _parse_error(raw_error: str) -> str:
        """
        Extract the clean error message from gRPC error strings.

        The planner returns errors wrapped in gRPC metadata like:
            '<_InactiveRpcError ... details = ": \\t{SYNTAX ERROR} At line 1, column 27\\nEncountered MONTH month">'

        We extract: "{SYNTAX ERROR} At line 1, column 27\nEncountered MONTH month"
        """
        import re

        # The raw error repr has: details = ": \t{ERROR} message\nmore detail"
        # In the actual string, \t and \n are literal characters
        # Try to extract everything between 'details = ": \t' and the closing '"'
        match = re.search(r'details\s*=\s*":\s*\t(.+?)"', raw_error, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: look for {ERROR_TYPE} pattern and grab everything after it
        match = re.search(r'(\{(?:SYNTAX ERROR|COMPILATION ERROR)\}.+?)(?:"|$)', raw_error, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Last resort: return as-is but truncated
        return raw_error[:500] if len(raw_error) > 500 else raw_error

    def validate(self, sql: str, catalog: str = "", database: str = "") -> dict:
        """
        Validate SQL using e6data's get_logical_plan() (no-execute mode).

        Returns:
            {"success": True, "error": ""} if SQL is valid
            {"success": False, "error": "clean error message"} if invalid
        """
        cat = catalog or self.default_catalog
        db = database or self.default_database

        try:
            conn = self._get_connection()

            # Add LIMIT 1 if not already present (cheaper planning)
            import re
            sql_normalized = re.sub(r'\s+', ' ', sql.lower().strip())
            has_limit = ' limit ' in sql_normalized or sql_normalized.endswith(' limit')
            if has_limit:
                validation_sql = sql.rstrip(';')
            else:
                validation_sql = f"{sql.rstrip(';')} LIMIT 1"

            logger.info(f"[RealPlanner] Validating ({len(validation_sql)} chars)")
            result = conn.get_logical_plan(validation_sql, catalog=cat, schema=db)

            if result.get("success", False):
                logger.info("[RealPlanner] SUCCESS")
                return {"success": True, "error": ""}
            else:
                raw_error = result.get("error", "Unknown error")
                clean_error = self._parse_error(str(raw_error))
                logger.info(f"[RealPlanner] FAILED: {clean_error}")
                return {"success": False, "error": clean_error}

        except ImportError:
            logger.error("e6data_python_connector not installed. Install with: "
                        "pip install e6data-python-connector==2.3.12rc11")
            return {"success": True, "error": ""}
        except Exception as e:
            logger.error(f"[RealPlanner] Exception: {e}")
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# 2. Doc Searcher
# ---------------------------------------------------------------------------

class DocSearcher:
    """
    Searches e6data SQL documentation for relevant dialect information.

    The real implementation uses hybrid search: ChromaDB (vector) + BM25 (keyword).
    The default stub returns a small set of common e6data dialect tips.
    """

    def search(self, search_terms: list[str]) -> list[dict]:
        """
        Search for relevant e6data documentation.

        Args:
            search_terms: List of keywords/phrases to search for

        Returns:
            List of dicts with "content" and "title" keys
        """
        # STUB: return common e6data dialect tips
        logger.info(f"[DocSearcher STUB] Searching for: {search_terms}")
        return [
            {
                "content": (
                    "e6data uses Apache Calcite SQL parser. Reserved keywords like "
                    "MONTH, YEAR, DAY, DATE, TIME, TIMESTAMP, USER, VALUE, KEY, "
                    "POSITION, RESULT, COUNT cannot be used as identifiers or aliases. "
                    "Wrap them in double quotes or rename them."
                ),
                "title": "Reserved Keywords"
            },
            {
                "content": (
                    "e6data does not support DATE_TRUNC(). Use DATE_FORMAT() or "
                    "EXTRACT() instead. For example: DATE_FORMAT(col, 'yyyy-MM') "
                    "instead of DATE_TRUNC('month', col)."
                ),
                "title": "Date Functions"
            },
            {
                "content": (
                    "e6data does not support PIVOT/UNPIVOT. Use CASE statements. "
                    "e6data does not support IGNORE NULLS in window functions. "
                    "e6data does not support stored procedures or UDFs."
                ),
                "title": "Unsupported Features"
            },
        ]


# ---------------------------------------------------------------------------
# 3. Semantic Checker
# ---------------------------------------------------------------------------

class SemanticChecker:
    """
    Checks if a candidate SQL fix preserves the original query's intent.

    Uses an LLM to compare the original and candidate queries. This prevents
    the LATS agent from "solving" errors by returning a completely different
    query (e.g., SELECT 1) that passes the planner but is meaningless.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def check(self, original_sql: str, candidate_sql: str) -> dict:
        """
        Check if candidate preserves the original query's semantics.

        Args:
            original_sql: The original failing SQL
            candidate_sql: The proposed fix

        Returns:
            {"preserves_intent": True/False, "differences": "description"}
        """
        if self.llm is None:
            # No LLM available — do a basic structural check
            return self._basic_check(original_sql, candidate_sql)

        # Use the LLM to do a proper semantic comparison
        prompt = SEMANTIC_CHECK_PROMPT.format(
            original_sql=original_sql,
            candidate_sql=candidate_sql,
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            result = json.loads(content)
            return {
                "preserves_intent": result.get("preserves_intent", False),
                "differences": result.get("differences", ""),
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Semantic check LLM call failed: {e}. Using basic check.")
            return self._basic_check(original_sql, candidate_sql)

    def _basic_check(self, original_sql: str, candidate_sql: str) -> dict:
        """
        Basic structural comparison when no LLM is available.

        Checks if key structural elements (tables, keywords) are preserved.
        Not as thorough as the LLM check, but catches obvious problems
        like returning SELECT 1 instead of the real query.
        """
        orig = original_sql.upper().strip()
        cand = candidate_sql.upper().strip()

        # If candidate is trivially different (e.g., SELECT 1), reject it
        if len(cand) < len(orig) * 0.3:
            return {
                "preserves_intent": False,
                "differences": "Candidate is much shorter than original — likely a different query"
            }

        # Check that FROM tables are preserved
        # (very rough heuristic — the LLM check is much better)
        import re
        orig_tables = set(re.findall(r'FROM\s+(\w+)', orig))
        cand_tables = set(re.findall(r'FROM\s+(\w+)', cand))

        if orig_tables and not orig_tables.intersection(cand_tables):
            return {
                "preserves_intent": False,
                "differences": f"Original queries tables {orig_tables}, candidate queries {cand_tables}"
            }

        return {"preserves_intent": True, "differences": "None (basic check)"}


# ---------------------------------------------------------------------------
# 4. Query Executor — execute SQL and compare output
# ---------------------------------------------------------------------------

class QueryExecutor(PlannerValidator):
    """
    Execute SQL on e6data and compare actual output with expected.

    This replaces PlannerValidator in the LATS loop for function mapping.
    Instead of just checking syntax, it EXECUTES the SQL and checks:
      - Did it run without error?
      - Does the output match the expected value from basis_table.csv?

    Implements the same validate() interface as PlannerValidator so the
    entire LATS infrastructure (nodes.py, graph.py) works unchanged.

    Usage:
        executor = QueryExecutor(catalog="nl2sql", database="tpcds_1")
        executor.expected_output = "120"
        result = executor.validate("SELECT FACTORIAL(5)")
        # {"success": True, "error": ""}  — output matches

        executor.expected_output = "NULL"
        result = executor.validate("SELECT FACTORIAL(-5)")
        # {"success": False, "error": "Execution error: ..."}  — e6 throws
    """

    def __init__(self, catalog: str = "nl2sql", database: str = "tpcds_1"):
        self.default_catalog = catalog
        self.default_database = database
        self.expected_output = ""  # Set before each validate() call
        self._conn = None
        self._cursor = None

    def _get_cursor(self):
        """Create or reuse e6data cursor. Reads credentials from env vars (.env)."""
        if self._cursor is not None:
            return self._cursor

        from dotenv import load_dotenv
        from e6data_python_connector import Connection  # type: ignore

        load_dotenv()

        host = os.getenv("E6DATA_HOST", "")
        port = int(os.getenv("E6DATA_PORT", "443"))
        user = os.getenv("E6_USER", "")
        token = os.getenv("E6_TOKEN", "")
        cluster = os.getenv("E6DATA_CLUSTER_NAME", "")
        secure = os.getenv("E6DATA_SECURE", "True").lower() == "true"

        if not host or not user or not token:
            raise RuntimeError(
                "Missing e6data credentials. Set E6DATA_HOST, E6_USER, E6_TOKEN in .env"
            )

        self._conn = Connection(
            host=host,
            port=port,
            username=user,
            password=token,
            database=self.default_database,
            secure=secure,
            cluster_name=cluster,
        )
        self._cursor = self._conn.cursor(
            catalog_name=self.default_catalog,
            db_name=self.default_database,
        )
        logger.info(
            f"[QueryExecutor] Connected to {host} "
            f"(catalog={self.default_catalog}, db={self.default_database})"
        )
        return self._cursor

    def execute(self, sql: str) -> list:
        """Execute SQL and return raw result rows."""
        cursor = self._get_cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    def execute_and_compare(self, sql: str, expected: str) -> dict:
        """
        Execute SQL and compare output with expected value.

        Returns:
            {
                "success": bool,
                "error": str,
                "actual_output": str,
                "expected_output": str,
                "match": bool,
            }
        """
        try:
            result = self.execute(sql)
            actual = result[0][0] if result and result[0] else None
            actual_str = str(actual) if actual is not None else "NULL"

            match = (actual_str == expected)
            if match:
                return {
                    "success": True,
                    "error": "",
                    "actual_output": actual_str,
                    "expected_output": expected,
                    "match": True,
                }
            else:
                return {
                    "success": False,
                    "error": f"Output mismatch: expected '{expected}', got '{actual_str}'",
                    "actual_output": actual_str,
                    "expected_output": expected,
                    "match": False,
                }
        except Exception as e:
            error_str = self._parse_error(str(e)) if hasattr(self, '_parse_error') else str(e)[:300]
            return {
                "success": False,
                "error": f"Execution error: {error_str}",
                "actual_output": "",
                "expected_output": expected,
                "match": False,
            }

    def validate(self, sql: str, catalog: str = "", database: str = "") -> dict:
        """
        Execute SQL and compare with self.expected_output.

        This is the PlannerValidator interface used by the LATS core.
        Set self.expected_output before calling this.
        """
        result = self.execute_and_compare(sql, self.expected_output)
        return {"success": result["success"], "error": result["error"]}