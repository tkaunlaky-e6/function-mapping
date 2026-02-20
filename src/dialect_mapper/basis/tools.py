"""Custom tools for the basis agent. Only Spark SQL — file I/O is handled by middleware."""

import os
from pathlib import Path
from langchain.tools import tool

_spark = None


def _get_spark():
    """Get or create a local SparkSession (singleton)."""
    global _spark
    if _spark is not None:
        return _spark

    from pyspark.sql import SparkSession

    _spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("basis-agent")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    _spark.sparkContext.setLogLevel("WARN")

    # Load test table if CSV exists
    csv_path = os.environ.get(
        "BASIS_CSV_PATH",
        str(Path(__file__).resolve().parents[3] / "output" / "web_sales.csv"),
    )
    if Path(csv_path).exists():
        df = _spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)
        df.createOrReplaceTempView("recent_web_sales")

    return _spark


@tool
def execute_spark_sql(query: str) -> str:
    """Execute a SQL query on PySpark local and return the result.

    Args:
        query: A valid Spark SQL SELECT query.

    Returns:
        The result value as a string, or an error message starting with 'ERROR:'.
    """
    spark = _get_spark()
    try:
        rows = spark.sql(query).collect()
        if not rows:
            return "NULL"
        value = rows[0][0]
        if value is None:
            return "NULL"
        if type(value).__name__ == "Decimal":
            value = float(value)
        return str(value)
    except Exception as e:
        return f"ERROR: {str(e)[:300]}"