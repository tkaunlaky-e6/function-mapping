"""Verify basis_table.csv by re-running every query and comparing actual_output."""

import csv
import sys
from pathlib import Path

# Set JAVA_HOME if needed
import os
os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")

from dialect_mapper.basis.tools import execute_spark_sql


def verify(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty.")
        sys.exit(1)

    passed = 0
    failed = 0

    print(f"Verifying {len(rows)} basis queries from {csv_path}\n")

    for row in rows:
        query = row["query"]
        expected = row["actual_output"]
        result = execute_spark_sql.invoke({"query": query})

        match = result.strip() == expected.strip()
        status = "PASS" if match else "FAIL"

        if match:
            passed += 1
        else:
            failed += 1

        if not match:
            print(f"  {status} | {row['id']} | {query}")
            print(f"         expected: {repr(expected)}")
            print(f"         got:      {repr(result)}")
        else:
            print(f"  {status} | {row['id']} | {query} => {result}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(rows)} total")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_basis.py <path_to_basis_table.csv>")
        print("Example: python verify_basis.py output/concat_ws/basis_table.csv")
        sys.exit(1)

    verify(sys.argv[1])