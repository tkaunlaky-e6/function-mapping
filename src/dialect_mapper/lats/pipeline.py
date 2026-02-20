"""
Pipeline: map Spark SQL functions to e6data by executing basis queries.

Takes basis_table.csv + basis_analysis.json as input.
For each basis query:
  1. EXECUTE the Spark SQL on e6data and compare output with expected
  2. For failures, use LATS tree search to find an e6 translation
     that produces the SAME output
  3. Group by partition — score = fraction of queries in partition that match
  4. Build coverage matrix + minimal e6 function set

The key difference from planner-only validation: we EXECUTE the SQL and
check the actual output against the expected value from basis_table.csv.

Usage:
    # Execute on real e6data:
    python -m dialect_mapper.lats.pipeline FACTORIAL

    # With LATS translation for failures:
    python -m dialect_mapper.lats.pipeline FACTORIAL --lats

    # Stub mode (no network, all pass — for testing pipeline logic):
    python -m dialect_mapper.lats.pipeline FACTORIAL --stub
"""

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Read basis data
# ---------------------------------------------------------------------------

def read_basis(func_name: str) -> tuple[list[dict], dict]:
    """Read basis_table.csv and basis_analysis.json for a Spark function."""
    func_dir = OUTPUT_DIR / func_name.lower()

    csv_path = func_dir / "basis_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No basis_table.csv at {csv_path}")

    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    analysis = {}
    for name in ["basis_analysis_new.json", "basis_analysis.json"]:
        p = func_dir / name
        if p.exists():
            with open(p, "r") as f:
                analysis = json.load(f)
            break

    return rows, analysis


def group_by_partition(rows: list[dict]) -> dict[str, list[dict]]:
    """Group basis queries by input_partition."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["input_partition"]].append(row)
    return dict(groups)


# ---------------------------------------------------------------------------
# Extract e6 function name from SQL
# ---------------------------------------------------------------------------

def extract_e6_function(sql: str) -> str:
    """Parse the primary function from a SQL query."""
    sql_upper = sql.strip().upper()

    if sql_upper.startswith("SELECT "):
        body = sql.strip()[7:].strip()
    else:
        body = sql.strip()

    if body.upper().startswith("CASE "):
        return "CASE_EXPRESSION"

    match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(', body)
    if match:
        return match.group(1).upper()

    if body.upper().startswith("CAST("):
        return "CAST"

    return "DIRECT"


# ---------------------------------------------------------------------------
# Coverage matrix + greedy set cover
# ---------------------------------------------------------------------------

def build_coverage_matrix(results: list[dict]) -> dict:
    """Group covered queries by e6 function."""
    matrix = {}
    for r in results:
        func = r.get("e6_function", "")
        if not func or not r.get("covered", False):
            continue
        if func not in matrix:
            matrix[func] = {"covered_ids": [], "count": 0}
        matrix[func]["covered_ids"].append(int(r["id"]))
        matrix[func]["count"] += 1
    return matrix


def greedy_set_cover(matrix: dict, total_ids: set[int]) -> list[str]:
    """Find minimal set of e6 functions covering all queries."""
    remaining = set(total_ids)
    selected = []

    while remaining:
        best_func = None
        best_cover = set()
        for func, info in matrix.items():
            covered = remaining & set(info["covered_ids"])
            if len(covered) > len(best_cover):
                best_func = func
                best_cover = covered

        if not best_func:
            break

        selected.append(best_func)
        remaining -= best_cover

    return selected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_coverage_test(
    func_name: str,
    catalog: str = "nl2sql",
    database: str = "tpcds_1",
    use_stub: bool = False,
    use_lats: bool = False,
    model_name: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """
    Execute all basis queries on e6data and compare outputs.

    Phase 1: Direct execution — run each Spark query as-is on e6data,
             compare actual output with expected from basis_table.csv.
    Phase 2: LATS translation — for failures, tree search for e6 SQL
             that produces the correct output.
    Phase 3: Partition scoring + coverage matrix + minimal function set.

    Args:
        func_name: Spark function name (e.g., "FACTORIAL")
        catalog: e6data catalog
        database: e6data database
        use_stub: True → stub executor (all pass, for testing logic)
        use_lats: True → LATS translation for failures
        model_name: Anthropic model for LATS

    Returns:
        Full report dict (also written to output/<func>/)
    """
    from .tools import PlannerValidator, QueryExecutor
    from .prompts import format_translation_context, format_cumulative_context

    basis_rows, analysis = read_basis(func_name)
    partitions = group_by_partition(basis_rows)

    # Set up executor
    if use_stub:
        executor = PlannerValidator()  # stub: always passes
    else:
        executor = QueryExecutor(catalog=catalog, database=database)

    # ── Phase 1: Direct execution ─────────────────────────────────────────

    results = []
    direct_pass = 0
    direct_fail = 0

    print(f"\n{'='*72}")
    print(f"  {func_name} — {len(basis_rows)} queries, {len(partitions)} partitions")
    print(f"  Executor: {'STUB' if use_stub else 'REAL (e6data)'}")
    print(f"  LATS: {'ON' if use_lats else 'OFF'}")
    print(f"{'='*72}")
    print(f"\n  Phase 1: Direct execution on e6data\n")

    for partition_name, queries in partitions.items():
        partition_passed = 0
        partition_total = len(queries)

        for row in queries:
            query = row["query"]
            expected = row.get("actual_output", "")

            # Execute and compare
            if use_stub:
                exec_result = {"success": True, "error": "", "actual_output": expected,
                               "expected_output": expected, "match": True}
            else:
                exec_result = executor.execute_and_compare(query, expected)

            entry = {
                "id": row["id"],
                "spark_query": query,
                "input_partition": row["input_partition"],
                "expected_output": expected,
                "actual_output": exec_result.get("actual_output", ""),
                "codomain_class": row.get("codomain_class", ""),
                "reasoning": row.get("reasoning", ""),
                "direct_pass": exec_result["match"] if "match" in exec_result else exec_result["success"],
                "direct_error": exec_result.get("error", ""),
                "e6_query": query if exec_result.get("match", exec_result["success"]) else "",
                "e6_function": extract_e6_function(query) if exec_result.get("match", exec_result["success"]) else "",
                "lats_pass": None,
                "lats_score": None,
                "lats_reasoning": "",
                "covered": exec_result.get("match", exec_result["success"]),
            }

            if entry["direct_pass"]:
                direct_pass += 1
                partition_passed += 1
            else:
                direct_fail += 1

            results.append(entry)

        # Print partition summary
        tag = "PASS" if partition_passed == partition_total else "FAIL"
        print(f"  [{tag}] {partition_name:<22} {partition_passed}/{partition_total} queries match")
        for entry in results[-partition_total:]:
            if not entry["direct_pass"]:
                err_short = entry["direct_error"][:65]
                print(f"         #{entry['id']:>2} {err_short}")

    # ── Phase 2: LATS translation for failures ────────────────────────────

    lats_translated = 0
    lats_failed = 0

    if use_lats and direct_fail > 0:
        print(f"\n  Phase 2: LATS translation ({direct_fail} failed queries)\n")

        from langchain_anthropic import ChatAnthropic
        from .graph import run_lats

        llm = ChatAnthropic(model=model_name, temperature=0)

        # For LATS, use QueryExecutor as the "planner" — it executes and
        # compares output, so LATS gets output-match as its success signal
        for entry in results:
            if entry["direct_pass"]:
                continue

            # Get all queries in this partition for context
            partition_queries = partitions.get(entry["input_partition"], [])
            query_row = {
                "query": entry["spark_query"],
                "input_partition": entry["input_partition"],
                "actual_output": entry["expected_output"],
                "codomain_class": entry["codomain_class"],
                "reasoning": entry["reasoning"],
            }
            domain_context = format_translation_context(
                analysis, query_row, partition_queries
            )

            # Add cumulative context — what worked/failed for other partitions
            cumulative = format_cumulative_context(results)
            if cumulative:
                domain_context = domain_context + "\n\n" + cumulative

            # Set expected output on executor so LATS validate() checks it
            if hasattr(executor, 'expected_output'):
                executor.expected_output = entry["expected_output"]

            print(f"  #{entry['id']} {entry['input_partition']}: {entry['spark_query'][:50]}...")

            try:
                lats_result = run_lats(
                    llm=llm,
                    original_sql=entry["spark_query"],
                    error_message=entry["direct_error"],
                    calcite_idioms=domain_context,
                    catalog=catalog,
                    database=database,
                    planner=executor,
                    num_candidates=3,
                    total_queries=len(basis_rows),
                )

                best_sql = lats_result["best_sql"]
                entry["e6_query"] = best_sql
                entry["e6_function"] = extract_e6_function(best_sql) if best_sql else ""
                entry["lats_pass"] = lats_result["is_solved"]
                entry["lats_score"] = lats_result["score"]
                entry["covered"] = lats_result["is_solved"]

                # Extract reasoning
                root = lats_result["root"]
                best_node = root.get_best_solution()
                if best_node and best_node.reflection:
                    entry["lats_reasoning"] = best_node.reflection.reflections

                if lats_result["is_solved"]:
                    lats_translated += 1
                    print(f"    SOLVED → {entry['e6_function']}: {best_sql[:55]}")
                else:
                    lats_failed += 1
                    print(f"    UNSOLVED (best score: {lats_result['score']:.2f})")

            except Exception as e:
                logger.error(f"LATS error for #{entry['id']}: {e}")
                entry["lats_pass"] = False
                entry["lats_score"] = 0
                lats_failed += 1
                print(f"    ERROR: {e}")

    # ── Phase 3: Partition scoring + coverage ─────────────────────────────

    total_covered = direct_pass + lats_translated
    coverage = total_covered / len(basis_rows) if basis_rows else 0.0
    all_ids = {int(r["id"]) for r in basis_rows}

    coverage_matrix = build_coverage_matrix(results)
    minimal_set = greedy_set_cover(coverage_matrix, all_ids)
    uncovered_ids = all_ids - {
        qid for info in coverage_matrix.values() for qid in info["covered_ids"]
    }
    uncovered_partitions = list({
        r["input_partition"] for r in results if int(r["id"]) in uncovered_ids
    })

    # Partition-level scores
    partition_scores = {}
    for pname, queries in partitions.items():
        qids = {int(q["id"]) for q in queries}
        covered_in_partition = sum(
            1 for r in results
            if int(r["id"]) in qids and r["covered"]
        )
        partition_scores[pname] = {
            "total": len(queries),
            "covered": covered_in_partition,
            "score": covered_in_partition / len(queries) if queries else 0.0,
            "fully_covered": covered_in_partition == len(queries),
        }

    # ── Print summary ─────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print(f"  COVERAGE: {func_name}")
    print(f"{'─'*72}")
    print(f"  Total queries:         {len(basis_rows)}")
    print(f"  Direct pass:           {direct_pass}")
    print(f"  Direct fail:           {direct_fail}")
    if use_lats:
        print(f"  LATS translated:       {lats_translated}")
        print(f"  LATS failed:           {lats_failed}")
    print(f"  ───────────────────────────")
    print(f"  Covered:               {total_covered}/{len(basis_rows)}")
    print(f"  Coverage:              {coverage:.1%}")

    print(f"\n  Partition scores:")
    for pname, ps in partition_scores.items():
        tag = "FULL" if ps["fully_covered"] else f"{ps['score']:.0%} "
        print(f"    {pname:<22} {ps['covered']}/{ps['total']}  [{tag}]")

    if coverage_matrix:
        print(f"\n  e6 functions used:")
        for func, info in sorted(coverage_matrix.items(), key=lambda x: -x[1]["count"]):
            print(f"    {func:<25} covers {info['count']} queries")

    if minimal_set:
        print(f"\n  Minimal function set:  {minimal_set}")

    if uncovered_partitions:
        print(f"  Uncovered partitions:  {uncovered_partitions}")

    print(f"{'='*72}\n")

    # ── Build report ──────────────────────────────────────────────────────

    report = {
        "function": func_name,
        "total": len(basis_rows),
        "direct_pass": direct_pass,
        "direct_fail": direct_fail,
        "lats_translated": lats_translated,
        "lats_failed": lats_failed,
        "coverage": coverage,
        "coverage_matrix": coverage_matrix,
        "minimal_function_set": minimal_set,
        "uncovered_partitions": uncovered_partitions,
        "partition_scores": partition_scores,
        "results": results,
    }

    _write_report(func_name, report, analysis)

    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_report(func_name: str, report: dict, analysis: dict):
    """Write e6_coverage.csv and e6_mapping.json."""
    out_dir = OUTPUT_DIR / func_name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "e6_coverage.csv"
    fieldnames = [
        "id", "spark_query", "input_partition", "expected_output",
        "actual_output", "codomain_class", "direct_pass", "direct_error",
        "e6_function", "e6_query", "lats_pass", "lats_score", "covered",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in report["results"]:
            writer.writerow(r)

    # JSON
    json_path = out_dir / "e6_mapping.json"

    translations = []
    for r in report["results"]:
        entry = {
            "id": int(r["id"]),
            "spark_query": r["spark_query"],
            "input_partition": r["input_partition"],
            "e6_query": r["e6_query"],
            "e6_function": r["e6_function"],
            "covered": r["covered"],
            "reasoning": r.get("lats_reasoning", "") or r.get("reasoning", ""),
        }
        translations.append(entry)

    mapping = {
        "spark_function": report["function"],
        "total_basis_queries": report["total"],
        "overall_coverage": report["coverage"],
        "minimal_function_set": report.get("minimal_function_set", []),
        "uncovered_partitions": report.get("uncovered_partitions", []),
        "coverage_matrix": {
            func: {
                "covered_queries": info["covered_ids"],
                "count": info["count"],
                "coverage": info["count"] / report["total"] if report["total"] else 0,
            }
            for func, info in report.get("coverage_matrix", {}).items()
        },
        "translations": translations,
    }

    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2, default=str)

    print(f"  Written: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  Written: {json_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Map Spark SQL functions to e6data by executing basis queries"
    )
    parser.add_argument("function", help="Spark function name (e.g., FACTORIAL, CONCAT_WS)")
    parser.add_argument("--catalog", default="nl2sql", help="e6data catalog")
    parser.add_argument("--database", default="tpcds_1", help="e6data database")
    parser.add_argument("--stub", action="store_true",
                        help="Stub mode (no network, all pass)")
    parser.add_argument("--lats", action="store_true",
                        help="Use LATS for translating failed queries")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929",
                        help="Anthropic model for LATS")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
    )

    run_coverage_test(
        func_name=args.function,
        catalog=args.catalog,
        database=args.database,
        use_stub=args.stub,
        use_lats=args.lats,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()