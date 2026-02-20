"""Basis synthesis agent using LangChain 1.0 create_agent + FilesystemMiddleware."""

import os
import logging
import httpx
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import create_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.filesystem import FilesystemBackend

from .prompt import SYSTEM_PROMPT, HUMAN_PROMPT
from .tools import execute_spark_sql

load_dotenv()

logger = logging.getLogger(__name__)

# Output directory: parents[3] = project root
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

# Doc URLs by dialect
DOC_URLS = {
    "spark": "https://spark.apache.org/docs/latest/api/sql/#_{func}",
}


def _fetch_docs(func_name: str, dialect: str) -> str:
    """Fetch function documentation from the web."""
    if dialect not in DOC_URLS:
        return f"No docs URL for {dialect}. Use your knowledge."

    url = DOC_URLS[dialect].format(func=func_name.lower())
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text[:15000]
        return f"HTTP {resp.status_code} fetching docs. Use your knowledge of {func_name}."
    except Exception as e:
        return f"Could not fetch docs: {e}. Use your knowledge of {func_name}."


def create_basis_agent(model: str = None):
    """Create the basis synthesis agent.

    Args:
        model: Model string (e.g. "anthropic:claude-sonnet-4-5-20250929").
               Defaults to ANTHROPIC_MODEL env var.

    Returns:
        Compiled agent graph.
    """
    if model is None:
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        model = f"anthropic:{model_name}"

    agent = create_agent(
        model=model,
        tools=[execute_spark_sql],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            FilesystemMiddleware(backend=FilesystemBackend()),
        ],
    )
    return agent


def run_basis_agent(
    func_name: str,
    dialect: str = "spark",
    model: str = None,
    output_dir: str = None,
) -> dict:
    """Run the basis agent for a SQL function.

    Args:
        func_name: SQL function name (e.g. "ABS", "NVL", "CONCAT_WS")
        dialect: SQL dialect (default: "databricks")
        model: Model string (optional, uses env var default)
        output_dir: Output directory (optional, uses default)

    Returns:
        The final agent state with messages.
    """
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)

    # Create output dir with template CSV
    func_dir = Path(output_dir) / func_name.lower()
    func_dir.mkdir(parents=True, exist_ok=True)
    csv_path = func_dir / "basis_table.csv"
    if not csv_path.exists():
        csv_path.write_text("id,query,input_partition,predicted_output,actual_output,codomain_class,reasoning\n")

    # Fetch documentation
    docs = _fetch_docs(func_name, dialect)

    # Build the human message
    human_msg = HUMAN_PROMPT.format(
        func_name=func_name.upper(),
        func_name_lower=func_name.lower(),
        dialect=dialect,
        documentation=docs,
        output_dir=output_dir,
    )

    # Create and run the agent
    agent = create_basis_agent(model=model)

    logger.info(f"Running basis agent for {func_name} ({dialect})")
    result = agent.invoke({
        "messages": [{"role": "user", "content": human_msg}],
    })

    logger.info(f"Agent finished. {len(result['messages'])} messages.")
    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m dialect_mapper.basis.agent <FUNCTION_NAME> [dialect]")
        print("Example: python -m dialect_mapper.basis.agent ABS")
        sys.exit(1)

    func = sys.argv[1].upper()
    dialect = sys.argv[2] if len(sys.argv) > 2 else "spark"

    result = run_basis_agent(func, dialect)
    print(f"\nDone. Check output/ directory for results.")