"""The same agent as agent.py, but the SDK's tool runner owns the loop.

Compare with agent.py: here there is no while-loop, no tool_result bookkeeping,
and the tool schemas are generated from the function signatures + docstrings.

Usage: uv run agent_runner.py "Which Maria is allergic to latex?"
"""

import sys

import anthropic
from anthropic import beta_tool

import tools
from agent import MAX_TURNS, MODEL, SYSTEM


def _call(name: str, tool_input: dict) -> str:
    # The runner turns a raised exception into an is_error tool_result for us.
    result, is_error = tools.run_tool(name, tool_input)
    if is_error:
        raise tools.ToolError(result)
    return result


@beta_tool
def search_patients(name: str) -> str:
    """Find patients whose name contains the given text. More than one patient can match.

    Args:
        name: Full or partial patient name.
    """
    return _call("search_patients", {"name": name})


@beta_tool
def get_lab_results(patient_id: str, test_name: str) -> str:
    """Get all results of one lab test for a patient, oldest first (empty list if none).

    Args:
        patient_id: ID from search_patients, e.g. P001.
        test_name: Lab test name, e.g. HbA1c, LDL, TSH.
    """
    return _call("get_lab_results", {"patient_id": patient_id, "test_name": test_name})


@beta_tool
def get_medications(patient_id: str) -> str:
    """Get a patient's medications (end=null means active) and allergies.

    Args:
        patient_id: ID from search_patients.
    """
    return _call("get_medications", {"patient_id": patient_id})


@beta_tool
def list_lab_tests(patient_id: str) -> str:
    """List the names of all lab tests a patient has results for.

    Use this to see what data exists, e.g. after get_lab_results returns nothing.

    Args:
        patient_id: ID from search_patients.
    """
    return _call("list_lab_tests", {"patient_id": patient_id})


def ask(question: str) -> str:
    runner = anthropic.Anthropic().beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        tools=[search_patients, get_lab_results, get_medications, list_lab_tests],
        messages=[{"role": "user", "content": question}],
        output_config={"effort": "medium"},
        max_iterations=MAX_TURNS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    final = runner.until_done()
    if final.stop_reason == "refusal":
        return "[declined by safety classifier]"
    return "".join(b.text for b in final.content if b.type == "text")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which Maria is allergic to latex?"
    print(f"Q: {question}\n\nA: {ask(question)}")
