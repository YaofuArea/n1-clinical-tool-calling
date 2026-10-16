"""Tools the agent can call, plus the JSON schemas Claude sees for them.

Each tool is a plain Python function over a synthetic EHR file. Claude never
runs these functions itself: it asks for a call, our loop runs the function,
and we send the result back.
"""

import json
from pathlib import Path

DATA = json.loads((Path(__file__).parent / "data" / "synthetic_ehr.json").read_text())
PATIENTS = {p["id"]: p for p in DATA["patients"]}


class ToolError(Exception):
    """A failure we want Claude to see and recover from (sent back with is_error)."""


def search_patients(name: str) -> list[dict]:
    query = name.strip().lower()
    if not query:
        raise ToolError("name must not be empty")
    return [
        {"id": p["id"], "name": p["name"], "birth_date": p["birth_date"]}
        for p in DATA["patients"]
        if query in p["name"].lower()
    ]


def _get_patient(patient_id: str) -> dict:
    if patient_id not in PATIENTS:
        raise ToolError(f"unknown patient_id {patient_id!r}; call search_patients first")
    return PATIENTS[patient_id]


def get_lab_results(patient_id: str, test_name: str) -> list[dict]:
    labs = _get_patient(patient_id)["labs"]
    matches = [lab for lab in labs if lab["test"].lower() == test_name.strip().lower()]
    return sorted(matches, key=lambda lab: lab["date"])


def get_medications(patient_id: str) -> dict:
    patient = _get_patient(patient_id)
    return {"medications": patient["medications"], "allergies": patient["allergies"]}

def list_lab_tests(patient_id: str) -> list[str]:
    labs = _get_patient(patient_id)["labs"]
    return sorted({lab["test"] for lab in labs})


# What Claude sees. The description is the main thing that decides *when* Claude
# picks a tool, so it says what the tool returns and when to use it.
# strict=True makes the API guarantee the input matches the schema.
TOOLS = [
    {
        "name": "search_patients",
        "description": (
            "Find patients whose name contains the given text (case-insensitive). "
            "Returns a list of {id, name, birth_date}. Use this first to get a patient_id. "
            "More than one patient can match."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Full or partial patient name"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_lab_results",
        "description": (
            "Get all results of one lab test for a patient, oldest first. "
            "Each result has test, value, unit, date. Known tests include HbA1c, LDL, TSH. "
            "Returns an empty list if the patient has no results for that test."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "ID from search_patients, e.g. P001"},
                "test_name": {"type": "string", "description": "Lab test name, e.g. HbA1c"},
            },
            "required": ["patient_id", "test_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_medications",
        "description": (
            "Get a patient's medication list (name, dose, start, end; end=null means still active) "
            "and their allergies."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "ID from search_patients"}},
            "required": ["patient_id"],
            "additionalProperties": False,
        },
    },
{
        "name": "list_lab_tests",
        "description": (
            "List the names of all lab tests a patient has results for, e.g. [\"HbA1c\", \"LDL\"]. "
            "Use this to see what data exists before calling get_lab_results, "
            "or to tell the user which tests are available."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string", "description": "ID from search_patients"}},
            "required": ["patient_id"],
            "additionalProperties": False,
        },
    },]

FUNCTIONS = {
    "search_patients": search_patients,
    "get_lab_results": get_lab_results,
    "get_medications": get_medications,
    "list_lab_tests": list_lab_tests,
}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run one tool call. Returns (result_text, is_error)."""
    func = FUNCTIONS.get(name)
    if func is None:
        return f"unknown tool {name!r}", True
    try:
        return json.dumps(func(**tool_input)), False
    except ToolError as e:
        return str(e), True
