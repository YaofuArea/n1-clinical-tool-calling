# N1 · Clinical Tool-Calling Agent

A minimal LLM agent that answers clinician-style questions ("What is Maria Alvarez's HbA1c trend?") by calling tools over a small **synthetic** EHR, instead of guessing.

This is the first of a series of small experiments toward a clinical chart assistant. The goal here is only to understand **how tool calling and the agent loop work**.

> All patient data in `data/synthetic_ehr.json` is invented. No real PHI.

## How it works

```
question ─▶ Claude (+ tool definitions)
               │ stop_reason == "tool_use"?
               ├─ yes ─▶ run the requested tools locally ─▶ append tool_result ─┐
               │                                                               │
               └─ no ──▶ final answer                    ◀──── loop again ◀────┘
```

| File | What it is |
|---|---|
| `tools.py` | 3 tools (`search_patients`, `get_lab_results`, `get_medications`), their JSON schemas, and `run_tool()` |
| `agent.py` | The agent loop written by hand: every request, tool call, and tool result is explicit |
| `agent_runner.py` | The same agent using the SDK's tool runner, which owns the loop |
| `data/synthetic_ehr.json` | 3 synthetic patients (two of them are named Maria, on purpose) |

## Run

```bash
uv sync
export ANTHROPIC_API_KEY=...        # your key
uv run agent.py "What is Maria Alvarez's HbA1c trend, and is she on any diabetes medication?"
uv run agent.py "What medications is Maria on?"          # ambiguous: two Marias
uv run agent.py "What is James Okafor's TSH?"            # data that does not exist
uv run agent_runner.py "Which Maria is allergic to latex?"
```

## Design decisions

- **Hand-written loop first.** `agent.py` writes the loop explicitly so every step is visible; `agent_runner.py` shows what the SDK abstracts away.
- **Tool errors go back to the model.** An unknown patient ID returns a `tool_result` with `is_error: true` and a hint ("call search_patients first"), so the agent can recover instead of crashing.
- **Strict schemas.** `strict: true` makes the API guarantee tool inputs match the schema.
- **Turn limit.** `MAX_TURNS = 8` stops a confused agent from looping forever.
- **Grounding rules in the system prompt.** Only answer from tool results, date every number, ask when a name is ambiguous.

## Experiments

<!-- TODO(me): run each of these and write down what actually happened. -->

| # | Question | What I expected | What happened |
|---|---|---|---|
| 1 | Maria Alvarez's HbA1c trend | 3 values, falling 8.4 → 7.1 | 3 turns: searched for the patient first, then fetched labs and medications in parallel. Answer matched (8.4 → 7.6 → 7.1, each with a date), and it noted metformin started after the first result without claiming it caused the drop.|
| 2 | "What medications is Maria on?" (two Marias) | asks which Maria | 2 turns: search returned both Marias, then it stopped and asked which one, listing birth dates, without fetching anyone's medications. Limitation: agent.py is single-turn, so I can't answer its question yet.|
| 3 | James Okafor's TSH (no data) | says no TSH results | 3 turns: found P002, TSH search returned an empty list, and it said there are no TSH values without inventing one. But it offered to check HbA1c, which James doesn't have; it copied the example tests from the tool description. A list_lab_tests tool would fix this.|
| 4 | | | |

## What I learned / pitfalls

<!-- TODO(me): in my own words. Examples of things worth noting:
- how many turns did a question take, and why
- did Claude call tools in parallel?
- what changed when I rewrote a tool description?
- what's different between agent.py and agent_runner.py?
-->
