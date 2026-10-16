"""A tool-calling agent written as an explicit loop.

The whole idea of an agent fits in one loop:
  1. send the conversation + tool definitions to Claude
  2. if Claude asks for tool calls, run them and append the results
  3. repeat until Claude answers without asking for a tool

Usage: uv run agent.py "What is Maria Alvarez's HbA1c trend?"
"""

import sys

import anthropic

from tools import TOOLS, run_tool

MODEL = "claude-opus-5-5"
MAX_TURNS = 8  # hard stop so a confused agent can't loop forever

SYSTEM = """You are a clinical data assistant working on a synthetic demo EHR.
- Answer only from tool results. Never guess values that the tools did not return.
- If a name matches more than one patient, list the matches and ask which one is meant.
- For every number you report, include its date.
- If the data needed is missing, say what is missing."""

client = anthropic.Anthropic()


def ask(question: str, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": question}]

    for turn in range(1, MAX_TURNS + 1):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
            # If a safety classifier declines the request, the API retries it on a
            # recommended fallback model inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        # Always append the full content (not just the text): tool_use blocks and
        # thinking blocks must be sent back unchanged on the next request.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            return "[declined by safety classifier]"
        if response.stop_reason == "max_tokens":
            return "[response cut off at max_tokens]"
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # Claude may ask for several tools in one turn. All results go back
        # together in a single user message.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result, is_error = run_tool(block.name, block.input)
            if verbose:
                flag = " (error)" if is_error else ""
                print(
                    f"  [turn {turn}] {block.name}({block.input}) -> {result[:120]}{flag}"
                )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return f"[stopped after {MAX_TURNS} turns without a final answer]"


if __name__ == "__main__":
    question = (
        " ".join(sys.argv[1:])
        or "What is Maria Alvarez's HbA1c trend, and is she on any diabetes medication?"
    )
    print(f"Q: {question}\n")
    print(f"\nA: {ask(question)}")
