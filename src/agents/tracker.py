"""Tracker agent — conversational CRUD over the applications database.

Unlike the single-shot specialists, this agent runs a classic tool-calling
(ReAct) loop with LangGraph's ToolNode: the model may chain several tool calls
in one turn ("mark arcadia applied AND show my stats") before answering.

Groq-hosted llama occasionally emits a tool call as literal text
("<function(update_application){...}") instead of using the tool-calling API.
tracker_node handles that in two layers: deterministically repair the fake
text into a real tool call when it parses cleanly; otherwise retry once with
a corrective nudge.
"""
import json
import re
import uuid
from datetime import date

from langchain_core.messages import AIMessage, SystemMessage

from src.config import llm_main, trim_history
from src.tools import TRACKER_TOOLS

SYSTEM = """You manage a student's job-application tracker. Today is {today}.
Rules:
- Use tools for EVERY data operation — never answer about tracker data from memory. Call tools ONLY through the tool-calling interface; never write function-call syntax in your text reply.
- Resolve relative dates yourself ("yesterday", "next Friday") to YYYY-MM-DD before calling tools.
- When the user says they applied, set status='applied' and applied_date.
- If a tool reports ambiguity, ask the user which application they meant (quote the # ids).
- After acting, confirm concisely what changed. Render lists as a markdown table with columns: #, Company, Role, Status, Deadline. Mention today's date when discussing deadlines.
- statuses: saved -> applied -> oa (online assessment) -> interview -> offer / rejected."""


_FAKE_TOOL_TEXT = re.compile(r"<function|<tool_call|\"name\"\s*:\s*\"(add|update|get|list)_", re.IGNORECASE)
_TOOL_NAMES = {t.name for t in TRACKER_TOOLS}


def _repair_fake_tool_call(text: str) -> AIMessage | None:
    """Parse '<function(name){json}</function>'-style text into a real tool call."""
    m = re.search(r"<function[=(\s]+(\w+)\s*\)?>?\s*(\{.*\})\s*(?:</function>)?", text, re.DOTALL) or re.search(
        r"\{\s*\"name\"\s*:\s*\"(\w+)\"\s*,\s*\"(?:arguments|parameters|args)\"\s*:\s*(\{.*\})\s*\}", text, re.DOTALL
    )
    if not m or m.group(1) not in _TOOL_NAMES:
        return None
    raw = m.group(2)
    for candidate in (raw, raw[: raw.find("}") + 1]):  # greedy first, then first flat object
        try:
            args = json.loads(candidate)
            break
        except json.JSONDecodeError:
            args = None
    if not isinstance(args, dict):
        return None
    return AIMessage(
        content="",
        tool_calls=[{"name": m.group(1), "args": args, "id": f"repaired_{uuid.uuid4().hex[:8]}", "type": "tool_call"}],
    )


def tracker_node(state: dict) -> dict:
    msgs = [SystemMessage(content=SYSTEM.format(today=date.today().isoformat()))] + trim_history(
        state["messages"], keep=10
    )
    llm = llm_main(temperature=0).bind_tools(TRACKER_TOOLS)
    ai = llm.invoke(msgs)
    if not getattr(ai, "tool_calls", None) and isinstance(ai.content, str) and _FAKE_TOOL_TEXT.search(ai.content):
        repaired = _repair_fake_tool_call(ai.content)
        if repaired is None:
            ai = llm.invoke(msgs + [SystemMessage(
                content="Your previous attempt wrote a function call as plain text, which does nothing. "
                "Invoke the tool through the tool-calling interface now."
            )])
            if not getattr(ai, "tool_calls", None) and isinstance(ai.content, str):
                repaired = _repair_fake_tool_call(ai.content)
        if repaired is not None:
            ai = repaired
    return {"messages": [ai]}
