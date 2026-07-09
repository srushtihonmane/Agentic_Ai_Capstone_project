"""Assistant agent — help, chit-chat, and the graph's graceful-failure path.

When the supervisor routes a job-specific request but no tracked application
matches the hint (or a 'track this' arrives with no pasted JD), the graph
lands here and the assistant asks a useful disambiguation question instead of
guessing. Runs on the fast model.
"""
from langchain_core.messages import AIMessage, SystemMessage

from src import db
from src.config import llm_fast, trim_history

SYSTEM = """You are the front desk of an AI Job Application Command Centre for a student. Be brief and warm.
The centre can: track applications (statuses: saved/applied/oa/interview/offer/rejected, deadlines, stats), parse pasted job descriptions from any source, analyse resume fit with ATS keyword coverage, draft recruiter outreach (DMs, referral asks, cold emails), prepare interview notes, and build a full "application kit" (all of the above at once, agents working in parallel).
If the user seems unsure, suggest one concrete next step, e.g. paste a JD and say "track this", or ask "what's due this week?"."""


def _tracked_list() -> str:
    apps = db.list_applications()
    if not apps:
        return "(nothing tracked yet)"
    return "\n".join(f"- #{a['id']} **{a['company']}** — {a['role']} `[{a['status']}]`" for a in apps[:15])


def assistant_node(state: dict) -> dict:
    route = state.get("route", "assistant")

    if route != "assistant":
        # Disambiguation path: a specialist was requested but no job could be resolved.
        hint = (
            "Paste the job description text right into the chat"
            if route == "jd_parser"
            else "Name the company from the list (or paste the JD)"
        )
        msg = (
            "I couldn't tell which job you meant. Here's what's tracked right now:\n\n"
            f"{_tracked_list()}\n\n{hint} and I'll get to work."
        )
        return {"messages": [AIMessage(content=msg)]}

    reply = llm_fast(temperature=0.4).invoke(
        [SystemMessage(content=SYSTEM + "\n\nCurrently tracked:\n" + _tracked_list())]
        + trim_history(state["messages"])
    )
    return {"messages": [reply]}
