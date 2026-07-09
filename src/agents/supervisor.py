"""Supervisor agent — the router at the top of the graph.

Runs on the small fast model (routing is cheap classification; Groq rate
limits are per-model, so the router never queues behind specialist calls).
The LLM only *classifies*; resolving "arcadia" to a tracked application row
is deterministic Python against SQLite — no tokens spent on lookups.
"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src import db
from src.config import llm_fast, structured_call
from src.schemas import RouteDecision

SYSTEM = """You are the dispatcher of a job-application command centre for a student.
Classify the LATEST user message into exactly one route:
- jd_parser: message contains a pasted job description and the user just wants it tracked/saved.
- full_kit: user wants complete preparation for one job (a "kit": resume fit + messages + interview prep), with or without a pasted JD.
- resume_analyst: resume fit, gaps, ATS keywords, or resume improvements for a job.
- outreach: recruiter DM, referral request, cold email, follow-up, or thank-you message.
- interview_prep: interview questions or preparation notes for a job.
- tracker: add/update/list applications, change status, deadlines, notes, or show stats — INCLUDING progress reported in the user's own words (they applied somewhere, cleared an online assessment, got an interview, an offer, or a rejection).
- assistant: greetings, help, capability questions, anything else.

Also report:
- job_hint: the company or role words the user referred to (null if none).
- has_jd_text: true ONLY if the message body contains an actual pasted job posting (multi-line text with requirements/responsibilities), not a mere mention of a job.

Examples:
"track this: <long posting text>" -> jd_parser, has_jd_text=true
"prepare the full kit for arcadia" -> full_kit, job_hint="arcadia"
"track this and prepare the full application kit: <posting>" -> full_kit, has_jd_text=true
"here's the JD, get me fully ready for it: <posting>" -> full_kit, has_jd_text=true
"how well does my resume fit the nimbuspay internship?" -> resume_analyst, job_hint="nimbuspay"
"draft a referral message for the zenith role" -> outreach, job_hint="zenith"
"what should I prepare for the meridian interview?" -> interview_prep, job_hint="meridian"
"mark arcadia as applied yesterday" -> tracker, job_hint="arcadia"
"I cleared the quantedge OA, they moved me to interviews" -> tracker, job_hint="quantedge"
"got rejected from cloudsprint today :(" -> tracker, job_hint="cloudsprint"
"what's due this week?" -> tracker
"hi, what can you do?" -> assistant"""


def decide_route(text: str, prev_reply: str = "") -> RouteDecision:
    user = text if not prev_reply else f"(previous assistant reply, for context: {prev_reply[:200]})\n\n{text}"
    return structured_call(
        llm_fast(), RouteDecision, [SystemMessage(content=SYSTEM), HumanMessage(content=user)]
    )


def _resolve_job(hint: str) -> tuple[dict | None, int | None]:
    """Deterministic lookup: hint -> newest matching application row (+ parsed JD if stored)."""
    matches = db.resolve_applications(hint)
    if not matches:
        return None, None
    row = matches[0]
    job = json.loads(row["jd_json"]) if row.get("jd_json") else {}
    job.setdefault("company", row["company"])
    job.setdefault("role", row["role"])
    job["db_status"] = row["status"]
    job["db_deadline"] = row.get("deadline")
    return job, row["id"]


def supervisor_node(state: dict) -> dict:
    latest = state["messages"][-1].content
    prev_ai = next(
        (m.content for m in reversed(state["messages"][:-1]) if m.type == "ai" and isinstance(m.content, str)),
        "",
    )
    decision = decide_route(latest, prev_ai)

    # Deterministic guard: "track this AND prep the kit" must never lose its
    # kit half to a routing miss — the flagship flow deserves a belt on top
    # of the few-shot suspenders.
    if decision.route == "jd_parser" and re.search(
        r"\b(full kit|application kit|the kit|fully ready|prepare everything|full prep)\b", latest, re.IGNORECASE
    ):
        decision.route = "full_kit"

    job, app_id = (None, None)
    if decision.job_hint and not decision.has_jd_text:
        job, app_id = _resolve_job(decision.job_hint)

    # Ephemeral keys are reset EVERY turn: the checkpointer persists state
    # across turns, and a previous kit's outputs must never leak into this one.
    return {
        "route": decision.route,
        "mode": "kit" if decision.route == "full_kit" else "single",
        "jd_text": latest if decision.has_jd_text else "",
        "job": job,
        "application_id": app_id,
        "resume_report": "",
        "outreach_drafts": "",
        "interview_notes": "",
        "kit_path": "",
    }
