"""Outreach agent — recruiter-facing messages personalized from JD + resume.

Free-form markdown (no schema): message drafting benefits from a little
temperature, and the output is copied by a human, not parsed by code.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import rag
from src.config import llm_main

SYSTEM = """You draft job-search outreach for a student. Personalize using ONLY the resume excerpts and job details provided — never invent experience, names, or mutual connections.
Tone: confident, specific, zero clichés ("I hope this finds you well", "esteemed organization" are banned). Every message must reference one concrete, relevant thing the candidate has built.
Produce exactly these sections in markdown:
### LinkedIn DM to recruiter
(under 90 words, ends with a clear ask)
### Referral request to an employee
(under 120 words, makes it easy to say yes — mention you'll send resume + JD link)
### Cold email
(subject line + under 150 words body)
### Follow-up nudge
(2 sentences, for one week later)"""


def outreach_node(state: dict) -> dict:
    job = state["job"] or {}
    skills = ", ".join((job.get("must_have_skills") or [])[:8])
    chunks = rag.search_resume(skills or f"{job.get('role', '')} projects experience", k=3)

    user = "\n\n".join(
        [
            f"JOB: {job.get('role')} at {job.get('company')}"
            + (f" ({job.get('location')})" if job.get("location") else ""),
            f"Job summary: {job.get('summary', 'n/a')}",
            f"Key skills they want: {skills or 'n/a'}",
            "CANDIDATE RESUME EXCERPTS\n" + "\n".join(f"- {c}" for c in chunks),
        ]
    )
    draft = llm_main(temperature=0.4).invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)]).content

    md = f"## Outreach drafts — {job.get('role')} @ {job.get('company')}\n\n{draft}"
    out = {"outreach_drafts": md}
    if state.get("mode") == "single":
        out["messages"] = [AIMessage(content=md)]
    return out
