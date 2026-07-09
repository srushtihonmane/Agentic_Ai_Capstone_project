"""Interview Prep agent — role-specific question bank + STAR prompts.

The STAR prompts are explicitly mapped to retrieved resume bullets ("use your
CampusQA project to answer this"), which is what makes the notes actually
usable in the 30 minutes before a call.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import rag
from src.config import llm_main

SYSTEM = """You prepare a student for a specific job interview. Ground everything in the provided job details and resume excerpts — never invent experience.
Produce these markdown sections:
### Likely technical questions
(6 questions derived from the must-have skills, each with a 1-line "what they're checking" pointer)
### Behavioral questions
(4 questions this kind of team actually asks)
### Your STAR stories
(3 prompts, each explicitly mapped to a specific project/experience from the resume excerpts: "Use <project> to answer ...", with 2 bullet hints on what to emphasize)
### Questions to ask them
(4 sharp questions that show research into the role)
### Logistics
(one line: deadline/process details if known, else "No process details captured — check the posting.")"""


def interview_prep_node(state: dict) -> dict:
    job = state["job"] or {}
    skills = ", ".join((job.get("must_have_skills") or []) + (job.get("nice_to_have_skills") or [])[:3])
    chunks = rag.search_resume((skills or job.get("role", "")) + " projects experience achievements", k=4)

    details = [f"JOB: {job.get('role')} at {job.get('company')}"]
    if job.get("summary"):
        details.append(f"Summary: {job['summary']}")
    if skills:
        details.append(f"Skills they want: {skills}")
    if job.get("responsibilities"):
        details.append("Responsibilities: " + "; ".join(job["responsibilities"][:5]))
    if job.get("deadline") or job.get("db_deadline"):
        details.append(f"Deadline: {job.get('deadline') or job.get('db_deadline')}")

    user = "\n".join(details) + "\n\nCANDIDATE RESUME EXCERPTS\n" + "\n".join(f"- {c}" for c in chunks)
    notes = llm_main(temperature=0.3).invoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)]).content

    md = f"## Interview prep — {job.get('role')} @ {job.get('company')}\n\n{notes}"
    out = {"interview_notes": md}
    if state.get("mode") == "single":
        out["messages"] = [AIMessage(content=md)]
    return out
