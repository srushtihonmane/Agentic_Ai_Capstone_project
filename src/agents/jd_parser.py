"""JD Parser agent — turns messy posting text from any source into a JobPosting.

Handles the three real-world formats students actually paste: formal careers
pages, emoji-laden LinkedIn posts, and recruiter emails. The parsed posting is
the shared contract every downstream specialist reads, so this agent runs on
the large model at temperature 0 with schema-validated output.
"""
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import db, rag
from src.config import llm_main, structured_call
from src.schemas import JobPosting

SYSTEM = """You extract structured job postings from messy text: careers pages, LinkedIn posts, recruiter emails.
Rules:
- Extract only what is stated. Use null for unknown fields — never invent salary, deadline, or location.
- deadline: return as YYYY-MM-DD when derivable. Today is {today}; resolve relative phrases ("this Friday", "end of July") against it.
- source: infer from the format — careers page -> "portal", LinkedIn post -> "linkedin", email -> "email"; if the user was referred, "referral".
- ats_keywords: 10-20 exact keywords/phrases an applicant tracking system would scan resumes for (skills, tools, frameworks, credentials).
- summary: 2-3 plain sentences on the role."""


def parse_jd(jd_text: str) -> JobPosting:
    return structured_call(
        llm_main(temperature=0),
        JobPosting,
        [
            SystemMessage(content=SYSTEM.format(today=date.today().isoformat())),
            HumanMessage(content=jd_text[:8000]),
        ],
    )


def jd_parser_node(state: dict) -> dict:
    posting = parse_jd(state["jd_text"])
    app_id, created = db.upsert_from_posting(posting.model_dump(), state["jd_text"])

    # Condensed doc into the jobs collection -> semantic search across tracked roles.
    skills = ", ".join(posting.must_have_skills + posting.nice_to_have_skills)
    condensed = f"{posting.role} at {posting.company}. {posting.summary} Skills: {skills}"
    rag.index_job(app_id, posting.company, posting.role, condensed[:1200])

    out = {"job": posting.model_dump(), "application_id": app_id}
    forwards = state.get("mode") == "kit" or state.get("route") in ("resume_analyst", "outreach", "interview_prep")
    if not forwards:
        # Terminal in this mode -> confirm to the user. In kit/forwarding mode the
        # parser stays silent and downstream agents speak.
        verb = "Tracked new application" if created else "Refreshed existing application"
        bits = [f"**{verb} #{app_id}: {posting.role} @ {posting.company}**"]
        if posting.location:
            bits.append(f"- Location: {posting.location}" + (f" ({posting.work_mode})" if posting.work_mode else ""))
        if posting.salary:
            bits.append(f"- Pay: {posting.salary}")
        if posting.deadline:
            bits.append(f"- Deadline: {posting.deadline}")
        bits.append(f"- Must-have skills: {', '.join(posting.must_have_skills) or '—'}")
        bits.append(f"- {len(posting.ats_keywords)} ATS keywords extracted")
        bits.append("\nAsk for a *resume fit check*, *outreach drafts*, *interview prep* — or the *full kit*.")
        out["messages"] = [AIMessage(content="\n".join(bits))]
    return out
