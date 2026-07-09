"""Resume Analyst agent — resume-vs-JD fit, gaps, and concrete rewrites.

Two-layer design: ATS keyword coverage is computed deterministically in Python
(reproducible, zero tokens) and handed to the LLM, which adds the qualitative
gap analysis and bullet rewrites on top. Resume evidence comes from RAG over
the resume collection, not the whole document — compact prompts on purpose.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import rag
from src.ats import coverage_markdown, keyword_coverage
from src.config import llm_main, structured_call
from src.schemas import ResumeFitReport

SYSTEM = """You are a sharp, honest resume coach for a student applying to internships/new-grad roles.
You are given: the job's details, the deterministic ATS keyword coverage (already computed — treat it as ground truth), and the most relevant excerpts of the candidate's resume.
Produce a fit report:
- fit_score consistent with the keyword coverage and evidence (do not flatter).
- strengths/gaps grounded ONLY in the provided excerpts — never invent experience.
- 3-5 bullet_rewrites: rewrite real bullets from the excerpts (quote them as `original`) to be quantified and loaded with this JD's keywords; use original="NEW" for at most one suggested addition the candidate could honestly write after a weekend project.
- verdict: 2-3 sentences — apply as-is, tweak first, or stretch."""


def _job_block(job: dict) -> str:
    lines = [f"Role: {job.get('role')} at {job.get('company')}"]
    if job.get("summary"):
        lines.append(f"Summary: {job['summary']}")
    if job.get("must_have_skills"):
        lines.append("Must-have: " + ", ".join(job["must_have_skills"]))
    if job.get("nice_to_have_skills"):
        lines.append("Nice-to-have: " + ", ".join(job["nice_to_have_skills"]))
    if job.get("qualifications"):
        lines.append("Qualifications: " + "; ".join(job["qualifications"][:4]))
    return "\n".join(lines)


def render_report(job: dict, report: ResumeFitReport, cov: dict) -> str:
    md = [
        f"## Resume fit — {job.get('role')} @ {job.get('company')}",
        f"**Fit score: {report.fit_score}/100** — {report.verdict}",
        "",
        coverage_markdown(cov),
        "",
        "**Strengths**",
        *[f"- {s}" for s in report.strengths],
        "",
        "**Gaps**",
        *[f"- {g}" for g in report.gaps],
        "",
        "**Suggested rewrites**",
    ]
    for r in report.bullet_rewrites:
        md += [f"- *Original:* {r.original}", f"  *Improved:* **{r.improved}**", f"  *Why:* {r.why}"]
    return "\n".join(md)


def resume_analyst_node(state: dict) -> dict:
    job = state["job"] or {}
    keywords = job.get("ats_keywords") or job.get("must_have_skills") or []
    cov = keyword_coverage(rag.resume_full_text(), keywords)

    query = ", ".join((job.get("must_have_skills") or []) + keywords[:8]) or f"{job.get('role', 'internship')} skills projects"
    chunks = rag.search_resume(query, k=5)

    coverage_note = (
        f"ATS coverage: {cov['coverage_pct']}% | present: {', '.join(cov['present']) or '—'} | missing: {', '.join(cov['missing']) or '—'}"
        if keywords
        else "ATS coverage: no parsed JD keywords available — score on evidence alone."
    )
    user = "\n\n".join(
        [
            "JOB\n" + _job_block(job),
            "DETERMINISTIC " + coverage_note,
            "RESUME EXCERPTS\n" + "\n".join(f"- {c}" for c in chunks),
        ]
    )
    report = structured_call(
        llm_main(temperature=0), ResumeFitReport, [SystemMessage(content=SYSTEM), HumanMessage(content=user)]
    )

    md = render_report(job, report, cov)
    out = {"resume_report": md}
    if state.get("mode") == "single":
        out["messages"] = [AIMessage(content=md)]
    return out
