"""Aggregator — joins the three parallel kit branches. Deliberately LLM-free.

The specialists already produced final markdown; composing them is
deterministic string work, so spending tokens here would add latency and a
failure mode for zero quality gain. Writes the kit file, records artifacts,
and emits the single final chat message for the whole kit flow.
"""
import re
from datetime import date

from langchain_core.messages import AIMessage

from src import db
from src.config import OUTPUTS_DIR


def _safe_name(*parts: str) -> str:
    stem = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", p).strip("_") for p in parts if p)
    return (stem or "application")[:80]


def aggregator_node(state: dict) -> dict:
    job = state["job"] or {}
    app_id = state.get("application_id")
    company, role = job.get("company", "Unknown"), job.get("role", "Role")

    header = [f"# Application Kit — {role} @ {company}", f"*Generated {date.today().isoformat()}*", ""]
    facts = [
        f"**Location:** {job['location']}" if job.get("location") else None,
        f"**Pay:** {job['salary']}" if job.get("salary") else None,
        f"**Deadline:** {job['deadline']}" if job.get("deadline") else None,
        f"**Summary:** {job['summary']}" if job.get("summary") else None,
    ]
    header += [f for f in facts if f] + [""]

    kit_md = "\n".join(header) + "\n\n---\n\n".join(
        [state["resume_report"], state["outreach_drafts"], state["interview_notes"]]
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{_safe_name(company, role)}_kit.md"
    path.write_text(kit_md, encoding="utf-8")

    db.add_artifact(app_id, "kit", str(path), kit_md)
    db.add_artifact(app_id, "resume_report", "", state["resume_report"])
    db.add_artifact(app_id, "outreach", "", state["outreach_drafts"])
    db.add_artifact(app_id, "interview_notes", "", state["interview_notes"])

    final = kit_md + f"\n\n---\n*Kit saved to `{path.name}` — also available in the Kits tab.*"
    return {"kit_path": str(path), "messages": [AIMessage(content=final)]}
