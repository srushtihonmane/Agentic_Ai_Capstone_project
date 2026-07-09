"""Pydantic contracts shared by all agents.

Every LLM structured-output call in the system validates against one of these
models, so downstream code (DB, aggregator, UI) never touches free-form text.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Route = Literal[
    "jd_parser",       # user pasted a JD, just track it
    "full_kit",        # parse (if needed) + resume fit + outreach + interview prep, in parallel
    "resume_analyst",  # resume-vs-JD gap analysis only
    "outreach",        # recruiter messages only
    "interview_prep",  # interview notes only
    "tracker",         # CRUD / stats over the applications DB
    "assistant",       # help, chit-chat, disambiguation
]


class RouteDecision(BaseModel):
    """Supervisor's verdict on where a user message should go."""

    route: Route
    job_hint: Optional[str] = Field(
        default=None,
        description="Company or role name the user referred to, if any (e.g. 'arcadia', 'the nimbuspay internship').",
    )
    has_jd_text: bool = Field(
        default=False,
        description="True if the message contains a pasted job description (multi-line posting text), not just a mention of a job.",
    )


class JobPosting(BaseModel):
    """Structured extraction of a job description from any source (portal, LinkedIn post, recruiter email)."""

    company: str
    role: str
    location: Optional[str] = None
    work_mode: Optional[str] = Field(default=None, description="remote / hybrid / onsite if stated")
    seniority: Optional[str] = Field(default=None, description="intern / new-grad / junior / senior etc.")
    salary: Optional[str] = Field(default=None, description="pay or stipend as written, if stated")
    deadline: Optional[str] = Field(default=None, description="application deadline as YYYY-MM-DD if derivable")
    source: Optional[str] = Field(default=None, description="portal / linkedin / referral / email / other")
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)
    ats_keywords: list[str] = Field(
        default_factory=list,
        description="10-20 exact keywords/phrases an ATS would scan resumes for (skills, tools, credentials).",
    )
    summary: str = Field(default="", description="2-3 sentence summary of the role")


class BulletRewrite(BaseModel):
    """One concrete resume edit suggestion."""

    original: str = Field(description="The existing resume bullet, or 'NEW' if this is a suggested addition.")
    improved: str = Field(description="The rewritten bullet: quantified, keyword-loaded, one line.")
    why: str = Field(description="One line on why this rewrite helps for this specific JD.")


class ResumeFitReport(BaseModel):
    """Resume Analyst's verdict for one resume against one JD."""

    fit_score: int = Field(ge=0, le=100, description="0-100, consistent with the ATS keyword coverage provided.")
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    bullet_rewrites: list[BulletRewrite] = Field(default_factory=list, description="3-5 concrete rewrites.")
    verdict: str = Field(default="", description="2-3 sentence bottom line: apply as-is, tweak first, or stretch.")
