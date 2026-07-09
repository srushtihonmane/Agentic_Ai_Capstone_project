"""Tracker agent's tools — thin, validated wrappers over src/db.py.

Every tool returns a compact JSON string so the model can quote exact fields
back to the user. Fuzzy references ("arcadia", "#7") are resolved in Python;
ambiguity is returned to the agent as data, so *it* can ask the user.
"""
import json
from typing import Optional

from langchain_core.tools import tool

from src import db

_SLIM = ("id", "company", "role", "status", "deadline", "applied_date", "source", "location", "notes")


def _slim(row: dict) -> dict:
    return {k: row.get(k) for k in _SLIM if row.get(k) is not None or k in ("id", "company", "role", "status")}


def _resolve_one(query: str):
    matches = db.resolve_applications(query)
    if not matches:
        return None, json.dumps({"error": f"No tracked application matches '{query}'."})
    if len(matches) > 1 and not query.lstrip("#").isdigit():
        exact = [m for m in matches if query.lower() in (m["company"].lower(), m["role"].lower())]
        if len(exact) != 1:
            return None, json.dumps(
                {"ambiguous": [f"#{m['id']} {m['company']} — {m['role']}" for m in matches[:5]],
                 "hint": "Ask the user which one they meant (use the # id)."}
            )
        matches = exact
    return matches[0], None


@tool
def add_application(company: str, role: str, location: Optional[str] = None, source: Optional[str] = None,
                    status: str = "saved", deadline: Optional[str] = None, notes: Optional[str] = None) -> str:
    """Track a new job application. status: saved|applied|oa|interview|offer|rejected. deadline: YYYY-MM-DD. source: portal|linkedin|referral|email|other."""
    if status not in db.STATUSES:
        return json.dumps({"error": f"status must be one of {db.STATUSES}"})
    app_id = db.add_application(company, role, location=location, source=source,
                                status=status, deadline=deadline, notes=notes)
    return json.dumps({"created": _slim(db.get_application(app_id))})


@tool
def update_application(query: str, status: Optional[str] = None, deadline: Optional[str] = None,
                       notes: Optional[str] = None, applied_date: Optional[str] = None) -> str:
    """Update a tracked application found by company name, role, or '#id'. status: saved|applied|oa|interview|offer|rejected. Dates: YYYY-MM-DD. Only pass fields that should change."""
    if status is not None and status not in db.STATUSES:
        return json.dumps({"error": f"status must be one of {db.STATUSES}"})
    row, err = _resolve_one(query)
    if err:
        return err
    updated = db.update_application(row["id"], status=status, deadline=deadline,
                                    notes=notes, applied_date=applied_date)
    return json.dumps({"updated": _slim(updated)})


@tool
def get_application(query: str) -> str:
    """Full detail for one application found by company name, role, or '#id', including its generated artifacts."""
    row, err = _resolve_one(query)
    if err:
        return err
    arts = [{"kind": a["kind"], "created_at": a["created_at"]} for a in db.list_artifacts(row["id"])]
    detail = _slim(row)
    detail["has_parsed_jd"] = bool(row.get("jd_json"))
    detail["artifacts"] = arts
    return json.dumps(detail)


@tool
def list_applications(status: Optional[str] = None, due_within_days: Optional[int] = None) -> str:
    """List tracked applications, optionally filtered by status (saved|applied|oa|interview|offer|rejected) and/or deadline within N days from today."""
    rows = db.list_applications(status=status, due_within_days=due_within_days)
    return json.dumps({"count": len(rows), "applications": [_slim(r) for r in rows]})


@tool
def get_stats() -> str:
    """Pipeline stats: totals by status, applications added this week, and deadlines in the next 7 days."""
    return json.dumps(db.get_stats())


TRACKER_TOOLS = [add_application, update_application, get_application, list_applications, get_stats]
