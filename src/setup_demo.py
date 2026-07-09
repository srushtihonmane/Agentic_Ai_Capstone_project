"""One-command demo bootstrap: python -m src.setup_demo

- Initializes the SQLite DB and seeds 4 demo applications (deadlines relative
  to today) so the board and KPIs look alive immediately. Idempotent.
- Ingests the resume into Chroma LAST — this forces the one-time ~80 MB
  embedding-model download here rather than mid-demo, and a network failure
  still leaves you with a fully seeded, usable tracker.
"""
import sys
from datetime import date, timedelta

from src import db


def _seed(company, role, *, status, source, days_to_deadline=None, applied_days_ago=None, notes=None, location=None):
    if db.resolve_applications(company):
        return f"  = {company} — already seeded, skipped"
    deadline = (date.today() + timedelta(days=days_to_deadline)).isoformat() if days_to_deadline is not None else None
    app_id = db.add_application(company, role, location=location, source=source, status=status,
                                deadline=deadline, notes=notes)
    if applied_days_ago is not None:
        db.update_application(app_id, applied_date=(date.today() - timedelta(days=applied_days_ago)).isoformat())
    return f"  + #{app_id} {company} — {role} [{status}]"


def seed_applications() -> None:
    db.init_db()
    print("Seeding demo applications ...")
    print(_seed("TechNova Solutions", "Backend Developer Intern", status="applied", source="portal",
                days_to_deadline=5, applied_days_ago=3, location="Pune (Remote)",
                notes="Referred by a senior; follow up if no reply by next week."))
    print(_seed("QuantEdge Capital", "Quantitative Developer Intern", status="oa", source="referral",
                days_to_deadline=8, location="Mumbai",
                notes="Online assessment link in email — 90 min, DSA + probability."))
    print(_seed("Meridian Analytics", "Data Science Intern", status="interview", source="linkedin",
                location="Bengaluru (Hybrid)",
                notes="Round 2 panel on Friday: DS lead + senior analyst."))
    print(_seed("CloudSprint", "Frontend Developer Intern", status="rejected", source="portal",
                location="Remote",
                notes="Rejected after OA. Recruiter said reapply next cycle."))


def ingest_resume() -> bool:
    from src import rag  # imported lazily: chroma is only touched here

    print("Ingesting resume into Chroma (first run downloads the local embedding model, ~80 MB) ...")
    try:
        n = rag.ingest_resume()
        print(f"  resume indexed: {n} chunks")
        return True
    except Exception as exc:
        print(f"  WARNING: resume ingestion failed ({type(exc).__name__}: {exc}).")
        print("  Usually a flaky download of the embedding model — just re-run: python -m src.setup_demo")
        return False


def main() -> None:
    print("Initializing database ...")
    seed_applications()
    ok = ingest_resume()
    print("\nReady. Launch the command centre with:\n  streamlit run app.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
