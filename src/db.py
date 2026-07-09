"""SQLite system of record for applications, artifacts, and status history.

Every function opens a short-lived connection, so the Streamlit UI thread and
the LangGraph worker threads never share a connection object (sidesteps
sqlite's check_same_thread entirely). The schema script is idempotent.
"""
import json
import sqlite3
from typing import Optional

from src.config import DB_PATH, STORAGE_DIR

STATUSES = ["saved", "applied", "oa", "interview", "offer", "rejected"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  company TEXT NOT NULL,
  role TEXT NOT NULL,
  location TEXT,
  source TEXT,
  status TEXT NOT NULL DEFAULT 'saved',
  deadline TEXT,
  applied_date TEXT,
  salary TEXT,
  jd_raw TEXT,
  jd_json TEXT,
  notes TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  application_id INTEGER REFERENCES applications(id),
  kind TEXT,
  path TEXT,
  content TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS status_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  application_id INTEGER,
  status TEXT,
  changed_at TEXT DEFAULT (datetime('now'))
);
"""


def _conn() -> sqlite3.Connection:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _rows(rows) -> list[dict]:
    return [dict(r) for r in rows]


def init_db() -> None:
    _conn().close()


# ---------------------------------------------------------------- applications

def add_application(
    company: str,
    role: str,
    location: Optional[str] = None,
    source: Optional[str] = None,
    status: str = "saved",
    deadline: Optional[str] = None,
    salary: Optional[str] = None,
    notes: Optional[str] = None,
    jd_raw: Optional[str] = None,
    jd_json: Optional[str] = None,
) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO applications (company, role, location, source, status, deadline, salary, notes, jd_raw, jd_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (company, role, location, source, status, deadline, salary, notes, jd_raw, jd_json),
        )
        conn.execute("INSERT INTO status_history (application_id, status) VALUES (?,?)", (cur.lastrowid, status))
        return cur.lastrowid


def upsert_from_posting(posting: dict, jd_raw: str) -> tuple[int, bool]:
    """Insert a parsed JobPosting, or refresh the existing row for the same company+role.

    Returns (application_id, created).
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT id FROM applications WHERE lower(company)=lower(?) AND lower(role)=lower(?)",
            (posting["company"], posting["role"]),
        ).fetchone()
        payload = json.dumps(posting, ensure_ascii=False)
        if row:
            conn.execute(
                "UPDATE applications SET location=COALESCE(?,location), source=COALESCE(?,source),"
                " deadline=COALESCE(?,deadline), salary=COALESCE(?,salary), jd_raw=?, jd_json=?,"
                " updated_at=datetime('now') WHERE id=?",
                (posting.get("location"), posting.get("source"), posting.get("deadline"),
                 posting.get("salary"), jd_raw, payload, row["id"]),
            )
            return row["id"], False
        cur = conn.execute(
            "INSERT INTO applications (company, role, location, source, status, deadline, salary, jd_raw, jd_json)"
            " VALUES (?,?,?,?,'saved',?,?,?,?)",
            (posting["company"], posting["role"], posting.get("location"), posting.get("source"),
             posting.get("deadline"), posting.get("salary"), jd_raw, payload),
        )
        conn.execute("INSERT INTO status_history (application_id, status) VALUES (?,'saved')", (cur.lastrowid,))
        return cur.lastrowid, True


def resolve_applications(query: str) -> list[dict]:
    """Fuzzy lookup by id ('7', '#7') or substring of company/role. Newest first."""
    q = (query or "").strip()
    with _conn() as conn:
        if q.lstrip("#").isdigit():
            rows = conn.execute("SELECT * FROM applications WHERE id=?", (int(q.lstrip("#")),)).fetchall()
            if rows:
                return _rows(rows)
        like = f"%{q}%"
        return _rows(
            conn.execute(
                "SELECT * FROM applications WHERE company LIKE ? OR role LIKE ? ORDER BY id DESC",
                (like, like),
            ).fetchall()
        )


def get_application(app_id: int) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM applications WHERE id=?", (app_id,)).fetchone()
        return dict(row) if row else None


def update_application(app_id: int, **fields) -> Optional[dict]:
    """Update allowed columns; status changes are appended to status_history."""
    allowed = {"status", "deadline", "notes", "applied_date", "location", "source", "salary"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_application(app_id)
    with _conn() as conn:
        current = conn.execute("SELECT status FROM applications WHERE id=?", (app_id,)).fetchone()
        if current is None:
            return None
        sets = ", ".join(f"{k}=?" for k in updates)
        conn.execute(
            f"UPDATE applications SET {sets}, updated_at=datetime('now') WHERE id=?",
            (*updates.values(), app_id),
        )
        if "status" in updates and updates["status"] != current["status"]:
            conn.execute("INSERT INTO status_history (application_id, status) VALUES (?,?)", (app_id, updates["status"]))
    return get_application(app_id)


def delete_application(app_id: int) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM applications WHERE id=?", (app_id,))
        conn.execute("DELETE FROM artifacts WHERE application_id=?", (app_id,))
        conn.execute("DELETE FROM status_history WHERE application_id=?", (app_id,))
        return cur.rowcount > 0


def list_applications(status: Optional[str] = None, due_within_days: Optional[int] = None) -> list[dict]:
    sql, params = "SELECT * FROM applications", []
    where = []
    if status:
        where.append("status=?")
        params.append(status)
    if due_within_days is not None:
        where.append("deadline IS NOT NULL AND deadline >= date('now') AND deadline <= date('now', ?)")
        params.append(f"+{int(due_within_days)} day")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY CASE WHEN deadline IS NULL THEN 1 ELSE 0 END, deadline, id DESC"
    with _conn() as conn:
        return _rows(conn.execute(sql, params).fetchall())


def get_stats() -> dict:
    with _conn() as conn:
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute("SELECT status, COUNT(*) n FROM applications GROUP BY status").fetchall()
        }
        total = sum(by_status.values())
        due_soon = _rows(
            conn.execute(
                "SELECT id, company, role, deadline, status FROM applications"
                " WHERE deadline IS NOT NULL AND deadline >= date('now') AND deadline <= date('now', '+7 day')"
                " AND status NOT IN ('offer','rejected') ORDER BY deadline"
            ).fetchall()
        )
        added_this_week = conn.execute(
            "SELECT COUNT(*) n FROM applications WHERE created_at >= datetime('now', '-7 day')"
        ).fetchone()["n"]
    return {"total": total, "by_status": by_status, "due_soon": due_soon, "added_this_week": added_this_week}


# ------------------------------------------------------------------- artifacts

def add_artifact(application_id: Optional[int], kind: str, path: str, content: str) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO artifacts (application_id, kind, path, content) VALUES (?,?,?,?)",
            (application_id, kind, path, content),
        )
        return cur.lastrowid


def list_artifacts(application_id: Optional[int] = None) -> list[dict]:
    with _conn() as conn:
        if application_id is not None:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE application_id=? ORDER BY id DESC", (application_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM artifacts ORDER BY id DESC").fetchall()
        return _rows(rows)


if __name__ == "__main__":  # self-check: python -m src.db
    init_db()
    app_id = add_application("SelfCheck Co", "QA Intern", deadline="2099-01-01", source="portal")
    update_application(app_id, status="applied", applied_date="2099-01-01")
    assert get_application(app_id)["status"] == "applied"
    assert any(a["id"] == app_id for a in list_applications(status="applied"))
    stats = get_stats()
    assert stats["total"] >= 1
    matches = resolve_applications("selfcheck")
    assert matches and matches[0]["id"] == app_id
    delete_application(app_id)
    assert get_application(app_id) is None
    print("db self-check OK —", json.dumps(get_stats()))
