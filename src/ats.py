"""Deterministic ATS keyword-coverage scorer.

Runs before the Resume Analyst LLM call: the coverage table shown to the user
is computed here (reproducible, zero tokens), and the LLM only adds the
qualitative analysis on top of it.
"""
import re


def _norm(s: str) -> str:
    """Lowercase and strip punctuation except symbols that matter in tech terms (+ # .)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#. ]", " ", s.lower())).strip()


def keyword_coverage(resume_text: str, keywords: list[str]) -> dict:
    hay = _norm(resume_text)
    present, missing = [], []
    seen = set()
    for kw in keywords:
        k = _norm(kw)
        if not k or k in seen:
            continue
        seen.add(k)
        # boundary match so 'sql' doesn't hit inside 'mysql'
        if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", hay):
            present.append(kw)
        else:
            missing.append(kw)
    total = len(present) + len(missing)
    pct = round(100 * len(present) / total) if total else 0
    return {"coverage_pct": pct, "present": present, "missing": missing}


def coverage_markdown(cov: dict) -> str:
    lines = [
        f"**ATS keyword coverage: {cov['coverage_pct']}%** "
        f"({len(cov['present'])}/{len(cov['present']) + len(cov['missing'])} keywords found)",
        "",
        "| Found in resume | Missing from resume |",
        "|---|---|",
    ]
    p, m = cov["present"], cov["missing"]
    for i in range(max(len(p), len(m))):
        left = f"`{p[i]}`" if i < len(p) else ""
        right = f"`{m[i]}`" if i < len(m) else ""
        lines.append(f"| {left} | {right} |")
    return "\n".join(lines)
