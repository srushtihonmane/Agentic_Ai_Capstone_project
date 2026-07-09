"""CLI verification harness — exercises each layer without the UI.

  python -m src.smoke --rag "python sql projects"      # retrieval sanity
  python -m src.smoke --route                          # supervisor routing on canned utterances
  python -m src.smoke --parse data/sample_jds/nimbuspay_swe_intern.txt
  python -m src.smoke --kit data/sample_jds/arcadia_ml_newgrad.txt
  python -m src.smoke --chat "mark arcadia as applied today"
"""
import argparse
import json
import uuid
from pathlib import Path


def run_rag(query: str) -> None:
    from src import rag
    hits = rag.search_resume(query, k=3)
    print(f"top {len(hits)} resume chunks for {query!r}:")
    for h in hits:
        print(" -", h[:120].replace("\n", " "))


def run_route() -> None:
    from src.agents.supervisor import decide_route
    canned = [
        "Here's a JD I found, track it for me:\n\nSoftware Intern at FooCorp\nRequirements: Python, SQL...",
        "Prepare the full application kit for the arcadia role",
        "How well does my resume fit the NimbusPay internship?",
        "Draft a referral request message for the Meridian data science role",
        "What questions should I prepare for the QuantEdge interview?",
        "Mark TechNova as applied and show my stats",
        "hey, what can you do?",
    ]
    for msg in canned:
        d = decide_route(msg)
        print(f"  {d.route:<15} jd_text={d.has_jd_text!s:<5} hint={d.job_hint!r:<25} <- {msg[:60]!r}")


def _stream(graph, text: str, thread: str) -> None:
    cfg = {"configurable": {"thread_id": thread}}
    for update in graph.stream({"messages": [("user", text)]}, config=cfg, stream_mode="updates"):
        for node, payload in update.items():
            print(f"  [{node}]")
    state = graph.get_state(cfg).values
    print("---- final message ----")
    print(state["messages"][-1].content[:2000])
    return state


def run_parse(path: str) -> None:
    from src.graph import build_graph
    graph = build_graph()
    jd = Path(path).read_text(encoding="utf-8")
    _stream(graph, f"Track this job for me:\n\n{jd}", f"smoke-parse-{uuid.uuid4().hex[:6]}")


def run_kit(path: str) -> None:
    from src.graph import build_graph
    graph = build_graph()
    jd = Path(path).read_text(encoding="utf-8")
    state = _stream(graph, f"Track this and prepare the full application kit:\n\n{jd}",
                    f"smoke-kit-{uuid.uuid4().hex[:6]}")
    for key in ("resume_report", "outreach_drafts", "interview_notes", "kit_path"):
        ok = "OK " if state.get(key) else "MISSING"
        print(f"  {ok} {key}")


def run_chat(text: str) -> None:
    from src.graph import build_graph
    graph = build_graph()
    _stream(graph, text, f"smoke-chat-{uuid.uuid4().hex[:6]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", metavar="QUERY")
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--parse", metavar="JD_FILE")
    ap.add_argument("--kit", metavar="JD_FILE")
    ap.add_argument("--chat", metavar="MESSAGE")
    args = ap.parse_args()
    if args.rag:
        run_rag(args.rag)
    if args.route:
        run_route()
    if args.parse:
        run_parse(args.parse)
    if args.kit:
        run_kit(args.kit)
    if args.chat:
        run_chat(args.chat)
