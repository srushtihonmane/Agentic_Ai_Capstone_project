"""AI Job Application Command Centre — Streamlit UI.

The UI and the agents are two clients of the same SQLite database: the board
reads/writes it directly, the agents reach it through their tools. The chat
panel streams LangGraph node updates live, so you can watch the supervisor
route and the kit specialists fan out in parallel.

Run:  streamlit run app.py
"""
import json
import uuid

import pandas as pd
import streamlit as st

from src import db, rag
from src.config import MODEL_FAST, MODEL_MAIN, RESUME_PATH, groq_key_present

st.set_page_config(page_title="Job Application Command Centre", page_icon="🎯", layout="wide")

STATUS_BADGES = {
    "saved": "🔖 saved", "applied": "📨 applied", "oa": "🧪 oa",
    "interview": "🎙️ interview", "offer": "🏆 offer", "rejected": "❌ rejected",
}
NODE_LABELS = {
    "supervisor": "🧭 Supervisor — routed the request",
    "jd_parser": "📄 JD Parser — posting parsed, tracked & indexed",
    "resume_analyst": "🧬 Resume Analyst — fit report ready",
    "outreach": "✉️ Outreach — messages drafted",
    "interview_prep": "🎯 Interview Prep — notes ready",
    "aggregator": "📦 Aggregator — kit assembled & saved",
    "tracker_agent": "🗂️ Tracker — thinking / using tools",
    "tracker_tools": "🔧 Tracker tools — database updated",
    "assistant": "💬 Assistant — replying",
}


@st.cache_resource(show_spinner="Compiling the agent graph ...")
def get_graph():
    # Cached across reruns so the InMemorySaver checkpointer (conversation
    # memory) survives Streamlit's script re-execution.
    from src.graph import build_graph

    return build_graph()


def init_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def run_agents(prompt: str) -> str:
    graph = get_graph()
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.status("Agents working ...", expanded=True) as status:
        seen_parallel = False
        for update in graph.stream({"messages": [("user", prompt)]}, config=cfg, stream_mode="updates"):
            for node, payload in update.items():
                if node == "supervisor" and isinstance(payload, dict):
                    route = payload.get("route", "?")
                    status.write(f"🧭 Supervisor — route: `{route}`" + (" (parallel kit fan-out)" if payload.get("mode") == "kit" else ""))
                    if payload.get("mode") == "kit":
                        seen_parallel = True
                else:
                    status.write(NODE_LABELS.get(node, f"• {node}"))
        status.update(label="Done" + (" — 3 specialists ran in parallel" if seen_parallel else ""), state="complete", expanded=False)
    state = graph.get_state(cfg).values
    msg = state["messages"][-1]
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def kpi_row():
    stats = db.get_stats()
    by = stats["by_status"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked applications", stats["total"])
    c2.metric("In process (OA + interview)", by.get("interview", 0) + by.get("oa", 0))
    c3.metric("Offers", by.get("offer", 0))
    c4.metric("Deadlines ≤ 7 days", len(stats["due_soon"]))


# ------------------------------------------------------------------- sidebar
init_state()
with st.sidebar:
    st.title("🎯 Command Centre")
    st.caption("A multi-agent HQ for your job hunt: track, parse, tailor, reach out, prepare.")

    if not groq_key_present():
        st.error("`GROQ_API_KEY` missing. Copy `.env.example` to `.env`, paste your free key from console.groq.com, restart.")
        st.stop()
    st.success("Groq connected", icon="⚡")

    if RESUME_PATH.exists():
        st.info(f"Resume: `{RESUME_PATH.name}`", icon="📄")
    else:
        st.warning(f"Resume missing at `{RESUME_PATH}`", icon="📄")

    col_a, col_b = st.columns(2)
    if col_a.button("Rebuild resume index", use_container_width=True):
        n = rag.ingest_resume()
        st.toast(f"Resume re-indexed: {n} chunks")
    if col_b.button("New conversation", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.chat = []
        st.rerun()

    st.divider()
    st.caption(f"Router: `{MODEL_FAST}`\n\nSpecialists: `{MODEL_MAIN}`\n\nEmbeddings: local MiniLM (Chroma)")

# -------------------------------------------------------------------- header
st.title("AI Job Application Command Centre")
kpi_row()
tab_chat, tab_board, tab_kits = st.tabs(["💬 Command Chat", "📋 Applications Board", "📦 Kits & Artifacts"])

# ---------------------------------------------------------------------- chat
with tab_chat:
    qa1, qa2, qa3 = st.columns(3)
    if qa1.button("📅 What's due this week?", use_container_width=True):
        st.session_state.pending_prompt = "What deadlines do I have in the next 7 days?"
    if qa2.button("📊 Show my pipeline stats", use_container_width=True):
        st.session_state.pending_prompt = "Show my application stats."
    if qa3.button("🤖 What can you do?", use_container_width=True):
        st.session_state.pending_prompt = "What can you do?"

    history = st.container(height=420)
    with history:
        if not st.session_state.chat:
            st.caption(
                "Try: paste a job description and say **“track this and prepare the full application kit”** — "
                "then watch the agents fan out in parallel. Or ask *“how well does my resume fit …?”*"
            )
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    prompt = st.chat_input("Paste a JD, ask for a kit, update a status, ask anything ...")
    if st.session_state.pending_prompt and not prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with history:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    answer = run_agents(prompt)
                except Exception as exc:  # rate limits / network — keep the app alive
                    answer = f"⚠️ Agent run failed: `{exc}`\n\nGroq free-tier rate limits are the usual cause — wait a few seconds and retry."
                st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.rerun()  # refresh KPIs/board with whatever the agents changed

# --------------------------------------------------------------------- board
with tab_board:
    statuses = ["all"] + db.STATUSES
    chosen = st.pills("Filter by status", statuses, default="all", selection_mode="single")
    rows = db.list_applications(status=None if chosen in (None, "all") else chosen)
    if not rows:
        st.info("Nothing here yet — paste a JD in the chat or run `python -m src.setup_demo`.")
    else:
        df = pd.DataFrame(rows)[["id", "company", "role", "status", "deadline", "applied_date", "source", "location"]]
        df["status"] = df["status"].map(lambda s: STATUS_BADGES.get(s, s))
        event = st.dataframe(
            df, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
        )
        sel = event.selection.rows if event and event.selection else []
        if sel:
            row = rows[[r["id"] for r in rows].index(int(df.iloc[sel[0]]["id"]))]
            st.subheader(f"#{row['id']} — {row['role']} @ {row['company']}")
            left, right = st.columns([2, 1])
            with left:
                if row.get("jd_json"):
                    posting = json.loads(row["jd_json"])
                    st.markdown(f"**Summary:** {posting.get('summary', '—')}")
                    st.markdown("**Must-have:** " + (", ".join(posting.get("must_have_skills", [])) or "—"))
                    st.markdown("**ATS keywords:** " + (", ".join(posting.get("ats_keywords", [])) or "—"))
                else:
                    st.caption("No parsed JD stored — paste the posting in chat to enrich this entry.")
                if row.get("notes"):
                    st.markdown(f"**Notes:** {row['notes']}")
                arts = db.list_artifacts(row["id"])
                if arts:
                    st.markdown("**Artifacts:** " + ", ".join(f"`{a['kind']}`" for a in arts))
            with right:
                new_status = st.selectbox("Status", db.STATUSES, index=db.STATUSES.index(row["status"]))
                new_deadline = st.text_input("Deadline (YYYY-MM-DD)", value=row.get("deadline") or "")
                if st.button("Save changes", type="primary", use_container_width=True):
                    db.update_application(row["id"], status=new_status, deadline=new_deadline or None)
                    st.toast(f"Updated #{row['id']}")
                    st.rerun()

# ---------------------------------------------------------------------- kits
with tab_kits:
    arts = db.list_artifacts()
    if not arts:
        st.info("No artifacts yet. Ask for a **full application kit** in the chat — the aggregator saves everything here.")
    else:
        apps = {a["id"]: a for a in db.list_applications()}
        for art in arts:
            app = apps.get(art["application_id"], {})
            title = f"{art['kind']} — {app.get('company', '?')} / {app.get('role', '?')} ({art['created_at']})"
            with st.expander(title):
                st.download_button(
                    "Download markdown", art["content"],
                    file_name=f"{art['kind']}_{app.get('company', 'app')}.md",
                    key=f"dl-{art['id']}",
                )
                st.markdown(art["content"])
