"""The multi-agent graph — both orchestration patterns live here.

1. SUPERVISOR: every turn starts at the supervisor, which classifies the
   message and routes to exactly one specialist (or the parser first when a
   JD is pasted). Specialists never talk to each other directly.
2. PARALLEL + AGGREGATOR: for a "full kit" the router returns a LIST of node
   names — LangGraph runs Resume Analyst, Outreach, and Interview Prep
   concurrently in one superstep, then joins them in the (LLM-free)
   Aggregator. The Send API is NOT needed: the fan-out set is static, and a
   conditional edge returning a list is the documented mechanism for that.
3. Plus a classic tool-calling (ReAct) loop inside the Tracker agent.

Join-correctness note: the three branch nodes are single-shot (exactly one
node execution each, no internal tool loops), so all three finish in the same
superstep and their edges converge; LangGraph dedupes the target so the
aggregator runs exactly ONCE with the merged state. If a branch ever becomes
multi-step, switch the join to `builder.add_edge([...branches], "aggregator")`.
"""
from typing import Annotated, Optional

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

try:  # langgraph >= 1.0
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pre-1.0 name
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from src.agents.aggregator import aggregator_node
from src.agents.assistant import assistant_node
from src.agents.interview_prep import interview_prep_node
from src.agents.jd_parser import jd_parser_node
from src.agents.outreach import outreach_node
from src.agents.resume_analyst import resume_analyst_node
from src.agents.supervisor import supervisor_node
from src.agents.tracker import tracker_node
from src.tools import TRACKER_TOOLS


class AppState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # chat history, persisted per thread
    route: str          # supervisor's decision this turn
    mode: str           # "single" | "kit"
    jd_text: str        # raw JD pasted this turn ("" if none)
    job: Optional[dict]           # active JobPosting dump (or minimal DB row info)
    application_id: Optional[int]
    # Parallel branch outputs: each branch writes ONLY its own key, so the
    # concurrent superstep never double-writes a channel (no reducer needed).
    resume_report: str
    outreach_drafts: str
    interview_notes: str
    kit_path: str


PARALLEL_SPECIALISTS = ["resume_analyst", "outreach", "interview_prep"]
SINGLE_SPECIALISTS = {"resume_analyst", "outreach", "interview_prep"}
JOB_REQUIRED = {"full_kit", "resume_analyst", "outreach", "interview_prep"}


def route_from_supervisor(state: AppState):
    r = state["route"]
    if state.get("jd_text"):
        return "jd_parser"  # fresh JD always gets parsed first; parser forwards onward
    if r == "jd_parser" or (r in JOB_REQUIRED and not state.get("job")):
        return "assistant"  # graceful failure: ask which job / for the JD text
    if r == "full_kit":
        return PARALLEL_SPECIALISTS  # list return -> parallel fan-out
    return "tracker_agent" if r == "tracker" else r


def after_jd_parser(state: AppState):
    if state.get("mode") == "kit":
        return PARALLEL_SPECIALISTS
    if state["route"] in SINGLE_SPECIALISTS:
        return state["route"]  # e.g. "paste JD + how well do I fit?" -> parse, then analyse
    return END  # plain "track this" — parser already confirmed to the user


def after_specialist(state: AppState):
    return "aggregator" if state.get("mode") == "kit" else END


def build_graph(checkpointer=None):
    builder = StateGraph(AppState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("jd_parser", jd_parser_node)
    builder.add_node("resume_analyst", resume_analyst_node)
    builder.add_node("outreach", outreach_node)
    builder.add_node("interview_prep", interview_prep_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("tracker_agent", tracker_node)
    builder.add_node("tracker_tools", ToolNode(TRACKER_TOOLS))
    builder.add_node("assistant", assistant_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["jd_parser", *PARALLEL_SPECIALISTS, "tracker_agent", "assistant"],
    )
    builder.add_conditional_edges("jd_parser", after_jd_parser, [*PARALLEL_SPECIALISTS, END])
    for node in PARALLEL_SPECIALISTS:
        builder.add_conditional_edges(node, after_specialist, ["aggregator", END])
    builder.add_edge("aggregator", END)
    builder.add_edge("assistant", END)

    # Tracker ReAct loop: model -> (tool calls?) -> ToolNode -> model -> ... -> END
    builder.add_conditional_edges("tracker_agent", tools_condition, {"tools": "tracker_tools", END: END})
    builder.add_edge("tracker_tools", "tracker_agent")

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


if __name__ == "__main__":  # print the real topology for the README
    print(build_graph().get_graph().draw_mermaid())
