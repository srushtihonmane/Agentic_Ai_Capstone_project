"""Environment, paths, and LLM factories.

Groq free-tier limits are enforced *per model*, so routing runs on the small
fast model and specialist work on the large one — the two never queue behind
each other. Each factory attaches a client-side rate limiter and retries so a
burst (e.g. the 3-way parallel kit fan-out) degrades to a short stagger
instead of a 429 failure.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"
OUTPUTS_DIR = ROOT / "outputs"
DB_PATH = STORAGE_DIR / "app.db"
CHROMA_DIR = STORAGE_DIR / "chroma"
RESUME_PATH = DATA_DIR / "sample_resume.md"

load_dotenv(ROOT / ".env")

MODEL_MAIN = os.getenv("GROQ_MODEL_MAIN", "llama-3.3-70b-versatile")
MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")

# Separate limiters per model family (Groq limits are per-model).
# 70B: ~2s between requests, small burst allowance for the parallel fan-out.
_main_limiter = InMemoryRateLimiter(requests_per_second=0.5, check_every_n_seconds=0.1, max_bucket_size=3)
_fast_limiter = InMemoryRateLimiter(requests_per_second=1.0, check_every_n_seconds=0.1, max_bucket_size=2)


def groq_key_present() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def llm_main(temperature: float = 0.3) -> ChatGroq:
    """Large model for specialist reasoning/drafting."""
    return ChatGroq(model=MODEL_MAIN, temperature=temperature, max_retries=3, rate_limiter=_main_limiter)


def llm_fast(temperature: float = 0.0) -> ChatGroq:
    """Small fast model for routing and lightweight replies."""
    return ChatGroq(model=MODEL_FAST, temperature=temperature, max_retries=3, rate_limiter=_fast_limiter)


def structured_call(llm: ChatGroq, schema, messages):
    """Structured output with a 3-step fallback chain.

    1. Native tool-calling structured output (most reliable on Groq).
    2. JSON mode with the schema pasted into the prompt.
    3. Plain completion + first-JSON-block extraction + Pydantic validation.
    """
    try:
        return llm.with_structured_output(schema).invoke(messages)
    except Exception:
        pass

    hint = HumanMessage(
        content="Respond ONLY with a single JSON object matching this JSON schema, no prose:\n"
        + json.dumps(schema.model_json_schema())
    )
    try:
        return llm.with_structured_output(schema, method="json_mode").invoke(list(messages) + [hint])
    except Exception:
        pass

    raw = llm.invoke(list(messages) + [hint]).content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model output: {raw[:200]}")
    return schema.model_validate_json(match.group(0))


def trim_history(messages: list, keep: int = 8) -> list:
    """Last `keep` conversation messages, safe for Groq's message rules.

    Never slices between a tool-calling AIMessage and its ToolMessages (Groq
    rejects orphaned tool results), and drops system messages from history.
    """
    msgs = [m for m in messages if m.type != "system"]
    start = max(0, len(msgs) - keep)
    # walk back while the window starts on a tool result (or its parent call)
    while start > 0 and msgs[start].type == "tool":
        start -= 1
    if start > 0 and getattr(msgs[start - 1], "tool_calls", None):
        start -= 1
    return msgs[start:]
