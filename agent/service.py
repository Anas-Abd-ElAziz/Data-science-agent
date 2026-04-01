"""Shared service layer for Streamlit now and FastAPI later."""

from datetime import datetime
from io import BytesIO
import hashlib
import os

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .config import DEFAULT_MODEL, build_llm_with_tools
from .graph import DataScienceGraph
from .helpers import _normalize_message_content

try:
    from langfuse.langchain import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None


EXCEL_ENGINES = {
    ".xls": "xlrd",
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
    ".xlsb": "pyxlsb",
    ".ods": "odf",
    ".odf": "odf",
    ".odt": "odf",
}

SUPPORTED_UPLOAD_TYPES = ("csv", *[ext.lstrip(".") for ext in EXCEL_ENGINES])


def build_thread_id(prefix: str = "session") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_figure_identifier(figure_payload) -> str:
    if not isinstance(figure_payload, dict):
        return ""

    figure_id = figure_payload.get("id")
    if figure_id:
        return str(figure_id)

    figure_json = figure_payload.get("figure_json", "")
    if figure_json:
        return hashlib.sha256(figure_json.encode("utf-8")).hexdigest()

    return ""


def load_tabular_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    extension = os.path.splitext(filename)[1].lower()
    buffer = BytesIO(file_bytes)

    if extension == ".csv":
        return pd.read_csv(buffer)

    if extension in EXCEL_ENGINES:
        return pd.read_excel(buffer, engine=EXCEL_ENGINES[extension])

    raise ValueError(
        "Unsupported file type. Please upload one of: "
        + ", ".join(sorted(SUPPORTED_UPLOAD_TYPES))
    )


def get_uploaded_file_signature(file_bytes: bytes, filename: str) -> dict:
    return {
        "name": filename,
        "size": len(file_bytes),
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
    }


def extract_final_answer(new_tool_results: list[dict], new_messages=None) -> str:
    # Tier 1 — preferred: the ai_message stored by store_response is always
    # the correct final answer for this turn and is never stale.
    for item in reversed(new_tool_results):
        if item.get("type") == "ai_message" and item.get("content"):
            return str(item["content"]).strip()

    # Tier 2 — scan messages for a clean terminal AIMessage (no tool_calls).
    for message in reversed(new_messages or []):
        if isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                continue
            content = _normalize_message_content(message.content)
            if content:
                return content

    return ""


class AgentSession:
    """Per-session runtime container with no module-level mutable state."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self.df = None
        self.memory = MemorySaver()
        self.llm_with_tools = None
        self.graph = None
        self.thread_id = build_thread_id()
        self.messages = []
        self.last_tool_results = []
        self.uploaded_file_signature = None
        self.figures = []

        if api_key:
            self.set_api_key(api_key, model=model)

    def _rebuild_graph(self):
        if self.llm_with_tools is None:
            self.graph = None
            return

        self.graph = DataScienceGraph(
            llm_with_tools=self.llm_with_tools,
            df_getter=lambda: self.df,
            memory=self.memory,
        )

    def set_api_key(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        if model:
            self.model = model

        self.llm_with_tools = build_llm_with_tools(api_key=api_key, model=self.model)
        self._rebuild_graph()

    def load_uploaded_file(self, file_bytes: bytes, filename: str):
        file_signature = get_uploaded_file_signature(file_bytes, filename)
        file_changed = file_signature != self.uploaded_file_signature

        self.df = load_tabular_bytes(file_bytes, filename)
        self.uploaded_file_signature = file_signature

        if file_changed:
            self.clear_memory()

        return {
            "file_signature": file_signature,
            "file_changed": file_changed,
        }

    def clear_memory(self):
        self.memory = MemorySaver()
        self.thread_id = build_thread_id()
        self.messages = []
        self.last_tool_results = []
        self.figures = []
        self._rebuild_graph()

    def _register_new_figures(self, figure_payloads: list[dict]) -> list[dict]:
        for figure_payload in figure_payloads:
            self.figures.append(figure_payload)

        return figure_payloads

    def run(
        self,
        query: str,
        thread_id: str | None = None,
        recursion_limit: int = 100,
        langfuse_handler=None,
    ):
        if self.df is None:
            raise ValueError("DataFrame not set for this session")
        if self.graph is None:
            raise RuntimeError("LLM not initialized for this session")

        if thread_id:
            self.thread_id = thread_id

        callbacks = [langfuse_handler] if langfuse_handler else None
        config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": recursion_limit,
            "callbacks": callbacks,
        }

        # Clear tool_results from previous queries in the same session
        try:
            self.graph.compiled_graph.update_state(config, {"tool_results": None})
        except Exception:
            # First run might not have state
            pass

        self.messages.append({"role": "user", "content": query})

        hit_recursion_limit = False
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config,
            )
        except Exception as e:
            # Catch GraphRecursionError regardless of the exact import path —
            # LangGraph changed the exception location across versions.
            if (
                "recursion" not in type(e).__name__.lower()
                and "recursion limit" not in str(e).lower()
            ):
                raise
            hit_recursion_limit = True
            # Salvage whatever partial state was checkpointed before the cutoff.
            try:
                partial_state = self.graph.get_state(config)
                result = partial_state.values if partial_state.values else {}
            except Exception:
                result = {}

        normalized_result = normalize_agent_result(result)
        normalized_result["hit_recursion_limit"] = hit_recursion_limit
        self.last_tool_results = normalized_result.get("tool_results", [])

        new_figures = self._register_new_figures(normalized_result.get("figures", []))
        normalized_result["figures"] = new_figures

        answer = normalized_result.get("answer")
        if hit_recursion_limit and not answer:
            answer = (
                "⚠️ The agent reached its step limit before completing. "
                "Try a simpler or more specific question, or ask me to continue."
            )
            normalized_result["answer"] = answer

        # Always store the message if we have an answer OR figures.
        # Gemini sometimes returns empty text after tool execution, but the
        # figures are still valid and should be shown to the user.
        if not answer and new_figures:
            answer = f"Here are the {len(new_figures)} figure(s) I generated."
            normalized_result["answer"] = answer

        if answer or new_figures:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": answer or "",
                    "figures": new_figures,
                    "timestamp": normalized_result["timestamp"],
                }
            )

        normalized_result["thread_id"] = self.thread_id
        normalized_result["session_messages"] = list(self.messages)
        normalized_result["session_figures"] = list(self.figures)
        normalized_result["uploaded_file_signature"] = self.uploaded_file_signature
        return normalized_result


def normalize_agent_result(result: dict) -> dict:
    # tool_results only contains the current turn (tools_node resets to [] each turn).
    tool_results = result.get("tool_results", []) or []

    # All messages are available for the fallback answer extraction, but the
    # primary source is the ai_message entry in tool_results (set by store_response).
    all_messages = result.get("messages", []) or []

    final_ai_message = extract_final_answer(tool_results, all_messages)

    figures = []
    for item in tool_results:
        if item.get("type") == "tool_result":
            figures.extend(item.get("figures", []))

    return {
        "answer": final_ai_message,
        "tool_results": tool_results,
        "figures": figures,
        "timestamp": datetime.now().isoformat(),
        "messages": [
            {
                "role": "ai" if isinstance(m, AIMessage) else "human",
                "content": _normalize_message_content(m.content),
            }
            for m in result.get("messages", [])
            if hasattr(m, "content")
        ],
    }
