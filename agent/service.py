"""Shared service layer for Streamlit now and FastAPI later."""

from datetime import datetime
from io import BytesIO
import hashlib
import os

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .config import build_llm_with_tools
from .graph import build_graph


EXCEL_ENGINES = {
    ".xls": "xlrd",
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
    ".xlsb": "pyxlsb",
    ".ods": "odf",
    ".odf": "odf",
    ".odt": "odf",
}

SUPPORTED_UPLOAD_TYPES = tuple(["csv", *[ext.lstrip(".") for ext in EXCEL_ENGINES]])


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


class AgentSession:
    """Per-session runtime container with no module-level mutable state."""

    def __init__(
        self, api_key: str | None = None, model: str = "gemini-2.5-flash-lite"
    ):
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
        self._known_figure_ids = set()

        if api_key:
            self.set_api_key(api_key, model=model)

    def _get_df(self):
        return self.df

    def _rebuild_graph(self):
        if self.llm_with_tools is None:
            self.graph = None
            return

        self.graph = build_graph(
            llm_with_tools=self.llm_with_tools,
            df_getter=self._get_df,
            memory=self.memory,
        )

    def set_api_key(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        if model:
            self.model = model

        self.llm_with_tools = build_llm_with_tools(api_key=api_key, model=self.model)
        self._rebuild_graph()
        return self.llm_with_tools

    def set_dataframe(self, df):
        self.df = df

    def load_uploaded_file(self, file_bytes: bytes, filename: str):
        df = load_tabular_bytes(file_bytes, filename)
        file_signature = get_uploaded_file_signature(file_bytes, filename)
        file_changed = file_signature != self.uploaded_file_signature

        self.set_dataframe(df)
        self.uploaded_file_signature = file_signature

        if file_changed:
            self.clear_memory()

        return {
            "dataframe": df,
            "file_signature": file_signature,
            "file_changed": file_changed,
        }

    def clear_memory(self):
        self.memory = MemorySaver()
        self.thread_id = build_thread_id()
        self.messages = []
        self.last_tool_results = []
        self.figures = []
        self._known_figure_ids = set()
        self._rebuild_graph()

    def _register_new_figures(self, figure_payloads: list[dict]) -> list[dict]:
        new_figures = []

        for index, figure_payload in enumerate(figure_payloads, start=1):
            figure_id = get_figure_identifier(figure_payload) or f"figure_{index}"
            if figure_id in self._known_figure_ids:
                continue

            self._known_figure_ids.add(figure_id)
            new_figures.append(figure_payload)
            self.figures.append(figure_payload)

        return new_figures

    def run(self, query: str, thread_id: str | None = None):
        if self.df is None:
            raise ValueError("DataFrame not set for this session")
        if self.graph is None:
            raise RuntimeError("LLM not initialized for this session")

        if thread_id:
            self.thread_id = thread_id

        config = {"configurable": {"thread_id": self.thread_id}}
        self.messages.append({"role": "user", "content": query})

        ## Fixes the issue of streamlit displaying old charts in new Messages
        try:
            current_state = self.graph.get_state(config)
            existing_tool_results_len = (
                len(current_state.values.get("tool_results", []))
                if current_state.values
                else 0
            )
        except Exception as e:
            print(f"Warning: Failed to get state length: {e}")
            existing_tool_results_len = 0

        result = self.graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        normalized_result = normalize_agent_result(result, existing_tool_results_len)
        self.last_tool_results = normalized_result.get("tool_results", [])

        new_figures = self._register_new_figures(normalized_result.get("figures", []))
        normalized_result["figures"] = new_figures

        if normalized_result.get("answer"):
            self.messages.append(
                {
                    "role": "assistant",
                    "content": normalized_result["answer"],
                    "figures": new_figures,
                    "timestamp": normalized_result["timestamp"],
                }
            )

        normalized_result["thread_id"] = self.thread_id
        normalized_result["session_messages"] = list(self.messages)
        normalized_result["session_figures"] = list(self.figures)
        normalized_result["uploaded_file_signature"] = self.uploaded_file_signature
        return normalized_result


def normalize_agent_result(result: dict, existing_tool_results_len: int = 0) -> dict:
    all_tool_results = result.get("tool_results", []) or []
    new_tool_results = all_tool_results[existing_tool_results_len:]

    final_ai_message = ""

    for item in new_tool_results:
        if item.get("type") == "ai_message" and item.get("content"):
            final_ai_message = item["content"]

    new_figures = []
    for item in new_tool_results:
        if item.get("type") == "tool_result":
            new_figures.extend(item.get("figures", []))

    return {
        "answer": final_ai_message,
        "tool_results": new_tool_results,
        "figures": new_figures,
        "timestamp": datetime.now().isoformat(),
        "messages": result.get("messages", []),
    }
