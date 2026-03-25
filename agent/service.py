"""Shared service layer for Streamlit now and FastAPI later."""

from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .config import build_llm_with_tools
from .graph import build_graph


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

    def clear_memory(self):
        self.memory = MemorySaver()
        self._rebuild_graph()

    def run(self, query: str, thread_id: str):
        if self.df is None:
            raise ValueError("DataFrame not set for this session")
        if self.graph is None:
            raise RuntimeError("LLM not initialized for this session")

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        return normalize_agent_result(result)


def normalize_agent_result(result: dict) -> dict:
    tool_results = result.get("tool_results", []) or []
    final_ai_message = ""

    for item in tool_results:
        if item.get("type") == "ai_message" and item.get("content"):
            final_ai_message = item["content"]

    new_figures = []
    for item in tool_results:
        if item.get("type") == "tool_result":
            new_figures.extend(item.get("figures", []))

    return {
        "answer": final_ai_message,
        "tool_results": tool_results,
        "figures": new_figures,
        "timestamp": datetime.now().isoformat(),
        "messages": result.get("messages", []),
    }
