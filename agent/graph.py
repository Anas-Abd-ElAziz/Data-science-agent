"""Graph construction for the data science agent."""

from typing import Callable

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import MessagesStateWithTools
from .nodes import create_agent_node, create_tools_node, should_continue, store_response


class DataScienceGraph:
    """Graph wrapper with injected runtime dependencies."""

    def __init__(
        self, llm_with_tools, df_getter: Callable[[], pd.DataFrame], memory=None
    ):
        self.memory = memory or MemorySaver()
        self.llm_with_tools = llm_with_tools
        self.df_getter = df_getter
        self.compiled_graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MessagesStateWithTools)
        workflow.add_node("agent", create_agent_node(self.llm_with_tools))
        workflow.add_node("tools", create_tools_node(self.df_getter))
        workflow.add_node("store_response", store_response)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, ["tools", "store_response"]
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("store_response", END)

        return workflow.compile(checkpointer=self.memory)

    def invoke(self, state: dict, config: dict = None):
        return self.compiled_graph.invoke(state, config=config)

    def get_state(self, config: dict):
        return self.compiled_graph.get_state(config)
