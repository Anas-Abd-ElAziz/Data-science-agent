"""Graph construction and query execution for the data science agent."""

from typing import Callable

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
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


def run_query(
    query: str,
    df: pd.DataFrame,
    llm_with_tools,
    thread_id: str = "default",
    debug: bool = False,
    show_all_responses: bool = True,
):
    from .service import normalize_agent_result

    graph = DataScienceGraph(llm_with_tools=llm_with_tools, df_getter=lambda: df)
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
        },
        config=config,
    )

    normalized = normalize_agent_result(result)

    if debug:
        print(f"\n[DEBUG] Messages: {len(result['messages'])}, Tool results: {len(result.get('tool_results', []))}")

    # Extract clean strings from the new JSON-safe messages
    ai_responses = [
        msg["content"] for msg in normalized.get("messages", [])
        if msg.get("role") == "ai" and msg.get("content")
    ]

    if not ai_responses:
        print("\nResponse: (No response generated - the agent may have hit a recursion limit or encountered an error)\n")
        return

    if show_all_responses and len(ai_responses) > 1:
        for i, response in enumerate(ai_responses[:-1], 1):
            print(f"\n--- Agent Update {i} ---")
            print(response)
            print()

        print(f"\n{'=' * 60}")
        print("FINAL RESPONSE:")
        print(f"{'=' * 60}")
        print(f"\n{ai_responses[-1]}\n")
    else:
        print(f"\nResponse: {ai_responses[-1]}\n")
