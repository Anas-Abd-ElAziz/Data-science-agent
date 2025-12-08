"""Graph construction and query execution for the data science agent."""

import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

from .config import MessagesStateWithTools
from .nodes import call_agent, should_continue, tools_node, store_response


class DataScienceGraph:
    """Graph wrapper that handles dynamic DataFrame injection."""
    
    def __init__(self):
        """Initialize the graph structure."""
        self.memory = MemorySaver()
        self.current_df = None  # Store the current DataFrame
        self._build_graph()
    
    def _build_graph(self):
        """Build the graph structure."""
        workflow = StateGraph(MessagesStateWithTools)
        
        workflow.add_node("agent", call_agent)
        workflow.add_node("tools", self._tools_node_wrapper)
        workflow.add_node("store_response", store_response)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", "store_response", END])
        workflow.add_edge("tools", "agent")
        workflow.add_edge("store_response", END)
        
        self.compiled_graph = workflow.compile(checkpointer=self.memory)
    
    def _tools_node_wrapper(self, state: MessagesStateWithTools) -> dict:
        """Wrapper that retrieves df from the instance."""
        if self.current_df is None:
            raise ValueError("DataFrame not provided. Use invoke(state, df=your_dataframe)")
        return tools_node(state, self.current_df)
    
    def invoke(self, state: dict, config: dict = None, df: pd.DataFrame = None):
        """
        Invoke the graph with a DataFrame.
        
        Args:
            state: Initial state with messages
            config: Configuration dict (e.g., thread_id)
            df: The pandas DataFrame to use for analysis
        
        Returns:
            Final state after graph execution
        """
        if df is None:
            raise ValueError("DataFrame must be provided via df parameter")
        
        # Store the DataFrame in the instance for tools_node to access
        self.current_df = df
        
        return self.compiled_graph.invoke(state, config=config)


# Create a global instance
graph = DataScienceGraph()


def run_query(query: str, df: pd.DataFrame, thread_id: str = "default", 
              debug: bool = False, show_all_responses: bool = True):
    """
    Run a query and display results.
    
    Args:
        query: The user's query string
        df: The pandas DataFrame to analyze
        thread_id: Thread ID for conversation memory (default: "default")
        debug: If True, print debug information (default: False)
        show_all_responses: If True, show all AI responses with content (default: True)
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
        },
        config=config,
        df=df
    )
    
    if debug:
        print(f"\n[DEBUG] Messages: {len(result['messages'])}, Tool results: {len(result.get('tool_results', []))}")
    
    # Collect all AIMessages with content
    ai_responses = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            ai_responses.append(msg.content)
    
    if not ai_responses:
        print("\nResponse: (No response generated - the agent may have hit a recursion limit or encountered an error)\n")
        if debug:
            print("[DEBUG] No AIMessage with content found!")
        return
    
    # Print all responses or just the final one
    if show_all_responses and len(ai_responses) > 1:
        for i, response in enumerate(ai_responses[:-1], 1):
            print(f"\n--- Agent Update {i} ---")
            print(response)
            print()
        
        print(f"\n{'='*60}")
        print("FINAL RESPONSE:")
        print(f"{'='*60}")
        print(f"\n{ai_responses[-1]}\n")
    else:
        print(f"\nResponse: {ai_responses[-1]}\n")