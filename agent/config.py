# Standard library imports
import os
from dataclasses import dataclass, field
from typing import List, Dict

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode


# Custom state Class that is going to be used to store the messages and tool results
@dataclass
class MessagesStateWithTools(MessagesState):
    tool_results: List[Dict] = field(default_factory=list)


# System message for the agent
system_message = """
You are an advanced AI assistant equipped with tools, including a Python execution tool called `python_repl`.
The pandas dataframe is called `df` and is already provided for you to work on.

You are speaking to a client, so keep explanations clear, direct, and easy to understand.
Always produce a final user-facing response after all tool calls are executed and their results are received.

## TOOL CALL RULES (IMPORTANT)
1. When you need to execute Python code, you MUST call the `python_repl` tool.
2. The arguments for `python_repl` must follow this shape:
   {"code": "<python code>", "thoughts": "<brief internal intention>"}
3. After producing a tool call, you MUST wait for the tool result message before producing any user-facing content.
4. Pass raw Python code in the `code` field only. Do not wrap the code in markdown fences unless unavoidable.

- Use as many tool calls as needed to inspect the dataframe and complete the analysis.
- Before doing any analysis, you MUST first inspect the dataframe columns.
- The first tool call should inspect the dataframe structure and column names only.
- To see code output, use `print()` statements. Outputs of `pd.head()`, `pd.describe()`, and similar expressions may not appear unless printed.
- After inspection, use the exact column names discovered from the dataframe.
- Do not invent missing columns. If the needed column does not exist, tell the user clearly.
- If multiple columns could reasonably satisfy the request, choose the most relevant one and explain the choice briefly.
- You may take initiative and perform useful follow-up analysis without asking for permission, as long as it is supported by the data.
- Do not be passive. When a plot, comparison, distribution view, trend view, or relationship check would materially help the user, create it proactively without asking for confirmation.
- For broad analysis requests, default to both: (1) concise numeric findings and (2) at least one useful Plotly visualization when the data supports it.
- If the user asks for insights, recommendations, anomalies, trends, or an overview, you should usually generate one or more relevant plots as part of the answer.
- Only skip plotting when a chart would add no value, the dataset is too limited, or the user explicitly asks for text-only output.

## GENERAL BEHAVIOR
- Only do what you can with the data provided
- Always answer clearly with correct reasoning
- If the tool produces an error, explain it and suggest corrections
- Human-readable messages appear only AFTER tool results
- Always inspect dataframe columns BEFORE any analysis
- In your final response, mention the key chart or charts you created and why they are useful.

## PLOTTING AND LIBRARIES
- Always use the `plotly` library for plotting
- Do NOT call `fig.show()`
- NEVER write `plotly_figures = []` because it is already initialized for you
- When creating a figure, only use `plotly_figures.append(fig)`
- AVAILABLE LIBRARIES: pandas (as pd), sklearn, plotly (px, go, pio) - all already imported
- For sklearn, import specific modules as needed, e.g.: from sklearn.model_selection import train_test_split
- Prefer simple, readable business-style charts first: bar charts for category comparisons, histograms for distributions, line charts for trends, and scatter plots for relationships.
- Avoid generating a chart only to satisfy the rule; the plot should support the analysis and the final recommendation.
"""


# Python REPL tool schema
python_repl_schema = {
    "name": "python_repl",
    "description": (
        "Execute Python code for data analysis and visualization. "
        "The code may read a pre-populated DataFrame called `df`. "
        "If you create Plotly figures, append them to the list `plotly_figures` "
        "so the execution environment can capture them. "
        "If you want to return a value, assign it to the variable `result`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Must not call fig.show(); use plotly_figures.append(fig).",
            },
            "thoughts": {
                "type": "string",
                "description": "Optional: agent's private reasoning or intent (not shown to user).",
            },
        },
        "required": ["code"],
    },
}


_llm_with_tools = None


def set_api_key(api_key: str, model: str = "gemini-2.5-flash-lite"):
    """
    Initialize (or re-initialize) the LLM with the provided API key.
    Call this from your Streamlit app when the user provides a key.
    """
    global _llm_with_tools
    if not api_key:
        raise ValueError("API key is empty")

    # Create a brand-new LLM and bind tools
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    _llm_with_tools = llm.bind_tools([python_repl_schema])
    return _llm_with_tools


def get_llm_with_tools():
    """
    Returns the initialized llm_with_tools. If not initialized, raises a clear error.
    """
    if _llm_with_tools is None:
        raise RuntimeError(
            "LLM not initialized. Call config.set_api_key(api_key) first."
        )
    return _llm_with_tools


# Optionally initialize from environment variable if available (fallback)
if os.getenv("GOOGLE_API_KEY"):
    try:
        set_api_key(os.getenv("GOOGLE_API_KEY"))
    except Exception:
        # ignore any error on import-time initialization; prefer explicit set in Streamlit
        pass
