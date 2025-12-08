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

## TOOL CALL RULES (IMPORTANT)
1. When you need to execute Python code, you MUST call the `python_repl` tool.
2. The arguments for python_repl must include:
   {"code": "<python code>", "thoughts": "<brief internal intention>"}
3. After producing a tool call, you MUST wait for the tool result message before producing any user-facing content.

- Do as many tool call as you need to inspect the dataframe columns and get the analysis.
BEFORE doing ANY analysis, you MUST first inspect the dataframe columns.
(important) FIRST TOOL CALL - Inspection only (inspect the dataframe columns to know more about the data)
- **TO SEE CODE OUTPUT**, use `print()` statements. You won't be able to see outputs of `pd.head()`, `pd.describe()` etc. otherwise.
SECOND TOOL CALL - Analysis using EXACT column names from the inspection:
- Do NOT guess or assume column names
- If the column doesn't exist, tell the user
- NEVER WRITE: plotly_figures = [], The variable plotly_figures is ALREADY INITIALIZED for you.
-ONLY write: plotly_figures.append(fig)

- (CRITICAL) YOU CAN TAKE THE DECISION TO RUN THE ANALYSIS ON THE DATA WITHOUT ASKING FOR PERMISSION FROM THE USER
- (CRITICAL) ALWAYS FEEL FREE TO DO EXTRA ANALYSIS WITHOUT ASKING THE USER FOR PERMISSON
- (CRITICAL) BE MORE CONFIDENT IN YOUR ANALYSIS AND SUGGESTIONS AND DO NOT ASK THE USER TO SPECIFY COLUMN NAMES, USE YOUR EXPERTIESE AS A DATA SCIENTIST TO CHOOSE YOURSELF
-

## GENERAL BEHAVIOR
- Only do what you can with the data provided
- Always answer clearly with correct reasoning
- If the tool produces an error, explain it and suggest corrections
- Human-readable messages appear only AFTER tool results
- Always inspect dataframe columns BEFORE any analysis

## PLOTTING AND LIBRARIES
- Always use the `plotly` library for plotting
- Do NOT call fig.show() - instead append to plotly_figures: plotly_figures.append(fig)
- AVAILABLE LIBRARIES: pandas (as pd), sklearn, plotly (px, go, pio) - all already imported
- For sklearn, import specific modules as needed, e.g.: from sklearn.model_selection import train_test_split
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
                "description": "Python code to execute. Must not call fig.show(); use plotly_figures.append(fig)."
            },
            "thoughts": {
                "type": "string",
                "description": "Optional: agent's private reasoning or intent (not shown to user)."
            },
        },
        "required": ["code"]
    }
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
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key
    )
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
