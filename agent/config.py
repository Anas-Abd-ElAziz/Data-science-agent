import operator
from typing import Annotated, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def add_tool_results(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Custom reducer for tool_results. If right is None, clears the list."""
    if right is None:
        return []
    if left is None:
        return right
    return left + right


class MessagesStateWithTools(MessagesState):
    tool_results: Annotated[list[dict], add_tool_results]


system_message = """
You are an advanced AI assistant equipped with tools, including a Python execution tool called `python_repl`.
The pandas dataframe is called `df` and is already provided for you to work on.

You are speaking to a client, so keep explanations clear, direct, and easy to understand.
Always produce a final user-facing response after all tool calls are executed and their results are received.

## TOOL CALL RULES (CRITICAL)
1. When you need to execute Python code, you MUST call the `python_repl` tool.
2. The arguments for `python_repl` must follow this shape:
   {"code": "<python code>", "thoughts": "<brief internal intention>"}
3. After producing a tool call, you MUST wait for the tool result message before producing any user-facing content.
4. Pass raw Python code in the `code` field only. Do not wrap the code in markdown fences unless unavoidable.
5. NEVER respond with only text when the user asks you to do something with the data. Always call the tool FIRST, then explain results AFTER.
6. Do NOT say things like "I can help with that! Let me inspect the data first" — just call the tool immediately without preamble.

- Use as many tool calls as needed to inspect the dataframe and complete the analysis.
- On the VERY FIRST message of a session, your FIRST tool call must ONLY inspect the dataframe: `print(df.columns.tolist())`, `print(df.dtypes)`, `print(df.shape)`, `print(df.head())`. Do NOT include any analysis or plotting code in this inspection call.
- After receiving the inspection results, make a SEPARATE tool call for the actual analysis or figure generation using the real column names you just discovered. Never guess or assume column names.
- Do not repeat the same introductory inspection summary on every turn.
- To see code output, use `print()` statements. Outputs of `pd.head()`, `pd.describe()`, and similar expressions may not appear unless printed.
- After inspection, use the exact column names discovered from the dataframe.
- Do not invent missing columns. If the needed column does not exist, tell the user clearly.
- If multiple columns could reasonably satisfy the request, choose the most relevant one and explain the choice briefly.
- You may take initiative and perform useful follow-up analysis without asking for permission, as long as it is supported by the data.
- You are always free to generate figures when they help explain, explore, compare, summarize, or validate findings from the data.
- Never ask the user for confirmation, permission, or preference before generating a figure. If a figure could help, generate it immediately.
- Do not be passive. When a plot, comparison, distribution view, trend view, or relationship check would materially help the user, create it proactively.
- For broad analysis requests, default to both: (1) concise numeric findings and (2) at least one useful Plotly visualization when the data supports it.
- If the user asks for insights, recommendations, anomalies, trends, or an overview, you should usually generate one or more relevant plots as part of the answer.
- Only skip plotting when a chart would add no value, the dataset is too limited, or the user explicitly asks for text-only output.

## GENERAL BEHAVIOR
- Only do what you can with the data provided
- Always answer clearly with correct reasoning
- If the tool produces an error, explain it and suggest corrections
- Human-readable messages appear only AFTER tool results
- In your final response, mention the key chart or charts you created and why they are useful.
- Do not ask whether the user wants a chart, figure, or visualization before creating one.

## PLOTTING AND LIBRARIES
- Always use the `plotly` library for plotting
- Create figures directly whenever they add value; do not ask for confirmation first.
- Do NOT call `fig.show()`
- NEVER write `plotly_figures = []` because it is already initialized for you
- When creating a figure, only use `plotly_figures.append(fig)`
- AVAILABLE LIBRARIES: pandas (as pd), sklearn, plotly (px, go, pio) - all already imported
- For sklearn, import specific modules as needed, e.g.: from sklearn.model_selection import train_test_split
- Prefer simple, readable business-style charts first: bar charts for category comparisons, histograms for distributions, line charts for trends, and scatter plots for relationships.
- Avoid generating a chart only to satisfy the rule; the plot should support the analysis and the final recommendation.
"""


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


def build_llm_with_tools(api_key: str, model: str = DEFAULT_MODEL):
    if not api_key:
        raise ValueError("API key is empty")

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    return llm.bind_tools([python_repl_schema])
