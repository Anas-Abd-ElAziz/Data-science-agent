"""LangGraph node functions for the data science agent."""

import uuid
from datetime import datetime, timezone
from typing import Callable

from langchain_core.messages import SystemMessage, ToolMessage

from .config import MessagesStateWithTools, system_message
from .helpers import _normalize_message_content, extract_code_and_thoughts, python_repl


def _extract_message_content(message) -> str:
    if hasattr(message, "content"):
        return _normalize_message_content(message.content)

    if isinstance(message, dict):
        return _normalize_message_content(message.get("content", ""))

    return ""


def create_agent_node(llm_with_tools) -> Callable[[MessagesStateWithTools], dict]:
    def call_agent(state: MessagesStateWithTools) -> dict:
        messages = state["messages"]
        has_system = any(isinstance(m, SystemMessage) for m in messages)

        if not has_system:
            messages = [SystemMessage(content=system_message)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return call_agent


def should_continue(state: MessagesStateWithTools) -> str:
    """
    Routing function to determine the next node.

    Returns:
        "tools" if there are tool calls to execute
        "store_response" for all other terminal messages (including empty ones)
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the model issued tool calls, execute them.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Always store the terminal message — even if content is empty.
    # Gemini occasionally emits a blank final message after tool execution;
    # routing through store_response lets extract_final_answer recover the
    # last meaningful content from tool_results.
    return "store_response"


def tools_node(state: MessagesStateWithTools, df) -> dict:
    """
    Look at the last LLM message for tool_calls and execute them in order.
    Store raw tool_result dicts in state["tool_results"] for later formatting.
    Return ToolMessages so the LLM can see the results.

    Args:
        state: Current state with messages and tool_results
        df: The pandas DataFrame to pass to python_repl

    Returns:
        Dictionary with messages (ToolMessages) and updated tool_results
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Build this node invocation's results. The operator.add reducer on
    # tool_results will append these to results from prior loops in the
    # same graph.invoke() call, so all figures survive multi-loop runs.
    tool_results = []

    # Extract tool_calls from the message (LangChain object or plain dict fallback)
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls and isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls", []) or []

    if not tool_calls:
        return {
            "messages": [],
            "tool_results": tool_results,
        }

    content = _extract_message_content(last_message)
    if content:
        tool_results.append(
            {
                "type": "ai_message",
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    tool_messages = []
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        tool_call_id = (
            tc.get("id")
            if isinstance(tc, dict)
            else getattr(tc, "id", str(uuid.uuid4()))
        )

        code, thoughts = extract_code_and_thoughts(last_message, tc)
        code = code or ""
        thoughts = thoughts or ""

        if name == "python_repl":
            tool_result = python_repl(code=code, thoughts=thoughts, df=df)
        else:
            tool_result = {
                "stdout": "",
                "result": None,
                "figures": [],
                "error": f"Unknown tool: {name}",
            }

        # Store the raw result for later display formatting
        normalized = {
            "type": "tool_result",
            "tool": name,
            "code": code,
            "stdout": tool_result.get("stdout", "") or "",
            "result": tool_result.get("result", None),
            "figures": tool_result.get("figures", []) or [],
            "error": tool_result.get("error", None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        tool_results.append(normalized)

        # Format the result as a readable string for the LLM
        result_parts = []

        if tool_result.get("stdout"):
            result_parts.append(f"STDOUT:\n{tool_result['stdout']}")

        if tool_result.get("result") is not None:
            result_parts.append(f"RESULT:\n{tool_result['result']}")

        if tool_result.get("figures"):
            for figure_payload in tool_result["figures"]:
                figure_label = figure_payload.get("title") or figure_payload.get("id")
                result_parts.append(f"FIGURE GENERATED: {figure_label}")

        if tool_result.get("error"):
            result_parts.append(f"ERROR:\n{tool_result['error']}")

        content = (
            "\n\n".join(result_parts)
            if result_parts
            else "Tool execution completed successfully."
        )

        # Create ToolMessage so the LLM can see the results
        tool_msg = ToolMessage(content=content, tool_call_id=tool_call_id, name=name)

        tool_messages.append(tool_msg)
    return {"messages": tool_messages, "tool_results": tool_results}


def create_tools_node(
    df_getter: Callable[[], object],
) -> Callable[[MessagesStateWithTools], dict]:
    def tools_node_wrapper(state: MessagesStateWithTools) -> dict:
        df = df_getter()
        if df is None:
            raise ValueError("DataFrame not provided for tool execution")
        return tools_node(state, df)

    return tools_node_wrapper


def store_response(state: MessagesStateWithTools) -> dict:
    """
    Capture the AI's final message and append it to this turn's tool_results
    so callers can find it as the last ai_message entry.

    With the operator.add reducer on tool_results, this entry is automatically
    appended to the accumulated list — no need to carry forward manually.
    """
    messages = state["messages"]
    last_message = messages[-1]
    content = _extract_message_content(last_message)

    ai_message_result = {
        "type": "ai_message",
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {"tool_results": [ai_message_result]}
