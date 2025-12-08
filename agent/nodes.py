"""LangGraph node functions for the data science agent."""

import uuid
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END

from .config import MessagesStateWithTools, system_message, get_llm_with_tools
from .helpers import extract_code_and_thoughts, python_repl



def call_agent(state: MessagesStateWithTools) -> dict:
    """
    Call the LLM with tools. Ensures the system message is present.
    Uses config.get_llm_with_tools() to obtain the LLM at runtime.
    """
    messages = state["messages"]
    has_system = any(isinstance(m, SystemMessage) for m in messages)

    if not has_system:
        messages = [SystemMessage(content=system_message)] + messages

    # Retrieve LLM (this will raise a helpful error if not set)
    llm_with_tools = get_llm_with_tools()
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesStateWithTools) -> str:
    """
    Routing function to determine the next node.
    
    Returns:
        "tools" if there are tool calls to execute
        "store_response" if there were previous tool results and this is the final response
        END if conversation is complete
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if there are tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Check if there were previous tool results (meaning this is a response after tool execution)
    if state.get("tool_results") and len(state["tool_results"]) > 0:
        # Check if the last tool_result is not already an AI message
        if state["tool_results"][-1].get("type") != "ai_message":
            return "store_response"
    return END


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
    
    # Get existing tool_results or start with empty list - DEFINE THIS EARLY!
    existing_results = state.get("tool_results") if isinstance(state.get("tool_results"), list) else []
    tool_results = existing_results.copy() if existing_results else []
    
    # Robustly extract tool_calls from a message object or dict
    tool_calls = []
    try:
        tool_calls = getattr(last_message, "tool_calls", None) or []
    except Exception:
        tool_calls = []

    if not tool_calls and isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls", []) or []

    if not tool_calls:
        return {"messages": [], "tool_results": tool_results}  # Return tool_results here too!

    tool_messages = []
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        tool_call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", str(uuid.uuid4()))

        code, thoughts = extract_code_and_thoughts(last_message, tc)
        code = code or ""
        thoughts = thoughts or ""

        if name == "python_repl":
            tool_result = python_repl(code=code, thoughts=thoughts, df=df)
        else:
            tool_result = {"stdout": "", "result": None, "figures": [], "error": f"Unknown tool: {name}"}

        # Store the raw result for later display formatting
        normalized = {
            "type": "tool_result",
            "tool": name,
            "stdout": tool_result.get("stdout", "") or "",
            "result": tool_result.get("result", None),
            "figures": tool_result.get("figures", []) or [],
            "error": tool_result.get("error", None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "_structured": False
        }
        tool_results.append(normalized)
        
        # Format the result as a readable string for the LLM
        result_parts = []
        
        if tool_result.get("stdout"):
            result_parts.append(f"STDOUT:\n{tool_result['stdout']}")
        
        if tool_result.get("result") is not None:
            result_parts.append(f"RESULT:\n{tool_result['result']}")
        
        if tool_result.get("figures"):
            for fig_path in tool_result["figures"]:
                result_parts.append(f"FIGURE SAVED: {fig_path}")
        
        if tool_result.get("error"):
            result_parts.append(f"ERROR:\n{tool_result['error']}")
        
        content = "\n\n".join(result_parts) if result_parts else "Tool execution completed successfully."
        
        # Create ToolMessage so the LLM can see the results
        tool_msg = ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=name
        )
        
        tool_messages.append(tool_msg)
    return {"messages": tool_messages, "tool_results": tool_results}


def store_response(state: MessagesStateWithTools) -> dict:
    """
    Capture the AI's final message after tool execution and store it in tool_results.
    This allows displaying tool results and AI responses in order.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    content = ""
    if hasattr(last_message, "content"):
        content = last_message.content
    elif isinstance(last_message, dict):
        content = last_message.get("content", "")
    
    ai_message_result = {
        "type": "ai_message",
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "_structured": False
    }
    
    state.setdefault("tool_results", [])
    state["tool_results"].append(ai_message_result)

    return {}
