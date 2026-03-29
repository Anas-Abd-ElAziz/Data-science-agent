"""Helper functions for code cleaning, extraction, and execution."""

import json
import ast
import traceback
import uuid
from contextlib import redirect_stdout
from io import StringIO
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


def _normalize_message_content(content) -> str:
    """Extract plain text from LangChain message content (str, list of blocks, or None)."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text).strip())
        return "\n".join(part for part in parts if part)

    if content is None:
        return ""

    return str(content).strip()


def clean_code_string(code: str) -> str:
    """
    Clean code string by:
    1. Decoding string-literal style payloads when needed
    2. Removing markdown code blocks
    3. Normalizing escaped newlines/tabs for model-generated code
    """
    if not isinstance(code, str):
        return ""

    cleaned = code.strip()

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, str):
                cleaned = parsed.strip()
        except (SyntaxError, ValueError):
            pass

    if "\\n" in cleaned and "\n" not in cleaned:
        cleaned = cleaned.replace("\\r\\n", "\n")
        cleaned = cleaned.replace("\\n", "\n")
        cleaned = cleaned.replace("\\t", "\t")

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return cleaned


def extract_code_and_thoughts(last_message, tc=None) -> Tuple[str, str]:
    """
    Extract code and thoughts from LLM message or tool call.

    Args:
        last_message: Message object or dict from LLM
        tc: Tool call dict (optional)

    Returns:
        Tuple of (code: str, thoughts: str)
    """
    # Try tool call first (most common case)
    if tc and isinstance(tc, dict):
        args = tc.get("args") or tc.get("arguments", {})

        # If args is a dict, extract directly
        if isinstance(args, dict):
            code = args.get("code", "") or ""
            thoughts = args.get("thoughts", "") or ""
            return code, thoughts

        # If args is a JSON string, parse it
        if isinstance(args, str) and args.strip():
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    code = parsed.get("code", "") or ""
                    thoughts = parsed.get("thoughts", "") or ""
                    return code, thoughts
            except json.JSONDecodeError:
                # Treat as raw code
                return args, ""

    # Try last_message function_call (fallback)
    if isinstance(last_message, dict):
        func = last_message.get("additional_kwargs", {}).get("function_call")
        if isinstance(func, dict):
            args_json = func.get("arguments", "")
            if args_json and isinstance(args_json, str):
                try:
                    parsed = json.loads(args_json)
                    if isinstance(parsed, dict):
                        code = parsed.get("code", "") or ""
                        thoughts = parsed.get("thoughts", "") or ""
                        return code, thoughts
                except json.JSONDecodeError:
                    pass

    # Nothing found
    return "", ""


def serialize_plotly_figure(fig, index: int) -> dict:
    title = None
    try:
        title_text = fig.layout.title.text
        if title_text:
            title = str(title_text)
    except Exception:
        title = None

    return {
        "id": str(uuid.uuid4()),
        "title": title or f"Figure {index}",
        "figure_json": fig.to_json(),
    }


def python_repl(code: str, thoughts: str, df: pd.DataFrame) -> dict:
    """
    Execute Python code and return:
      { stdout: str, result: any or None, figures: [figure payloads], error: str or None }
    """
    # Lazy imports — only pay the load cost when code is actually executed.
    # Python caches these after the first call so there's no repeat overhead.
    import sklearn
    import statsmodels.api as sm

    code = clean_code_string(code)
    serialized_figures = []
    stdout_buf = StringIO()

    env_vars = {
        "__builtins__": __builtins__,
        "df": df.copy(),
        "pd": pd,
        "px": px,
        "go": go,
        "pio": pio,
        "sklearn": sklearn,
        "sm": sm,
        "plotly_figures": [],
    }

    try:
        # redirect_stdout is context-local — safe under concurrent FastAPI requests
        # unlike a global sys.stdout swap which would race across threads.
        with redirect_stdout(stdout_buf):
            exec(code, env_vars, env_vars)

        for index, fig in enumerate(env_vars.get("plotly_figures", []), start=1):
            serialized_figures.append(serialize_plotly_figure(fig, index))

        return {
            "stdout": stdout_buf.getvalue() or "",
            "result": env_vars.get("result", None),
            "figures": serialized_figures,
            "error": None,
        }
    except Exception:
        return {
            "stdout": stdout_buf.getvalue() or "",
            "result": None,
            "figures": serialized_figures,
            "error": traceback.format_exc(),
        }
