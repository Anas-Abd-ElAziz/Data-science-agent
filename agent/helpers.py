"""Helper functions for code cleaning, extraction, and execution."""

import os
import sys
import json
import traceback
import pickle
import uuid
from io import StringIO
from typing import Tuple

import pandas as pd
import sklearn
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


def clean_code_string(code: str) -> str:
    """
    Clean code string by:
    1. Unescaping newlines and other escape sequences
    2. Removing markdown code blocks
    """
    # Strip markdown code blocks if present FIRST
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\\n")
        # Remove first line (```python or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\\n".join(lines).strip()
        
    # Now handle escape sequences
    # Replace escaped newlines with actual newlines
    code = code.replace('\\\\n', '\n') 
    # Replace escaped tabs with actual tabs
    code = code.replace('\\\\t', '\t')
    # Replace escaped quotes
    code = code.replace("\\\\'", "'")
    code = code.replace('\\\\"', '"')
    # Replace escaped backslashes
    code = code.replace('\\\\\\\\', '\\\\')
    
    return code


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


def python_repl(code: str, thoughts: str, df: pd.DataFrame) -> dict:
    """
    Execute Python code and return:
      { stdout: str, result: any or None, figures: [paths], error: str or None }
    
    Args:
        code: Python code to execute
        thoughts: Agent's reasoning (not used in execution)
        df: The pandas DataFrame to make available in execution environment
    
    Returns:
        Dictionary with stdout, result, figures, and error
    """
    code = clean_code_string(code)
    os.makedirs("images/plotly_figures/pickle", exist_ok=True)
    saved_figures = []
    stdout_buf = StringIO()
    
    original_stdout = sys.stdout
    try:
        sys.stdout = stdout_buf
        
        env_vars = {
            "__builtins__": __builtins__,
            "df": df,
            "pd": pd,           
            "px": px,           
            "go": go,           
            "pio": pio,        
            "sklearn": sklearn, 
            "plotly_figures": []
        }
        exec(code, env_vars, env_vars)

        for fig in env_vars.get("plotly_figures", []):
            fname = f"images/plotly_figures/pickle/{uuid.uuid4()}.pickle"
            with open(fname, "wb") as f:
                pickle.dump(fig, f)
            saved_figures.append(fname)

        stdout_val = stdout_buf.getvalue()
        result_val = env_vars.get("result", None)

        return {
            "stdout": stdout_val or "",
            "result": result_val,
            "figures": saved_figures,
            "error": None
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "stdout": stdout_buf.getvalue() or "",
            "result": None,
            "figures": saved_figures,
            "error": tb
        }
    finally:
        sys.stdout = original_stdout
