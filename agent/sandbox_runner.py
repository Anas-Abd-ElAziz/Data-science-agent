"""Run python_repl code inside a constrained subprocess."""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import traceback
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


MAX_STDOUT_CHARS = 20_000
MAX_ERROR_CHARS = 12_000
MAX_RESULT_JSON_CHARS = 50_000
ALLOWED_IMPORT_PREFIXES = (
    "math",
    "statistics",
    "numpy",
    "pandas",
    "plotly",
    "sklearn",
    "statsmodels",
)
BLOCKED_CALL_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "vars",
    "__import__",
}
BLOCKED_ATTRIBUTE_NAMES = {
    "__class__",
    "__dict__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "builtins",
    "f_globals",
    "joblib",
    "listdir",
    "open",
    "os",
    "pathlib",
    "popen",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "remove",
    "rename",
    "rmdir",
    "save",
    "socket",
    "subprocess",
    "sys",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_html",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "unlink",
    "write_html",
    "write_image",
    "write_json",
}


class CodeSafetyError(ValueError):
    """Raised when the sandbox rejects unsafe code."""


class SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._ensure_allowed_import(alias.name, node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._ensure_allowed_import(node.module or "", node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") or node.id in BLOCKED_CALL_NAMES:
            raise CodeSafetyError(f"Use of '{node.id}' is not allowed in python_repl")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or node.attr in BLOCKED_ATTRIBUTE_NAMES:
            raise CodeSafetyError(
                f"Attribute '{node.attr}' is not allowed in python_repl"
            )
        self.generic_visit(node)

    def _ensure_allowed_import(self, module_name: str, node: ast.AST) -> None:
        if _is_allowed_import(module_name):
            return
        raise CodeSafetyError(
            f"Import '{module_name or '<relative import>'}' is not allowed in python_repl"
        )


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    remaining = len(text) - limit
    return f"{text[:limit]}\n... [truncated {remaining} characters]"


def _is_allowed_import(module_name: str) -> bool:
    if not module_name:
        return False
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in ALLOWED_IMPORT_PREFIXES
    )


# NOTE: Intentionally duplicated from helpers._make_json_safe because the
# sandbox runs as an isolated subprocess and cannot import from the package.
def _make_json_safe(obj):
    import datetime as dt

    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, dt.timedelta)):
        return str(obj)
    if hasattr(obj, "__class__") and "interval" in obj.__class__.__name__.lower():
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_make_json_safe(item) for item in obj)
    return obj


def _normalize_result(result):
    safe_result = _make_json_safe(result)
    try:
        serialized = json.dumps(safe_result)
    except TypeError:
        return str(safe_result)

    if len(serialized) <= MAX_RESULT_JSON_CHARS:
        return safe_result

    return {
        "summary": "Result too large to return directly.",
        "preview": serialized[:MAX_RESULT_JSON_CHARS],
    }


# NOTE: Intentionally duplicated from helpers.serialize_plotly_figure
# (subprocess isolation — same reason as _make_json_safe above).
def _serialize_plotly_figure(fig, index: int) -> dict:
    title = None
    try:
        title_text = fig.layout.title.text
        if title_text:
            title = str(title_text)
    except Exception:
        title = None

    try:
        fig_dict = fig.to_dict()
        cleaned_dict = _make_json_safe(fig_dict)
        figure_json = json.dumps(cleaned_dict)
    except Exception:
        figure_json = json.dumps({"error": "Failed to serialize figure"})

    return {
        "id": f"figure_{index}",
        "title": title or f"Figure {index}",
        "figure_json": figure_json,
    }


def _disable_network() -> None:
    import socket

    def _blocked(*args, **kwargs):
        raise RuntimeError("Network access is blocked inside python_repl")

    for name in ("create_connection", "create_server"):
        if hasattr(socket, name):
            setattr(socket, name, _blocked)

    if hasattr(socket.socket, "connect"):
        socket.socket.connect = _blocked
    if hasattr(socket.socket, "connect_ex"):
        socket.socket.connect_ex = _blocked


def _apply_resource_limits(timeout_seconds: int, memory_limit_mb: int) -> None:
    try:
        import resource
    except ImportError:
        return

    cpu_soft = max(1, timeout_seconds)
    cpu_hard = cpu_soft + 1
    memory_bytes = max(128, memory_limit_mb) * 1024 * 1024
    file_bytes = 10 * 1024 * 1024

    for limit_name, values in (
        ("RLIMIT_CPU", (cpu_soft, cpu_hard)),
        ("RLIMIT_AS", (memory_bytes, memory_bytes)),
        ("RLIMIT_FSIZE", (file_bytes, file_bytes)),
    ):
        resource_limit = getattr(resource, limit_name, None)
        if resource_limit is None:
            continue
        try:
            resource.setrlimit(resource_limit, values)
        except (OSError, ValueError):
            continue


def _build_safe_builtins():
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("Relative imports are not allowed in python_repl")
        if not _is_allowed_import(name):
            raise ImportError(f"Import '{name}' is not allowed in python_repl")
        return __import__(name, globals, locals, fromlist, level)

    return {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "zip": zip,
    }


def _compile_user_code(code: str):
    tree = ast.parse(code, mode="exec")
    SafetyVisitor().visit(tree)
    return compile(tree, "<python_repl>", "exec")


def _execute_user_code(code: str, df, timeout_seconds: int, memory_limit_mb: int):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    _apply_resource_limits(timeout_seconds, memory_limit_mb)

    compiled_code = _compile_user_code(code)
    stdout_buffer = StringIO()
    env_vars = {
        "__builtins__": _build_safe_builtins(),
        "__name__": "__main__",
        # The dataframe is already isolated in this subprocess.
        "df": df,
        "np": np,
        "pd": pd,
        "px": px,
        "go": go,
        "pio": pio,
        "plotly_figures": [],
    }

    with redirect_stdout(stdout_buffer):
        exec(compiled_code, env_vars, env_vars)

    serialized_figures = [
        _serialize_plotly_figure(fig, index)
        for index, fig in enumerate(env_vars.get("plotly_figures", []), start=1)
    ]

    return {
        "stdout": _truncate_text(stdout_buffer.getvalue() or "", MAX_STDOUT_CHARS),
        "result": _normalize_result(env_vars.get("result", None)),
        "figures": serialized_figures,
        "error": None,
    }


def _write_payload(output_file: Path, payload: dict) -> None:
    output_file.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--df-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--memory-limit-mb", type=int, required=True)
    args = parser.parse_args()

    output_file = Path(args.output_file)

    try:
        _disable_network()

        code = Path(args.code_file).read_text(encoding="utf-8")
        with Path(args.df_file).open("rb") as handle:
            df = pickle.load(handle)

        payload = _execute_user_code(
            code,
            df,
            timeout_seconds=args.timeout_seconds,
            memory_limit_mb=args.memory_limit_mb,
        )
    except Exception:
        payload = {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": _truncate_text(traceback.format_exc(), MAX_ERROR_CHARS),
        }

    _write_payload(output_file, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
