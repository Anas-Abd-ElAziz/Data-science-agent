"""Helper functions for code cleaning, extraction, and execution."""

import ast
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import pandas as pd


logger = logging.getLogger(__name__)


SANDBOX_TIMEOUT_SECONDS = max(1, int(os.getenv("PYTHON_REPL_TIMEOUT_SECONDS", "20")))
SANDBOX_MEMORY_LIMIT_MB = max(128, int(os.getenv("PYTHON_REPL_MEMORY_LIMIT_MB", "512")))
SANDBOX_RUNNER_PATH = Path(__file__).with_name("sandbox_runner.py")
NSJAIL_CONFIG_PATH = Path(__file__).with_name("nsjail.cfg")
NSJAIL_PATH = shutil.which("nsjail")
NSJAIL_AVAILABLE = NSJAIL_PATH is not None


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
        return "\n".join(parts)

    if content is None:
        return ""

    return str(content).strip()


def clean_code_string(code: str) -> str:
    """Remove markdown code blocks and normalize escaped newlines/tabs."""
    if not isinstance(code, str):
        return ""

    cleaned = code.strip()

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


def extract_code_and_thoughts(tc: dict | None) -> Tuple[str, str]:
    """Extract code and thoughts from a tool call dict.

    Returns:
        Tuple of (code, thoughts). Both empty strings if extraction fails.
    """
    if not tc or not isinstance(tc, dict):
        return "", ""

    args = tc.get("args") or tc.get("arguments", {})

    if isinstance(args, dict):
        return args.get("code", "") or "", args.get("thoughts", "") or ""

    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed.get("code", "") or "", parsed.get("thoughts", "") or ""
        except json.JSONDecodeError:
            return args, ""

    return "", ""


def _make_json_safe(obj):
    """Recursively convert non-JSON-serializable objects to safe equivalents.

    NOTE: Intentionally duplicated in sandbox_runner.py because the sandbox
    runs as an isolated subprocess and cannot import from this package.
    """
    import datetime as dt
    import numpy as np

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
        return list(obj)
    return obj


def serialize_plotly_figure(fig, index: int) -> dict:
    # NOTE: Intentionally duplicated in sandbox_runner.py (subprocess isolation).
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
        "id": str(uuid.uuid4()),
        "title": title or f"Figure {index}",
        "figure_json": figure_json,
    }


def _build_sandbox_env() -> dict[str, str]:
    """Build minimal env for sandbox subprocess.

    Windows needs system variables forwarded for subprocess execution.
    Only relevant for local dev; Docker uses Linux.
    """
    env = {"PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    return env


def _run_code_in_sandbox(code: str, df: pd.DataFrame) -> dict:
    if not SANDBOX_RUNNER_PATH.exists():
        return {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": "Sandbox runner is missing.",
        }

    with TemporaryDirectory(prefix="agent_sandbox_") as sandbox_dir:
        sandbox_path = Path(sandbox_dir)
        code_file = sandbox_path / "tool_code.py"
        dataframe_file = sandbox_path / "dataframe.pkl"
        output_file = sandbox_path / "result.json"
        runner_file = sandbox_path / "sandbox_runner.py"

        jailed_code_file = Path("/sandbox/tool_code.py")
        jailed_dataframe_file = Path("/sandbox/dataframe.pkl")
        jailed_output_file = Path("/sandbox/result.json")
        jailed_runner_file = Path("/sandbox/sandbox_runner.py")

        code_file.write_text(code, encoding="utf-8")
        with dataframe_file.open("wb") as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.copy2(SANDBOX_RUNNER_PATH, runner_file)

        local_python_command = [
            sys.executable,
            "-I",
            str(runner_file),
            "--code-file",
            str(code_file),
            "--df-file",
            str(dataframe_file),
            "--output-file",
            str(output_file),
            "--timeout-seconds",
            str(SANDBOX_TIMEOUT_SECONDS),
            "--memory-limit-mb",
            str(SANDBOX_MEMORY_LIMIT_MB),
        ]

        jailed_python_command = [
            sys.executable,
            "-I",
            str(jailed_runner_file),
            "--code-file",
            str(jailed_code_file),
            "--df-file",
            str(jailed_dataframe_file),
            "--output-file",
            str(jailed_output_file),
            "--timeout-seconds",
            str(SANDBOX_TIMEOUT_SECONDS),
            "--memory-limit-mb",
            str(SANDBOX_MEMORY_LIMIT_MB),
        ]

        used_nsjail = False
        if NSJAIL_AVAILABLE and os.name != "nt":
            used_nsjail = True
            command = [
                str(NSJAIL_PATH),
                "--config",
                str(NSJAIL_CONFIG_PATH),
                "--bindmount",
                f"{sandbox_dir}:/sandbox",
                "--",
                *jailed_python_command,
            ]
        else:
            command = local_python_command

        try:
            completed = subprocess.run(
                command,
                cwd=sandbox_dir,
                env=_build_sandbox_env(),
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT_SECONDS + 1,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "result": None,
                "figures": [],
                "error": (
                    "Python execution timed out after "
                    f"{SANDBOX_TIMEOUT_SECONDS} seconds."
                ),
            }

        if output_file.exists():
            try:
                payload = json.loads(output_file.read_text(encoding="utf-8"))
                return {
                    "stdout": payload.get("stdout", "") or "",
                    "result": payload.get("result", None),
                    "figures": payload.get("figures", []) or [],
                    "error": payload.get("error", None),
                }
            except json.JSONDecodeError:
                pass

        stderr = (completed.stderr or "").strip()

        if used_nsjail and completed.returncode != 0:
            logger.warning(
                "nsjail failed to start sandboxed execution; falling back to subprocess-only sandbox. stderr=%s",
                stderr or "<empty>",
            )
            try:
                completed = subprocess.run(
                    local_python_command,
                    cwd=sandbox_dir,
                    env=_build_sandbox_env(),
                    capture_output=True,
                    text=True,
                    timeout=SANDBOX_TIMEOUT_SECONDS + 1,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                return {
                    "stdout": "",
                    "result": None,
                    "figures": [],
                    "error": (
                        "Python execution timed out after "
                        f"{SANDBOX_TIMEOUT_SECONDS} seconds."
                    ),
                }

            if output_file.exists():
                try:
                    payload = json.loads(output_file.read_text(encoding="utf-8"))
                    return {
                        "stdout": payload.get("stdout", "") or "",
                        "result": payload.get("result", None),
                        "figures": payload.get("figures", []) or [],
                        "error": payload.get("error", None),
                    }
                except json.JSONDecodeError:
                    pass

            stderr = (completed.stderr or "").strip()

        if completed.returncode != 0 and stderr:
            error_message = stderr
        elif completed.returncode != 0:
            error_message = (
                "Sandbox runner failed before returning a result "
                f"(exit code {completed.returncode})."
            )
        else:
            error_message = "Sandbox runner did not return a result."

        return {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": error_message,
        }


def python_repl(
    code: str,
    thoughts: str,
    df: pd.DataFrame,
) -> dict:
    """
    Execute Python code and return:
      { stdout: str, result: any or None, figures: [figure payloads], error: str or None }
    """
    code = clean_code_string(code)
    if not code:
        return {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": "No Python code was provided to python_repl.",
        }

    # Static analysis — shared gate for all execution backends.
    from .sandbox_runner import CodeSafetyError, SafetyVisitor

    try:
        tree = ast.parse(code, mode="exec")
        SafetyVisitor().visit(tree)
    except CodeSafetyError as e:
        return {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": f"Code rejected by safety check: {e}",
        }
    except SyntaxError as e:
        return {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": f"Syntax error in generated code: {e}",
        }

    try:
        return _run_code_in_sandbox(code=code, df=df)
    except Exception:
        return {
            "stdout": "",
            "result": None,
            "figures": [],
            "error": traceback.format_exc(),
        }
