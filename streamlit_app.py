import streamlit as st
import time
import plotly.io as pio
from agent import AgentSession, SUPPORTED_UPLOAD_TYPES, get_figure_identifier
import streamlit.components.v1 as components

from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse.langchain import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None

_langfuse_handler = None
_langfuse_client = None
if LANGFUSE_AVAILABLE:
    try:
        from langfuse import get_client

        _langfuse_client = get_client()
        _langfuse_handler = CallbackHandler()
    except Exception:
        pass


def summarize_figures(figure_payloads):
    summaries = []
    for index, figure_payload in enumerate(figure_payloads, start=1):
        title = (
            figure_payload.get("title", f"Figure {index}")
            if isinstance(figure_payload, dict)
            else f"Figure {index}"
        )
        summaries.append(
            {
                "id": get_figure_identifier(figure_payload) or f"figure_{index}",
                "title": title,
            }
        )
    return summaries


def render_figures(figure_payloads, key_prefix):
    for index, figure_payload in enumerate(figure_payloads, start=1):
        figure_json = (
            figure_payload.get("figure_json")
            if isinstance(figure_payload, dict)
            else None
        )
        figure_id = get_figure_identifier(figure_payload) or f"figure_{index}"

        if not figure_json:
            st.warning("⚠️ Could not load visualization: missing figure data")
            continue

        try:
            fig = pio.from_json(figure_json)
            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{key_prefix}_{figure_id}_{index}",
            )
        except Exception as e:
            st.warning(f"⚠️ Could not load visualization: {e}")


# Page configuration
st.set_page_config(page_title="Data Science Agent", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-bottom: 6rem; } /* leave room for chat input */
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Data Science Agent")
st.markdown("Ask questions about your data and get insights with visualizations!")
st.markdown(
    "The agent now handles final responses more reliably, but if a response still fails once, retrying the same prompt usually works."
)


def scroll_to_bottom():
    # embed a timestamp to ensure uniqueness so the snippet runs every rerun
    ts = int(time.time() * 1000)
    html = f"""
    <script>
    try {{
        // unique marker: {ts}
        window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }});
        // Also try to focus the chat input if present
        const input = document.querySelector('textarea[placeholder="Ask about the data..."], input[placeholder="Ask about the data..."]');
        if (input) {{
            input.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            input.focus();
        }}
    }} catch(e) {{ console.log("scroll error", e); }}
    </script>
    """
    # don't pass `key=` — older Streamlit may not accept it
    components.html(html, height=1)


# Initialize session state
if "agent_session" not in st.session_state:
    st.session_state.agent_session = AgentSession()
if "ui_warning" not in st.session_state:
    st.session_state.ui_warning = None

agent_session = st.session_state.agent_session

#
# Sidebar: upload and API key
#
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV/Excel file",
        type=list(SUPPORTED_UPLOAD_TYPES),
        key="sidebar_uploader",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            upload_result = agent_session.load_uploaded_file(
                file_bytes,
                uploaded_file.name,
            )

            if upload_result["file_changed"]:
                st.success(f"✅ Loaded: {uploaded_file.name}")
            else:
                st.success(f"✅ Current file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error loading the file from your computer: {e}")
    elif (
        agent_session.df is not None
        and agent_session.uploaded_file_signature is not None
    ):
        st.info(f"📁 Current: {agent_session.uploaded_file_signature['name']}")
    else:
        st.info("⚠️ Please upload a CSV/Excel file to begin.")

    st.divider()

    st.header("🔐 Enter API Key (temporary for this session)")
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    entered_key = st.text_input(
        "Google API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter your Google API key here",
        key="apikey_input",
    )

    if entered_key and entered_key != st.session_state.api_key:
        st.session_state.api_key = entered_key
        try:
            st.session_state.agent_session.set_api_key(entered_key)
            st.success("API key set for this session.")
        except Exception as e:
            st.error(f"Failed to set API key: {e}")
    st.divider()

    st.header("🔧 Session Info")
    st.text(f"Thread ID: {agent_session.thread_id}")
    st.text(f"Messages: {len(agent_session.messages)}")

    if st.button("🗑️ Clear Chat"):
        agent_session.clear_memory()
        st.rerun()

#
# Main area: tabs (messages + logs). Note: chat_input will be placed *below* tabs.
#
tab1, tab2 = st.tabs(["💬 Chat", "🔧 Tool Execution Logs"])

with tab1:
    st.subheader("Chat")
    # Render chat messages
    for msg in agent_session.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if msg.get("figures"):
                    render_figures(
                        msg["figures"],
                        key_prefix=f"history_{msg.get('timestamp', 'message')}",
                    )

    # After rendering messages, ensure view is at the bottom
    if agent_session.messages:
        scroll_to_bottom()

with tab2:
    st.subheader("🔧 Tool Execution Logs")
    if not agent_session.last_tool_results:
        st.info("Tool logs will appear here after a query.")
    else:
        for i, item in enumerate(agent_session.last_tool_results):
            if item.get("type") == "tool_result":
                with st.expander(
                    f"🔧 Tool Call #{i + 1}: {item.get('tool', 'unknown')}",
                    expanded=True,
                ):
                    st.caption(f"⏰ {item.get('timestamp', 'N/A')}")
                    if item.get("code"):
                        st.markdown("**Code Executed:**")
                        st.code(item["code"], language="python")
                    if item.get("stdout"):
                        st.markdown("**📊 Output:**")
                        st.code(item["stdout"])
                    if item.get("error"):
                        st.error(f"❌ **Error:**\n```\n{item['error']}\n```")
                    if item.get("figures"):
                        st.success(
                            f"✅ Generated {len(item['figures'])} visualization(s)"
                        )
                        st.json(summarize_figures(item["figures"]))
            elif item.get("type") == "ai_message":
                with st.expander(f"💬 AI Response #{i + 1}", expanded=True):
                    st.caption(f"⏰ {item.get('timestamp', 'N/A')}")
                    st.markdown(item.get("content", ""))

#
# IMPORTANT: put chat_input OUTSIDE the tabs (top-level) so Streamlit doesn't raise
#
if agent_session.df is None:
    st.warning("Upload a CSV/Excel file in the sidebar to enable the chat input.")
else:
    if st.session_state.ui_warning:
        st.warning(st.session_state.ui_warning)
        st.session_state.ui_warning = None

    # top-level chat_input (not inside any container like tab/expander/sidebar)
    prompt = st.chat_input("Ask about the data...")

    if prompt:
        st.session_state.ui_warning = None

        with st.spinner("🤔 Analyzing..."):
            try:
                result = agent_session.run(
                    query=prompt, langfuse_handler=_langfuse_handler
                )
                if _langfuse_client:
                    _langfuse_client.flush()
            except Exception as e:
                st.error(f"Agent invocation failed: {e}")
            else:
                if not result.get("answer") and not result.get("figures"):
                    st.session_state.ui_warning = (
                        "⚠️ The agent didn't generate a response. Please try again."
                    )
                st.rerun()
