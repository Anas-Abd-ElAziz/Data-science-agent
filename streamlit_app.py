import streamlit as st
import os
import time
import hashlib
import pandas as pd
import plotly.io as pio
from datetime import datetime
from agent import graph
from langchain_core.messages import HumanMessage
from agent.config import set_api_key
import streamlit.components.v1 as components


EXCEL_ENGINES = {
    ".xls": "xlrd",
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
    ".xlsb": "pyxlsb",
    ".ods": "odf",
    ".odf": "odf",
    ".odt": "odf",
}


def load_tabular_file(uploaded_file):
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    uploaded_file.seek(0)

    if extension == ".csv":
        return pd.read_csv(uploaded_file)

    if extension in EXCEL_ENGINES:
        return pd.read_excel(uploaded_file, engine=EXCEL_ENGINES[extension])

    supported_types = ["csv", *[ext.lstrip(".") for ext in EXCEL_ENGINES]]
    raise ValueError(
        "Unsupported file type. Please upload one of: "
        + ", ".join(sorted(supported_types))
    )


def get_uploaded_file_signature(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    return {
        "name": uploaded_file.name,
        "size": len(file_bytes),
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
    }


def get_figure_identifier(figure_payload):
    if not isinstance(figure_payload, dict):
        return ""

    figure_id = figure_payload.get("id")
    if figure_id:
        return str(figure_id)

    figure_json = figure_payload.get("figure_json", "")
    if figure_json:
        return hashlib.sha256(figure_json.encode("utf-8")).hexdigest()

    return ""


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
                use_container_width=True,
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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_file_signature" not in st.session_state:
    st.session_state.uploaded_file_signature = None
if "displayed_figures" not in st.session_state:
    st.session_state.displayed_figures = set()
if "last_tool_results" not in st.session_state:
    st.session_state.last_tool_results = []

#
# Sidebar: upload and API key
#
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV/Excel file",
        type=["csv", "xls", "xlsx", "xlsm", "xlsb", "ods", "odf", "odt"],
        key="sidebar_uploader",
    )

    if uploaded_file is not None:
        try:
            uploaded_file_signature = get_uploaded_file_signature(uploaded_file)
            new_df = load_tabular_file(uploaded_file)

            # If the uploaded file content changes, replace the DataFrame and reset chat state
            if st.session_state.uploaded_file_signature != uploaded_file_signature:
                st.session_state.df = new_df
                st.session_state.uploaded_file_signature = uploaded_file_signature
                st.session_state.messages = []
                st.session_state.thread_id = (
                    f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                st.session_state.displayed_figures = set()
                st.session_state.last_tool_results = []
                st.success(f"✅ Loaded: {uploaded_file.name}")
                # note: don't st.rerun() here; keep flow simple. You may choose to rerun if desired.
            else:
                st.success(f"✅ Current file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error loading the file from your computer: {e}")
    elif st.session_state.df is not None:
        st.info(f"📁 Current: {st.session_state.uploaded_file_signature['name']}")
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
            set_api_key(entered_key)
            st.success("API key set for this session.")
        except Exception as e:
            st.error(f"Failed to set API key: {e}")
    st.divider()

    st.header("🔧 Session Info")
    st.text(f"Thread ID: {st.session_state.thread_id}")
    st.text(f"Messages: {len(st.session_state.messages)}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = (
            f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        st.session_state.displayed_figures = set()
        st.session_state.last_tool_results = []
        st.experimental_rerun()

#
# Main area: tabs (messages + logs). Note: chat_input will be placed *below* tabs.
#
tab1, tab2 = st.tabs(["💬 Chat", "🔧 Tool Execution Logs"])

with tab1:
    st.subheader("Chat")
    # Render chat messages
    for msg in st.session_state.messages:
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
    if st.session_state.messages:
        scroll_to_bottom()

with tab2:
    st.subheader("🔧 Tool Execution Logs")
    if not st.session_state.last_tool_results:
        st.info("Tool logs will appear here after a query.")
    else:
        for i, item in enumerate(st.session_state.last_tool_results):
            if item.get("type") == "tool_result":
                with st.expander(
                    f"🔧 Tool Call #{i + 1}: {item.get('tool', 'unknown')}",
                    expanded=True,
                ):
                    st.caption(f"⏰ {item.get('timestamp', 'N/A')}")
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
if st.session_state.df is None:
    st.warning("Upload a CSV/Excel file in the sidebar to enable the chat input.")
else:
    # top-level chat_input (not inside any container like tab/expander/sidebar)
    prompt = st.chat_input("Ask about the data...")

    if prompt:
        # Immediately append user message (so UI shows it)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message right away
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the agent (pass df)
        with st.spinner("🤔 Analyzing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    df=st.session_state.df,  # pass the DataFrame into the agent
                )
            except Exception as e:
                # surface invocation errors and stop
                st.error(f"Agent invocation failed: {e}")
                # Optionally remove the user's message if invocation failed
                # st.session_state.messages.pop()
            else:
                tool_results = result.get("tool_results", [])
                st.session_state.last_tool_results = tool_results

                # Gather new figures only
                new_figures = []
                final_ai_message = None

                for item in tool_results:
                    if item.get("type") == "tool_result":
                        for index, figure_payload in enumerate(
                            item.get("figures", []), start=1
                        ):
                            figure_id = (
                                get_figure_identifier(figure_payload)
                                or f"generated_{index}_{len(new_figures)}"
                            )
                            if figure_id not in st.session_state.displayed_figures:
                                new_figures.append(figure_payload)
                                st.session_state.displayed_figures.add(figure_id)
                    elif item.get("type") == "ai_message":
                        content = item.get("content", "")
                        if content:
                            final_ai_message = content

                if final_ai_message:
                    msg = {
                        "role": "assistant",
                        "content": final_ai_message,
                        "figures": new_figures,
                        "timestamp": datetime.now().isoformat(),
                    }
                    st.session_state.messages.append(msg)

                    # Display assistant message immediately
                    with st.chat_message("assistant"):
                        st.markdown(final_ai_message)
                        if new_figures:
                            render_figures(
                                new_figures,
                                key_prefix=f"current_{msg['timestamp']}",
                            )

                else:
                    st.warning(
                        "⚠️ The agent didn't generate a response. Please try again."
                    )

        # scroll to bottom after the new messages render
        scroll_to_bottom()
