import streamlit as st
import pickle
import os
import uuid
import time
import pandas as pd
from datetime import datetime
from agent import graph
from langchain_core.messages import HumanMessage
from agent.config import set_api_key
import streamlit.components.v1 as components


# Page configuration
st.set_page_config(
    page_title="Data Science Agent",
    page_icon="🤖",
    layout="wide"
)

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
st.markdown("If the model doesnt respond from the first time, please ask the same question again")

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
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "displayed_figures" not in st.session_state:
    st.session_state.displayed_figures = set()
if "last_tool_results" not in st.session_state:
    st.session_state.last_tool_results = []

#
# Sidebar: upload and API key
#
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV/Excel file", type=["csv", "xlsx"], key="sidebar_uploader")

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)

            # If different file, replace DF and reset chat state
            if st.session_state.uploaded_filename != uploaded_file.name:
                st.session_state.df = new_df
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.messages = []
                st.session_state.thread_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.displayed_figures = set()
                st.session_state.last_tool_results = []
                st.success(f"✅ Loaded: {uploaded_file.name}")
                # note: don't st.rerun() here; keep flow simple. You may choose to rerun if desired.
            else:
                st.success(f"✅ Current file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error loading the file from your computer: {e}")
    elif st.session_state.df is not None:
        st.info(f"📁 Current: {st.session_state.uploaded_filename}")
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
        st.session_state.thread_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                    for fig_path in msg["figures"]:
                        if os.path.exists(fig_path):
                            try:
                                with open(fig_path, "rb") as f:
                                    fig = pickle.load(f)
                                    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
                            except Exception as e:
                                st.warning(f"⚠️ Could not load visualization: {e}")

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
                with st.expander(f"🔧 Tool Call #{i+1}: {item.get('tool', 'unknown')}", expanded=True):
                    st.caption(f"⏰ {item.get('timestamp', 'N/A')}")
                    if item.get("stdout"):
                        st.markdown("**📊 Output:**")
                        st.code(item["stdout"])
                    if item.get("error"):
                        st.error(f"❌ **Error:**\n```\n{item['error']}\n```")
                    if item.get("figures"):
                        st.success(f"✅ Generated {len(item['figures'])} visualization(s)")
                        st.json(item["figures"])
            elif item.get("type") == "ai_message":
                with st.expander(f"💬 AI Response #{i+1}", expanded=True):
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
                        for fig_path in item.get("figures", []):
                            if fig_path not in st.session_state.displayed_figures:
                                new_figures.append(fig_path)
                                st.session_state.displayed_figures.add(fig_path)
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
                            for fig_path in new_figures:
                                if os.path.exists(fig_path):
                                    try:
                                        with open(fig_path, "rb") as f:
                                            fig = pickle.load(f)
                                            st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not load visualization: {e}")

                else:
                    st.warning("⚠️ The agent didn't generate a response. Please try again.")

        # scroll to bottom after the new messages render
        scroll_to_bottom()
