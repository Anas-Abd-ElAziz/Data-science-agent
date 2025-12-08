"""Streamlit interface for the Data Science Agent."""

import streamlit as st
import pickle
import os
import uuid
import pandas as pd
from datetime import datetime
from agent import graph
from langchain_core.messages import HumanMessage
from agent.config import set_api_key

# Page configuration
st.set_page_config(
    page_title="Data Science Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS to keep chat input at bottom
st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        bottom: 20px;
        background-color: white;
    }
    
    /* Make the chat messages scrollable */
    .stChatMessageContainer {
        max-height: calc(100vh - 250px);
        overflow-y: auto;
    }
    
    /* Ensure proper spacing */
    .block-container {
        padding-bottom: 5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Data Science Agent")
st.markdown("Ask questions about your data and get insights with visualizations!")
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

# Check if DataFrame is loaded
if st.session_state.df is None:
    st.warning("⚠️ Please upload a CSV file in the sidebar to get started!")
    st.info("👈 Use the file uploader in the sidebar to begin analyzing your data.")
else:
    # Create tabs only when data is loaded
    tab1, tab2 = st.tabs(["💬 Chat", "🔧 Tool Execution Logs"])
    
    # TAB 1: Clean Chat Interface
    with tab1:
        # Create a container for messages that can scroll
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
                        
                        # Display figures after the message
                        if msg.get("figures"):
                            for fig_path in msg["figures"]:
                                if os.path.exists(fig_path):
                                    try:
                                        with open(fig_path, "rb") as f:
                                            fig = pickle.load(f)
                                            st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not load visualization: {e}")

        # Chat input - this will stay at the bottom
        if prompt := st.chat_input("Ask about the data..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Show thinking indicator
            with st.spinner("🤔 Analyzing..."):
                # Invoke the agent with the uploaded DataFrame
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    df=st.session_state.df  # Pass the uploaded DataFrame
                )
                
                # Process tool_results from state
                tool_results = result.get("tool_results", [])
                
                # Collect ONLY NEW figures (ones we haven't displayed before)
                new_figures = []
                final_ai_message = None
                
                for item in tool_results:
                    if item["type"] == "tool_result":
                        # Check each figure to see if it's new
                        figures = item.get("figures", [])
                        for fig_path in figures:
                            if fig_path not in st.session_state.displayed_figures:
                                new_figures.append(fig_path)
                                st.session_state.displayed_figures.add(fig_path)
                    
                    elif item["type"] == "ai_message":
                        # Keep track of the last AI message
                        content = item.get("content", "")
                        if content:
                            final_ai_message = content
                
                # Display the final AI message with only NEW figures
                if final_ai_message:
                    msg = {
                        "role": "assistant",
                        "content": final_ai_message,
                        "figures": new_figures,  # Only attach NEW figures
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(msg)
                    
                    # Store tool results for the logs tab
                    st.session_state.last_tool_results = tool_results
                    
                    # Rerun to display the new message and auto-scroll
                    st.rerun()
                    new_figures = []
                else:
                    # No AI message generated
                    st.warning("⚠️ The agent didn't generate a response. Please try again.")

    # TAB 2: Tool Execution Logs
    with tab2:
        st.markdown("### 🔧 Detailed Tool Execution Logs")
        st.markdown("This tab shows all tool calls and their outputs for debugging.")
        
        # Get the latest result from the graph if available
        if st.session_state.messages:
            st.info("💡 Tool logs are shown here after each query. Ask a question in the Chat tab to see execution details.")
            
            # Show tool execution details for the current session
            if "last_tool_results" in st.session_state:
                for i, item in enumerate(st.session_state.last_tool_results):
                    if item["type"] == "tool_result":
                        with st.expander(f"🔧 Tool Call #{i+1}: {item['tool']}", expanded=True):
                            st.caption(f"⏰ {item.get('timestamp', 'N/A')}")
                            
                            if item.get("stdout"):
                                st.markdown("**📊 Output:**")
                                st.code(item["stdout"], language="text")
                            
                            if item.get("error"):
                                st.error(f"❌ **Error:**\n```\n{item['error']}\n```")
                            
                            if item.get("figures"):
                                st.success(f"✅ Generated {len(item['figures'])} visualization(s)")
                                st.json(item["figures"])
                    
                    elif item["type"] == "ai_message":
                        with st.expander(f"💬 AI Response #{i+1}", expanded=True):
                            st.caption(f"⏰ {item.get('timestamp', 'N/A')}")
                            st.markdown(item.get("content", ""))

# Sidebar with info
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="sidebar_uploader")
    
    if uploaded_file is not None:
        try:
            # Load the CSV
            new_df = pd.read_csv(uploaded_file)
            
            # Check if it's a different file
            if st.session_state.uploaded_filename != uploaded_file.name:
                st.session_state.df = new_df
                st.session_state.uploaded_filename = uploaded_file.name
                # Reset chat when new file is loaded
                st.session_state.messages = []
                st.session_state.thread_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.displayed_figures = set()
                if "last_tool_results" in st.session_state:
                    del st.session_state.last_tool_results
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.rerun()
            else:
                st.success(f"✅ Current file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
    elif st.session_state.df is not None:
        st.info(f"📁 Current: {st.session_state.uploaded_filename}")
    
    st.divider()
    
    st.sidebar.header("🔐 Enter API Key (temporary for this session)")
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    entered_key = st.sidebar.text_input(
        "Google API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter your Google API key here"
    )

    if entered_key and entered_key != st.session_state.api_key:
        # Save into session and initialize the LLM in config
        st.session_state.api_key = entered_key
        try:
            set_api_key(entered_key)
            st.sidebar.success("API key set for this session.")
        except Exception as e:
            st.sidebar.error(f"Failed to set API key: {e}")
    
    st.divider()
    
    st.header("🔧 Session Info")
    st.text(f"Thread ID: {st.session_state.thread_id}")
    st.text(f"Messages: {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.displayed_figures = set()
        if "last_tool_results" in st.session_state:
            del st.session_state.last_tool_results
        st.rerun()