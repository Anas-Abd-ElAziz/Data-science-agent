# Data Science Agent with LangChain & LangGraph

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-science-agent.streamlit.app/)

An AI-powered data science agent that can analyze CSV and spreadsheet data, generate visualizations, and perform machine learning tasks using natural language.

## Overview

This project is a conversational AI agent built with LangChain and LangGraph that can:
- Analyze pandas DataFrames through natural language queries
- Generate interactive Plotly visualizations
- Perform machine learning tasks with scikit-learn
- Maintain conversation history across sessions
- Execute Python code dynamically in a controlled environment

## Features

- **Natural Language Interface**: Ask questions about your data in plain English
- **Smart Code Execution**: The agent writes and executes Python code to answer your queries
- **Visualization Support**: Automatically generates interactive Plotly charts and keeps them as in-memory JSON for rendering
- **Memory Persistence**: Remembers conversation context using LangGraph's checkpointing
- **Tool Results Tracking**: Stores all tool executions and AI responses for better traceability
- **Web Interface**: User-friendly Streamlit interface for easy interaction
- **Spreadsheet Upload Support**: Accepts CSV, Excel, and OpenDocument spreadsheet files and resets chat context when the uploaded file changes

## Tech Stack

- **LangChain**: For LLM integration and tool management
- **LangGraph**: For building the agent workflow with state management
- **Google Gemini 2.5 Flash Lite**: The underlying language model
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **Scikit-learn**: Machine learning capabilities
- **Streamlit**: Web interface for the application

## Live Demo

Try the live application here: [Data Science Agent on Streamlit](https://data-science-agent.streamlit.app/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Anas-Abd-ElAziz/Data-science-agent
cd Data-science-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
# Windows
set GOOGLE_API_KEY=your_api_key_here

# Linux/Mac
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

Then open your browser to the URL shown in the terminal (typically `http://localhost:8501`).

The Streamlit uploader supports `.csv`, `.xls`, `.xlsx`, `.xlsm`, `.xlsb`, `.ods`, `.odf`, and `.odt` files.

### Using the Jupyter Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook Finalmodel.ipynb
```

2. Run all cells to initialize the agent

3. Use the `run_query()` function to interact:
```python
run_query("Show me the top 5 loan amounts")
run_query("Create a bar chart of loan status distribution")
run_query("What's the correlation between income and loan amount?")
```

## Project Structure

```
Data-science-agent/
├── agent/
│   ├── __init__.py        # Public package API
│   ├── config.py          # Configuration, system prompt, and LLM setup
│   ├── graph.py           # LangGraph graph construction (DataScienceGraph)
│   ├── helpers.py         # Code cleaning, extraction, and python_repl execution
│   ├── nodes.py           # LangGraph node functions
│   └── service.py         # Shared service layer (AgentSession, serialization)
├── streamlit_app.py       # Streamlit web interface
├── Finalmodel.ipynb       # Original Jupyter notebook
├── requirements.txt       # Python dependencies
└── README.md
```

## Architecture

The agent uses a custom LangGraph workflow:

```
START → Agent → Tools → Agent → Store Response → END
```

- **Agent Node**: Calls the LLM to decide what to do
- **Tools Node**: Executes Python code with the `python_repl` tool
- **Store Response Node**: Captures AI responses for display
- **Custom State**: Extends `MessagesState` to track tool results and dataframe

## Key Components

### Agent Module (`agent/`)

- **config.py**: Handles API key configuration, system prompts, and LLM initialization
- **graph.py**: Defines the `DataScienceGraph` class and execution logic with DataFrame injection
- **helpers.py**: Contains utility functions for code cleaning, extraction, and execution
- **nodes.py**: Implements the graph nodes (agent, tools, response storage)
- **service.py**: The shared backend service layer — `AgentSession`, file loading, result normalization, and figure deduplication. Has no Streamlit dependency, making it reusable for a future API layer.

### Streamlit Interface

The `streamlit_app.py` provides:
- CSV file upload functionality
- Interactive chat interface
- Visualization display
- Tool execution logs
- Session management

## Challenges I Faced

### 1. **Global State Management**
At first, I was using `globals().get("df")` everywhere to access the dataframe. It worked in the notebook but would break if I tried to turn this into a proper module or restart the kernel.

**Solution**: I learned to use LangGraph's state management properly by adding the dataframe to my custom state class and implementing a graph wrapper for dynamic DataFrame injection. This made the code much cleaner and more maintainable!

Later, I ran into a second version of the same problem: the app relied on module-level objects like a global graph instance and a global tool-bound LLM. That worked for a single Streamlit flow, but it would become a problem for a future FastAPI version because multiple sessions could end up sharing runtime state.

**Solution**: I introduced a shared backend session layer with `AgentSession`, moved graph/model creation to dependency-injected builders, and used `agent/__init__.py` to expose the backend pieces cleanly. This keeps Streamlit working while making the core architecture ready for a future API layer.

### 2. **Tool Result Visibility**
The agent was executing code, but I couldn't see what was happening between tool calls. The conversation felt like a black box.

**Solution**: I created a custom `store_response` node and added a `tool_results` list to the state. Now I can track every tool execution and AI response in order. This was actually pretty cool to implement!

### 3. **Code String Parsing**
The LLM sometimes returned code wrapped in markdown blocks (```python ... ```), and other times it had weird escape sequences like `\\n` instead of actual newlines. This broke the `exec()` function constantly.

**Solution**: Built a `clean_code_string()` helper function that handles both markdown removal and escape sequence conversion. Took some trial and error but now it's solid.

### 4. **Making Libraries Available**
I kept getting `NameError: name 'pd' is not defined` even though pandas was imported at the notebook level. Turns out the `exec()` environment is isolated!

**Solution**: Had to explicitly pass all needed libraries (pd, px, go, sklearn) into the `env_vars` dict. Now the agent can use any of these libraries in its generated code.

### 5. **Final Response Not Appearing Reliably**
Sometimes the model clearly finished the analysis, but the Streamlit app still showed "The agent didn't generate a response." The real issue was that the final AI message was not always being routed and stored cleanly inside the graph state.

**Solution**: I fixed the LangGraph flow so final AI responses are detected more reliably, normalized before reading, and returned as explicit `tool_results` state updates instead of depending on in-place mutation.

### 6. **Plot Serialization for the UI**
I originally stored generated figures as pickle files on disk so Streamlit could load them later, but that made the app more filesystem-dependent and less suitable for a future API layer.

**Solution**: I switched the app to serialize Plotly figures as JSON payloads in memory, keep them inside `tool_results`, and reconstruct them directly in Streamlit when rendering the chat.

### 7. **Preparing the Backend for FastAPI**
As the project grew, some code accumulated redundancy: a duplicate escape-sequence cleanup pass, a `getattr` chain that a `try/except` block already covered, a one-liner method wrapper, an unused return value, a pass-through factory function, and a Streamlit-version compatibility shim that had been dead since Streamlit 1.33.

**Solution**: Systematically removed all of it. The agent's response dict also leaked raw LangChain message objects (not JSON-serializable), which were replaced with plain `{"role", "content"}` dicts. The backend (`AgentSession`) now has zero Streamlit dependency and returns a fully serializable result — ready to be served by a FastAPI layer.

## Next Steps

- **FastAPI layer** — add REST endpoints (`/sessions`, `/sessions/{id}/query`, `/sessions/{id}/upload`) on top of the existing `AgentSession` backend
- **Support for multiple DataFrames** - right now it only works with one
- **Support for SQL databases** - not just CSV files

## What I Learned

1. **State management** - LangGraph's state manipulation and custom state classes
2. **Tool design** - The `python_repl` tool is simple but needed a lot of thought
3. **Prompting** - Changed the system prompt multiple times to get the best results from the agent
4. **Modularization** - How to structure a project for maintainability and deployment
5. **Web deployment** - Using Streamlit for quick prototyping and deployment
6. **Session architecture** - How to separate UI session handling from reusable backend runtime logic

## Contributing

If you have suggestions or find bugs, feel free to open an issue!

## License

MIT License - feel free to use this for your own learning!

## Acknowledgments

- LangChain documentation and tutorials
- The LangGraph examples that helped me understand state management
- Streamlit for making deployment so accessible
---

**Note**: This is a learning project and not production-ready. Use at your own risk, especially with the code execution features!
---
