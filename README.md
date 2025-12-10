# Data Science Agent with LangChain & LangGraph

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-science-agent.streamlit.app/)

An AI-powered data science agent that can analyze CSV data, generate visualizations, and perform machine learning tasks using natural language.

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
- **Visualization Support**: Automatically generates and saves Plotly charts
- **Memory Persistence**: Remembers conversation context using LangGraph's checkpointing
- **Tool Results Tracking**: Stores all tool executions and AI responses for better traceability
- **Web Interface**: User-friendly Streamlit interface for easy interaction

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
│   ├── __init__.py
│   ├── config.py          # Configuration and LLM setup
│   ├── graph.py           # Graph construction and execution
│   ├── helpers.py         # Helper functions for code execution
│   └── nodes.py           # LangGraph node functions
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
- **graph.py**: Defines the graph structure and execution logic with DataFrame injection
- **helpers.py**: Contains utility functions for code cleaning, extraction, and execution
- **nodes.py**: Implements the graph nodes (agent, tools, response storage)

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

### 2. **Tool Result Visibility**
The agent was executing code, but I couldn't see what was happening between tool calls. The conversation felt like a black box.

**Solution**: I created a custom `store_response` node and added a `tool_results` list to the state. Now I can track every tool execution and AI response in order. This was actually pretty cool to implement!

### 3. **Code String Parsing**
The LLM sometimes returned code wrapped in markdown blocks (```python ... ```), and other times it had weird escape sequences like `\\n` instead of actual newlines. This broke the `exec()` function constantly.

**Solution**: Built a `clean_code_string()` helper function that handles both markdown removal and escape sequence conversion. Took some trial and error but now it's solid.

### 4. **Making Libraries Available**
I kept getting `NameError: name 'pd' is not defined` even though pandas was imported at the notebook level. Turns out the `exec()` environment is isolated!

**Solution**: Had to explicitly pass all needed libraries (pd, px, go, sklearn) into the `env_vars` dict. Now the agent can use any of these libraries in its generated code.

## Next Steps

- **Support for multiple DataFrames** - right now it only works with one
- **Support for SQL databases** - not just CSV files

## What I Learned

1. **State management** - LangGraph's state manipulation and custom state classes
2. **Tool design** - The `python_repl` tool is simple but needed a lot of thought
3. **Prompting** - Changed the system prompt multiple times to get the best results from the agent
4. **Modularization** - How to structure a project for maintainability and deployment
5. **Web deployment** - Using Streamlit for quick prototyping and deployment

## Contributing

This is a learning project, but if you have suggestions or find bugs, feel free to open an issue!

## License

MIT License - feel free to use this for your own learning!

## Acknowledgments

- LangChain documentation and tutorials
- The LangGraph examples that helped me understand state management
- Streamlit for making deployment so accessible

---

**Note**: This is a learning project and not production-ready. Use at your own risk, especially with the code execution features!