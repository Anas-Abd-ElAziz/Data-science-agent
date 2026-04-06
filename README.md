# Data Science Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-science-agent.streamlit.app/)
[![API Docs](https://img.shields.io/badge/API-Docs-blue)](http://data-agent-api-alb-919427104.eu-north-1.elb.amazonaws.com/docs#/Configuration)

Conversational data analysis with LangChain, LangGraph, Gemini, pandas, and Plotly.

## Overview

This project has two entrypoints:

- `streamlit_app.py`: local interactive UI with an in-memory `AgentSession`
- `api.py`: FastAPI backend with optional Redis- and S3-backed persistence for API sessions

The agent can:

- inspect uploaded CSV and spreadsheet files
- write Python to analyze the dataset
- generate Plotly figures
- return tool logs and final answers
- keep conversational state through LangGraph checkpoints

## Current Architecture

The LangGraph flow is:

```text
START -> Agent -> Tools -> Agent -> Store Response -> END
```

Core pieces:

- `agent/config.py`: LLM setup and system prompt
- `agent/nodes.py`: LangGraph nodes
- `agent/helpers.py`: code extraction and sandboxed `python_repl`
- `agent/service.py`: `AgentSession` runtime container
- `agent/graph.py`: graph construction wrapper
- `agent/session_store.py`: Redis-backed API session metadata, recent messages, and figure state
- `agent/dataset_store.py`: S3-backed uploaded dataset storage for the API

## Important Behavior

### FastAPI API

- API sessions are identified by `session_id`
- uploaded datasets can be stored in S3
- session metadata, recent chat history, and restored figures can be stored in Redis
- Redis can also be used as the LangGraph checkpointer backend
- Gemini API keys are set per session and are **memory-only** for live API sessions
- if the API restarts and a session is restored from Redis, the client must call `POST /sessions/{id}/api-key` again before querying

### Streamlit App

- Streamlit still uses a local in-memory `AgentSession`
- it does not automatically use the FastAPI Redis session store or S3 dataset store
- it is best treated as the interactive local/demo UI

### Session TTL

Redis session TTL is refreshed on mutating activity only:

- create session
- set API key
- upload file
- run query
- clear session

Read endpoints do not extend TTL.

## Python Execution Safety

The `python_repl` tool no longer runs raw in-process `exec()` inside the API worker.

Execution backend:

- constrained subprocess sandbox in this repository

The local sandbox uses:

- code validation before execution
- restricted imports and builtins
- blocked network access
- execution timeout
- memory limit controls

## Features

- natural language analysis over uploaded tabular data
- Plotly chart generation
- tool execution logs in both API and Streamlit flows
- Redis-backed API session metadata, recent messages, and figure state
- optional Redis LangGraph checkpoint persistence
- optional S3-backed dataset storage for API uploads
- Langfuse callback-based tracing
- Docker and ECS deployment support

## Setup

1. Clone the repository.

```bash
git clone https://github.com/Anas-Abd-ElAziz/Data-science-agent
cd Data-science-agent
```

2. Create and activate a virtual environment.

```bash
uv venv
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

3. Install dependencies.

```bash
uv pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in the values you need.

## Environment Variables

### Langfuse

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

### Redis

Use either `REDIS_URL` or the host/port fields.

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_USERNAME=
REDIS_PASSWORD=
REDIS_URL=
REDIS_SESSION_TTL_SECONDS=86400
REDIS_SOCKET_TIMEOUT_SECONDS=3
REDIS_CONNECT_TIMEOUT_SECONDS=3
```

Redis is used for:

- API session metadata, recent chat history, and restored figure state
- optional LangGraph checkpoint storage

### S3 Dataset Storage

```env
AWS_REGION=eu-north-1
S3_DATASET_BUCKET=
S3_DATASET_PREFIX=datasets
S3_DELETE_ON_SESSION_DELETE=false
```

When configured, API uploads are stored under keys like:

```text
datasets/<session_id>/<filename>
```

### Python Sandbox

```env
PYTHON_REPL_TIMEOUT_SECONDS=20
PYTHON_REPL_MEMORY_LIMIT_MB=512
```

Behavior:

- on Linux containers with `nsjail` available, code runs with kernel-level isolation on top of the subprocess sandbox
- on platforms without `nsjail`, the app falls back to the constrained subprocess sandbox only

## Running Locally

Gemini API keys are entered at runtime:

- in Streamlit, use the sidebar field
- in FastAPI, call `POST /sessions/{id}/api-key`

### Streamlit

```bash
uv run streamlit run streamlit_app.py
```

### FastAPI

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
docker-compose up --build
```

`docker-compose.yml` runs the API container in privileged mode with `SYS_ADMIN` so `nsjail` works locally the same way it does on ECS EC2.

## API Flow

Typical API usage looks like this:

1. `POST /sessions`
2. `POST /sessions/{id}/api-key`
3. `POST /sessions/{id}/upload`
4. `POST /sessions/{id}/query`
5. optional `GET /sessions/{id}/data/preview`

Important details:

- if Redis is enabled, sessions can be restored after restart
- if S3 is enabled, uploaded datasets can be reloaded after restart
- after a restore, the Gemini API key must be set again because it is not stored in Redis

## Health Endpoint

`GET /health` reports:

- `status`
- `active_sessions`
- `dataset_store`
- `graph_checkpointer`
- `langfuse`
- `python_repl_backend`
- `session_store`

`status` is `ok` during normal operation and `degraded` when optional services like Redis or S3 are configured but currently unavailable.

## AWS / ECS Notes

This stack is currently deployed on ECS EC2, not Fargate.

The API task uses:

- an `executionRoleArn` for image pulls and logs
- a `taskRoleArn` for S3 access from `boto3`
- `networkMode: bridge`
- `requiresCompatibilities: ["EC2"]`
- a privileged container with `SYS_ADMIN` so `nsjail` can start

If you use S3 dataset storage, the task role should allow at least:

- `s3:ListBucket` on the dataset bucket
- `s3:GetObject`
- `s3:PutObject`
- optional `s3:DeleteObject`

Recommended AWS setup:

- Redis for API session metadata and optional graph checkpoints
- S3 for uploaded datasets
- GitHub Actions secrets for deploy-time configuration
- an S3 lifecycle rule to expire dataset objects after `1 day` if you want `24h` retention

The checked-in deploy workflow injects runtime environment variables into the ECS task definition from GitHub secrets. Keep these secrets up to date in GitHub:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `ECS_CLUSTER`
- `ECS_SERVICE`
- `REDIS_URL`
- `S3_DATASET_BUCKET`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL`

## Limitations

- API authentication is still minimal
- Streamlit and FastAPI do not share the same runtime session store
- restored API sessions recover metadata, recent messages, figures, and dataset references, but they still require `set_api_key` again
- Streamlit state and per-query tool results are still in-memory only
- Redis checkpoint storage can grow quickly on small Redis plans
- the Python sandbox is safer than raw `exec()`, but not equivalent to isolated per-job containers

## Project Structure

```text
Data-science-agent/
├── .aws/
│   └── task-definition.json
├── .github/
│   └── workflows/
├── agent/
│   ├── __init__.py
│   ├── checkpoint_store.py
│   ├── config.py
│   ├── dataset_store.py
│   ├── graph.py
│   ├── helpers.py
│   ├── nodes.py
│   ├── service.py
│   └── session_store.py
├── api.py
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.in
├── requirements.txt
└── README.md
```

## Tech Stack

- LangChain
- LangGraph
- Google Gemini
- pandas
- Plotly
- scikit-learn
- FastAPI
- Streamlit
- Redis
- Amazon S3
- Langfuse

## Next Improvements

- add real API authentication and authorization
- persist Streamlit state and per-query tool results beyond process memory
- consider a fully isolated executor service for model-written Python

## License

MIT
