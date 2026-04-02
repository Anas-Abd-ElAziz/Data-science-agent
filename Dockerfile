# Use a lightweight Python base image matching our project
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # uv config
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy the dependency requirements file
COPY requirements.txt .

# Install dependencies using uv directly into the system python
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# The default command (can be overridden in docker-compose or AWS task definition)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
