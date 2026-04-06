# Pin the distro so package availability does not drift underneath the build.
FROM python:3.12.13-slim-bookworm AS python-base

# Runtime libraries required by the compiled nsjail binary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libc6 \
        libnl-route-3-200 \
        libprotobuf32 \
        libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

FROM python-base AS nsjail-build

ARG NSJAIL_VERSION=3.6

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        bison \
        flex \
        g++ \
        gcc \
        git \
        libnl-route-3-dev \
        libprotobuf-dev \
        libtool \
        make \
        pkg-config \
        protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/nsjail
RUN git clone --depth 1 --branch "${NSJAIL_VERSION}" https://github.com/google/nsjail.git . && \
    make

FROM python-base

# uv configuration
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=nsjail-build /tmp/nsjail/nsjail /usr/local/bin/nsjail

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
