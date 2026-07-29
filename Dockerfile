# PHMForge — A Scenario-Driven Agentic Benchmark for Industrial Asset Lifecycle Maintenance
#
# Multi-stage Dockerfile producing a platform-independent runtime image.
# Builds on slim-bookworm so it works on linux/amd64 and linux/arm64.
#
# Build:
#   docker build -t phmforge:latest .
#
# Run benchmark verification:
#   docker run --rm phmforge:latest bash -c \
#     "cd /app/demo && .venv/bin/python mcp_servers/verify_servers.py --quick"
#
# Run a 5-scenario Pass@1 benchmark (requires WatsonX credentials):
#   docker run --rm \
#     -e WATSONX_APIKEY=$WATSONX_APIKEY \
#     -e WATSONX_URL=$WATSONX_URL \
#     -e WATSONX_PROJECT_ID=$WATSONX_PROJECT_ID \
#     -v $(pwd)/results:/app/demo/results \
#     phmforge:latest bash -c \
#     "cd /app/demo && .venv/bin/python benchmark_pass1.py \
#         --framework react --model 'ibm/granite-4-h-small' --limit 5"
#
# Launch the dashboard (visit http://localhost:8501):
#   docker run --rm -p 8501:8501 phmforge:latest \
#     bash -c "cd /app/demo && .venv/bin/streamlit run frontend/app.py \
#       --server.address=0.0.0.0 --server.port=8501"

# ---------------------------------------------------------------------------
# Stage 1: builder — install Python deps with uv into an isolated venv
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

# System packages: build tools for native deps (numpy, pandas, torch wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# uv (fast Python package manager). Pinned to a known-good release.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && uv --version

WORKDIR /app

# Copy only the demo's pyproject + minimal lockable surface first so we
# get a cacheable "deps install" layer that doesn't bust on code changes.
COPY ReActXen/src/reactxen/demo/intent_implementation_demo/pyproject.toml \
     /app/demo/pyproject.toml

# Copy the ReActXen package skeleton needed by the demo's editable install.
# We keep it minimal so layer caching is effective.
COPY ReActXen/pyproject.toml /app/ReActXen/pyproject.toml
COPY ReActXen/README.md /app/ReActXen/README.md
COPY ReActXen/src /app/ReActXen/src

# Create a venv inside /app/demo and install deps.
WORKDIR /app/demo
RUN uv venv .venv --python 3.10
ENV VIRTUAL_ENV=/app/demo/.venv
ENV PATH=/app/demo/.venv/bin:$PATH

# Install the demo package in editable mode + MCP/FastMCP/Pydantic.
# We pin numpy/pandas to versions that have prebuilt wheels on
# linux/amd64 and linux/arm64 to avoid long compile times.
RUN uv pip install -e /app/demo \
    && uv pip install \
        "mcp[cli]>=1.26.0" \
        "fastmcp>=2.14.5" \
        "pydantic>=2.0" \
        "streamlit>=1.30.0" \
        "plotly>=5.18.0"

# ---------------------------------------------------------------------------
# Stage 2: runtime — copy venv + source code into a small final image
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm

# Minimal runtime deps (libgomp for numpy/torch BLAS, openssl for HTTPS to
# WatsonX/HuggingFace).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for safer execution
RUN useradd --create-home --shell /bin/bash phmforge
WORKDIR /app

# Copy the prebuilt venv from the builder stage
COPY --from=builder /app/demo/.venv /app/demo/.venv
COPY --from=builder /app/ReActXen /app/ReActXen

# Copy the demo source (everything except the venv we just copied)
COPY ReActXen/src/reactxen/demo/intent_implementation_demo /app/demo

# Make sure the venv we copied takes precedence
ENV VIRTUAL_ENV=/app/demo/.venv
ENV PATH=/app/demo/.venv/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Default data directory (override with --env PHMFORGE_DATA_DIR=/data)
ENV PHMFORGE_DATA_DIR=/app/demo/multi_agent_implementation_demo/PDMBench_Data_Directory/submission096

# Streamlit config: reduce telemetry noise, allow external connections
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Drop privileges
RUN chown -R phmforge:phmforge /app
USER phmforge

WORKDIR /app/demo

EXPOSE 8501

# Healthcheck: verify MCP servers can register tools (no network required)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD .venv/bin/python -c "from mcp_servers.prognostics_server import mcp; \
        assert len(mcp._tool_manager._tools) == 15, 'tool count mismatch'" \
    || exit 1

# Default: run verification suite (no credentials needed; validates tools work)
CMD [".venv/bin/python", "mcp_servers/verify_servers.py", "--quick"]
