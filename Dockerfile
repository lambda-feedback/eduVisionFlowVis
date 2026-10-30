FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12 AS builder

# Must match the Poetry major version that generated poetry.lock
RUN pip install "poetry>=2,<3"

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --without dev --no-root

FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12

# System libraries opencv-python (pulled in by ultralytics) needs to import
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the evaluation function to the app directory
COPY evaluation_function ./evaluation_function

# Precompile python files for faster startup. Runs after the COPY so the
# function's own code is included: Lambda's filesystem is read-only, so
# anything not compiled here is recompiled on every cold start.
RUN python -m compileall -q .

# Command to start the evaluation function with
ENV FUNCTION_COMMAND="python"

# Args to start the evaluation function with
ENV FUNCTION_ARGS="-m,evaluation_function.main"

# The transport to use for the RPC server
ENV FUNCTION_RPC_TRANSPORT="ipc"

ENV LOG_LEVEL="debug"
