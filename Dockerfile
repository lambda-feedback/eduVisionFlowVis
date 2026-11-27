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

# Stop Ultralytics from pip-installing missing packages at runtime; the
# deployed filesystem is read-only, so every dependency comes from poetry.lock
ENV YOLO_AUTOINSTALL="false"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the evaluation function to the app directory
COPY evaluation_function ./evaluation_function

# Lambda extension that keeps the instance awake after the response while a
# background heartbeat asks for it (see evaluation_function/keepalive_extension.py).
# Lambda starts every executable in /opt/extensions, and the file name is the
# extension's name. The launcher is written here, not copied, so it always has
# Unix line endings and the absolute path of this image's Python. -I keeps the
# launcher's Python isolated from PYTHON* settings such as PYTHONPROFILEIMPORTTIME.
COPY evaluation_function/keepalive_extension.py /opt/eduvision-keepalive/keepalive_extension.py
RUN mkdir -p /opt/extensions \
    && printf '#!/bin/sh\nexec %s -I -u /opt/eduvision-keepalive/keepalive_extension.py\n' "$(command -v python3)" \
       > /opt/extensions/eduvision-keepalive \
    && chmod +x /opt/extensions/eduvision-keepalive

# Precompile python files for faster startup. Runs after the COPY so the
# function's own code is included: Lambda's filesystem is read-only, so
# anything not compiled here is recompiled on every cold start.
RUN python -m compileall -q .

# Command to start the evaluation function with
ENV FUNCTION_COMMAND="python"

# Args to start the evaluation function with
ENV FUNCTION_ARGS="-m,evaluation_function.main"

# Talk to the worker over stdio rather than an IPC socket. With ipc, shimmy
# blocks Lambda's 10 s init until Python has opened the socket, which the
# torch import can't do on a cold start. With stdio, init returns at once and
# the request waits on stdin while Python finishes importing.
ENV FUNCTION_INTERFACE="rpc"
ENV FUNCTION_RPC_TRANSPORT="stdio"

# The first request after a cold start waits for torch/ultralytics to import
# and the model to load, which takes longer than shimmy's 30 s default.
ENV FUNCTION_WORKER_SEND_TIMEOUT="90s"

# TEMPORARY: log per-module import times to stderr (shimmy forwards it to
# CloudWatch) to find what dominates the cold start. Remove once measured.
ENV PYTHONPROFILEIMPORTTIME=1

ENV LOG_LEVEL="debug"
