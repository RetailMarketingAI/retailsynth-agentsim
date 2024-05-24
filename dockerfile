# `python-base` sets up all our shared environment variables

FROM python:3.10.10-slim as builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Lightweight runtime image, used for retail_agent specific code execution
FROM python:3.10.10-slim as runtime

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Get the poetry.lock and pyproject.toml files from builder layer.
COPY --from=builder /app /app  
COPY --from=builder /usr/local /usr/local
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY workflow ./workflow
COPY src/retail_agent ./retail_agent
COPY tests/cfg ./tests/cfg
COPY README.md ./README.md

RUN poetry install --only main

CMD ["python", "workflow/1_collect_offline_data.py"]