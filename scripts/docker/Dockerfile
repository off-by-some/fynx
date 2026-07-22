ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_PYTHON_VERSION_WARNING=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /workspace

RUN pip install --no-cache-dir "poetry>=2,<3"

COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev,test --no-root --no-ansi

CMD ["bash"]
