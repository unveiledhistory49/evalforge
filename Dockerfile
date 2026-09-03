# EvalForge CLI. Runtime deps: numpy only.
FROM python:3.11-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

FROM python:3.11-slim
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/evalforge /usr/local/bin/evalforge
COPY examples/ ./examples/
RUN mkdir -p /app/.evalforge && chown -R appuser:appuser /app
USER appuser
ENV EVALFORGE_STORE=/app/.evalforge PYTHONHASHSEED=0
ENTRYPOINT ["evalforge"]
CMD ["--help"]
