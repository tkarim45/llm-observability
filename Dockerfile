FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[serve]"

COPY api ./api
# Seeds offline demo traffic, then serves the observability API.
EXPOSE 8000
CMD ["sh", "-c", "python -m llmobs.cli seed --reset --n 40 && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
