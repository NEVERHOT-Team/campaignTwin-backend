FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir . \
    && useradd --system --no-create-home campaign \
    && mkdir -p runs cache/llm data/calib \
    && chown -R campaign runs cache/llm

USER campaign
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
