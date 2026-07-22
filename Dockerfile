# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS frontend-build
WORKDIR /workspace
COPY frontend/package.json frontend/package-lock.json frontend/
RUN npm --prefix frontend ci
COPY frontend frontend
COPY backend/src/tutor/api/static backend/src/tutor/api/static
RUN npm --prefix frontend run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

RUN addgroup --system tutor && adduser --system --ingroup tutor tutor
WORKDIR /app

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/alembic.ini backend/alembic.ini
COPY backend/src backend/src
COPY --from=frontend-build \
    /workspace/backend/src/tutor/api/static/dist \
    backend/src/tutor/api/static/dist
RUN python -m pip install --no-cache-dir "./backend[api,pilot,llm]"

USER tutor
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn tutor.api.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips ${TUTOR_TRUSTED_PROXY_CIDRS:-127.0.0.1"]
