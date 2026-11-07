FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STORAGE_DIR=/var/lib/arterio
WORKDIR /app
COPY requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock \
    && useradd --uid 10001 --create-home arterio \
    && mkdir -p /var/lib/arterio \
    && chown arterio:arterio /var/lib/arterio
FROM base AS runtime
COPY app ./app
USER arterio
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_INDEX_URL"
COPY requirements-inference.txt ./requirements-inference.txt
RUN pip install --no-cache-dir -r requirements-inference.txt
COPY app ./app
USER arterio
CMD ["python", "-m", "app.worker"]

FROM runtime AS tests
USER root
COPY requirements-dev.lock ./requirements-dev.lock
RUN pip install --no-cache-dir -r requirements-dev.lock
COPY tests ./tests
COPY scripts ./scripts
USER arterio
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
