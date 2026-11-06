FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STORAGE_DIR=/var/lib/arterio
WORKDIR /app
COPY requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock \
    && useradd --uid 10001 --create-home arterio \
    && mkdir -p /var/lib/arterio \
    && chown arterio:arterio /var/lib/arterio
COPY app ./app
USER arterio
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS tests
USER root
COPY requirements-dev.lock ./requirements-dev.lock
RUN pip install --no-cache-dir -r requirements-dev.lock
COPY tests ./tests
COPY scripts ./scripts
USER arterio
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
