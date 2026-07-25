FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies required for DWG → DXF conversion
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libredwg-tools && \
    rm -rf /var/lib/apt/lists/*

# Verify that the DWG converter is available
RUN dwg2dxf --version

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --create-home appuser && \
    mkdir -p /app/data/artifacts && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]