FROM python:3.11-slim

WORKDIR /app

# System deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# App dependencies
COPY pyproject.toml uv.lock* ./
RUN pip install "uv==0.4.18" && \
    uv pip install --system .

# Copy code + model artifacts
COPY src/fraud_mlops_demo ./fraud_mlops_demo
COPY ./src/artifacts ./artifacts

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "fraud_mlops_demo.service.app:app", "--host", "0.0.0.0", "--port", "8000"]