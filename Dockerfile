# Container image for the AI PR Reviewer — used for Google Cloud Run (or any
# container host). Render uses render.yaml/buildpacks instead and ignores this.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080). Shell form so $PORT expands.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
