# Backend: Flask API for Cloud Run

A minimal Flask service with a single health endpoint, ready to deploy to Google Cloud Run.

## Endpoints
- `GET /health` → returns plaintext `healthy`
- `GET /` → returns a small JSON status

## Local Development
Using a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
# In another terminal:
curl -s http://127.0.0.1:8080/health
```

## Deploy to Cloud Run (source-based)
From this `backend` folder:

```bash
# Set your region and (optionally) project
REGION=us-central1
SERVICE_NAME=flask-backend
# gcloud config set project YOUR_PROJECT_ID

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated
```

This uses Cloud Buildpacks, the included Procfile, and `requirements.txt`.

## Deploy to Cloud Run (container image)
Build and deploy a container image instead of source:

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
SERVICE_NAME=flask-backend
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"

gcloud builds submit --tag "$IMAGE" .

gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated
```

## Notes
- The service binds to the `PORT` environment variable (default 8080), as required by Cloud Run.
- Gunicorn is used in production (Procfile / Dockerfile); Flask's built-in server is used for local dev.
