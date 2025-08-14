# Minimal production image for Cloud Run
FROM python:3.12-slim

# Prevents Python from writing .pyc files and enables stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install security updates and runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Use a non-root user for better security
RUN useradd -m appuser
USER appuser

# Cloud Run provides PORT env var
ENV PORT=8080
EXPOSE 8080

# Start via Gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 0 app:app
