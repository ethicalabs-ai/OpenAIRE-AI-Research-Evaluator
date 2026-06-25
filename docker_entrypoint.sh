#!/bin/bash
set -e

# Ensure we are in the backend directory to find static files correctly
cd /app/backend

if [ "$SERVICE_TYPE" = "worker-cpu" ]; then
    echo "Starting Celery Worker CPU (Queue: cpu)..."
    exec celery -A celery_app.celery_app worker -Q cpu --loglevel=info --concurrency=${CONCURRENCY:-2}
else
    echo "Running Alembic migrations..."
    alembic upgrade head
    echo "Starting Echo-DSRN Research Intent Classifier API Server..."
    exec python server.py
fi
