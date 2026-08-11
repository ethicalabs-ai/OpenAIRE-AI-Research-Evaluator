#!/bin/bash
set -e

echo "Starting Echo-DSRN Telemetry Service..."

# Ensure we are in the backend directory to find static files correctly
cd /app/backend

# Launch the unified server
exec python server.py
