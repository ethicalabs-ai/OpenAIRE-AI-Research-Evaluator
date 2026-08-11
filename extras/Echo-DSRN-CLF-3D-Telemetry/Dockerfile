# Use a lightweight Python base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies & Node.js for Vite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution globally
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Copy only requirements to leverage Docker cache
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the entire application code
COPY . .

# Footer attribution text, inlined into the frontend bundle at build time
# (Vite statically replaces import.meta.env.VITE_*). Override with:
#   docker build --build-arg VITE_FOOTER_TEXT="..."
ARG VITE_FOOTER_TEXT="OpenAIRE Graph API data: CC-BY 4.0."
ENV VITE_FOOTER_TEXT=$VITE_FOOTER_TEXT

# Build the frontend locally
RUN cd frontend && npm ci && npm run build

# Ensure entrypoint is executable
RUN chmod +x docker_entrypoint.sh

# Expose the default Gradio port
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["/app/docker_entrypoint.sh"]
