# Multi-stage build for efficient GPU-enabled training container
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy language model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Set Python path to include the app directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Default environment variables for Vertex AI
ENV VERTEX_AI_ENABLED=true
ENV GCS_BUCKET=""
ENV GCP_PROJECT=""
ENV EXPERIMENT_NAME=""
ENV CONFIG_OVERRIDES="{}"

# Create non-root user for security
RUN useradd -m -u 1001 trainer && \
    chown -R trainer:trainer /app

USER trainer

# Default entrypoint for training
ENTRYPOINT ["python", "-m"]
CMD ["scripts.vertex_train"]
