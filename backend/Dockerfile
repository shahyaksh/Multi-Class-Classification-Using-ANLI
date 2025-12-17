FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy download script
COPY download_model.py .

# Download model during build (requires WANDB_API_KEY and HF_TOKEN as build args)
ARG WANDB_API_KEY
ARG HF_TOKEN
ENV WANDB_API_KEY=$WANDB_API_KEY
ENV HF_TOKEN=$HF_TOKEN

RUN python download_model.py && rm download_model.py

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
