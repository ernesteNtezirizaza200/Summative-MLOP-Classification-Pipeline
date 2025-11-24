# Dockerfile for Brain Tumor Classification Application

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/train data/test data/retrain_uploads static

# Set environment variables to suppress CUDA warnings
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# Expose port for FastAPI
EXPOSE 8000

# Health check (uses PORT environment variable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD sh -c "python -c \"import requests; requests.get('http://localhost:${PORT:-8000}/health')\"" || exit 1

# Run FastAPI server
# Use PORT environment variable if set (for Railway/Render), otherwise default to 8000
# Use shell form to properly expand PORT variable
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info --access-log"]

