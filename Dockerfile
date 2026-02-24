# Use Python 3.13 slim base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg is required for audio conversion)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY whisper-app.py .
COPY static/ static/

# Copy SSL certificates (required for browser microphone access)
# Note: These should be mounted as volumes in production
COPY cert.pem key.pem ./

# Expose HTTPS port
EXPOSE 8000

# Create directory for temporary files
RUN mkdir -p /tmp

# Run the FastAPI application
CMD ["python", "app.py"]
