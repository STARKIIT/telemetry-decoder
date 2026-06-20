# Use an official Python slim runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIPELINE_DIR=/app/pipeline

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (git is useful for HF, libgomp for torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Use CPU-only PyTorch build to keep the image lightweight (saves ~1.5 GB space)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch>=2.0 \
    && pip install --no-cache-dir -r requirements.txt

# Copy server application and DSP pipeline scripts
COPY app.py .
COPY pipeline/ ./pipeline/

# Hugging Face Spaces runs on port 7860 by default
EXPOSE 7860

# Run uvicorn to start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
