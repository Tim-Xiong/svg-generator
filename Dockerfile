FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    git \
    wget \
    libcairo2-dev \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir 'tensorflow[and-cuda]' && \
    pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Copy the whole application
COPY . .

# Install and build star-vector if it exists
# COPY star-vector/ ./star-vector/
# RUN if [ -d "star-vector" ]; then cd star-vector && pip install -e . && cd ..; fi

# Set environment variables for GPU usage
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port for Streamlit
EXPOSE 8501

# Create a healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set entry point
CMD yes | streamlit run app.py --server.port=8501 --server.address=0.0.0.0