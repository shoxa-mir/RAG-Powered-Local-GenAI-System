FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools \
    build-essential cmake ninja-build git curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set up Python symlink
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Register CUDA driver stub so linker finds libcuda.so
RUN echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig

# Build llama-cpp-python with CUDA
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV CUDACXX=/usr/local/cuda/bin/nvcc
RUN pip3 install --no-cache-dir "llama-cpp-python[server]>=0.3.0"

# Install CPU PyTorch (embedding model is CPU-only)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install app dependencies
COPY requirements-app.txt .
RUN pip3 install --no-cache-dir -r requirements-app.txt

# Copy all source files
COPY app.py config.json serve_llm.py download_models.py ./
COPY templates/ templates/
COPY static/ static/

# Create supervisord config
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/supervisord.conf

# Create startup script
RUN mkdir -p /app/scripts
RUN cat > /app/scripts/startup.sh << 'EOFSCRIPT'
#!/bin/bash
set -e

echo "[Startup] Checking/downloading models..."
python download_models.py --auto

echo "[Startup] Creating data directories..."
mkdir -p /data/qdrant /data/documents /app/models

echo "[Startup] Syncing models to persistent storage (if needed)..."
if [ ! -d "/data/models" ]; then
  cp -r ./models/* /data/models/ 2>/dev/null || echo "[Startup] Models already in /data"
fi

echo "[Startup] Starting supervisord..."
exec supervisord -c /etc/supervisor/supervisord.conf
EOFSCRIPT
RUN chmod +x /app/scripts/startup.sh

# Create data directory
RUN mkdir -p /app/data

# Expose ports
EXPOSE 8001 8000 6333

# Entrypoint: startup script
ENTRYPOINT ["/app/scripts/startup.sh"]
