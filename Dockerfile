FROM python:3.11-slim

WORKDIR /app

# Install system deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids downloading ~2GB of CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python deps (app only, no llama-cpp)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy app source
COPY app.py config.json ./
COPY templates/ templates/
COPY static/ static/

EXPOSE 8001

CMD ["python", "app.py"]
