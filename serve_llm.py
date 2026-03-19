"""
Local LLM server using llama-cpp-python with OpenAI-compatible API.
Cross-platform: works on Windows, Linux, and macOS.

Usage:
    python serve_llm.py
    # Server starts at http://localhost:8000/v1

The server exposes an OpenAI-compatible chat completions API,
so app.py connects to it via the standard OpenAI client.
"""

import json
import logging
import sys
from pathlib import Path

import uvicorn
from llama_cpp.server.app import Settings, create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm-server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

LLM_CONFIG = CONFIG["llm"]
MODEL_PATH = str(BASE_DIR / LLM_CONFIG["model_path"])
N_GPU_LAYERS = LLM_CONFIG.get("n_gpu_layers", -1)  # -1 = offload all to GPU
N_CTX = LLM_CONFIG.get("n_ctx", 4096)
PORT = 8000

# Verify model file exists
if not Path(MODEL_PATH).exists():
    logger.error("Model file not found: %s", MODEL_PATH)
    logger.error("Run: python download_models.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Start llama-cpp-python's built-in OpenAI-compatible server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = Settings(
        model=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        chat_format="chatml",
        verbose=True,
        api_key="sk-local",  # dummy key for local use
    )

    app = create_app(settings=settings)

    logger.info("Starting LLM server on port %d...", PORT)
    logger.info("Model: %s", MODEL_PATH)
    logger.info("GPU layers: %s, Context: %d", N_GPU_LAYERS, N_CTX)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
