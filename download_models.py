"""
Download all required models to the local models/ directory.
Run this once to make the system fully offline-capable.

Usage:
    python download_models.py

Environment:
    MODELS_DIR: Override models directory (default: /data/models for HF Spaces, ./models for local with env var)
"""

import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_embedding_model():
    """Download BAAI/bge-m3 embedding model."""
    from sentence_transformers import SentenceTransformer

    model_name = "BAAI/bge-m3"
    save_path = MODELS_DIR / "bge-m3"

    if save_path.exists() and any(save_path.iterdir()):
        print(f"[OK] Embedding model already exists: {save_path}")
        return

    print(f"[...] Downloading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    model.save(str(save_path))
    print(f"[OK] Saved to {save_path}")


def download_reranker_model():
    """Download dragonkue/bge-reranker-v2-m3-ko Korean cross-encoder reranker."""
    from sentence_transformers import CrossEncoder

    model_name = "dragonkue/bge-reranker-v2-m3-ko"
    save_path = MODELS_DIR / "bge-reranker-v2-m3-ko"

    if save_path.exists() and any(save_path.iterdir()):
        print(f"[OK] Reranker model already exists: {save_path}")
        return

    print(f"[...] Downloading reranker model: {model_name} (~560MB) ...")
    model = CrossEncoder(model_name)
    model.save(str(save_path))
    print(f"[OK] Saved to {save_path}")


def download_llm_model():
    """Download Yarn-Solar-10b-64k GGUF (Q4_K_M) for llama-cpp-python."""
    from huggingface_hub import hf_hub_download

    repo_id = "MaziyarPanahi/Yarn-Solar-10b-64k-GGUF"
    filename = "Yarn-Solar-10b-64k.Q4_K_M.gguf"
    save_dir = MODELS_DIR / "Yarn-Solar-10B-64K"
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / filename

    if target.exists():
        print(f"[OK] LLM model already exists: {target}")
        return

    print(f"[...] Downloading LLM model: {repo_id}/{filename}")
    print("      This is ~6.5GB (Q4_K_M quantized) ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(save_dir),
    )
    print(f"[OK] Saved to {target}")


if __name__ == "__main__":
    import sys

    auto_mode = "--auto" in sys.argv

    print("=" * 60)
    print("Downloading models for Local GenAI System Demo")
    print(f"Target directory: {MODELS_DIR}")
    print("=" * 60)

    print("\n--- 1/3: Embedding Model (BAAI/bge-m3) ---")
    download_embedding_model()

    print("\n--- 2/3: Reranker Model (dragonkue/bge-reranker-v2-m3-ko) ---")
    download_reranker_model()

    print("\n--- 3/3: LLM Model (Yarn-Solar-10b-64k-GGUF Q4_K_M) ---")
    print("NOTE: The GGUF model is ~6.5GB. Requires GPU for 65K context window.")

    if auto_mode:
        print("[AUTO] Auto-downloading LLM model...")
        try:
            download_llm_model()
        except ImportError:
            print("[SKIP] huggingface_hub not installed. Run: pip install huggingface-hub")
    else:
        response = input("Download LLM model now? [y/N]: ").strip().lower()
        if response == "y":
            try:
                download_llm_model()
            except ImportError:
                print("[SKIP] huggingface_hub not installed. Run: pip install huggingface-hub")
        else:
            print("[SKIP] LLM model download skipped.")

    print("\n" + "=" * 60)
    print("Done! Use paths in config.json (auto-set to /data/models for HF Spaces, ./models for local):")
    print('  "embedding": { "model_name": "/data/models/bge-m3" }')
    print('  "reranker": { "model_name": "/data/models/bge-reranker-v2-m3-ko" }')
    print('  "llm": { "model_path": "/data/models/Yarn-Solar-10B-64K/Yarn-Solar-10b-64k.Q4_K_M.gguf" }')
    print("=" * 60)
