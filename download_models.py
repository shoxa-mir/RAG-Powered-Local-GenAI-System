"""
Download all required models to the local models/ directory.
Run this once to make the system fully offline-capable.

Usage:
    python download_models.py
"""

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


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
    """Download SOLAR-10.7B-Instruct GGUF (Q4_K_M) for llama-cpp-python."""
    from huggingface_hub import hf_hub_download

    repo_id = "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF"
    filename = "solar-10.7b-instruct-v1.0.Q4_K_M.gguf"
    save_dir = MODELS_DIR / "SOLAR-10.7B-Instruct-v1.0-GGUF"
    save_dir.mkdir(exist_ok=True)
    target = save_dir / filename

    if target.exists():
        print(f"[OK] LLM model already exists: {target}")
        return

    print(f"[...] Downloading LLM model: {repo_id}/{filename}")
    print("      This is ~6.6GB (Q4_K_M quantized) ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(save_dir),
    )
    print(f"[OK] Saved to {target}")


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading models for Local GenAI System Demo")
    print(f"Target directory: {MODELS_DIR}")
    print("=" * 60)

    print("\n--- 1/3: Embedding Model (BAAI/bge-m3) ---")
    download_embedding_model()

    print("\n--- 2/3: Reranker Model (dragonkue/bge-reranker-v2-m3-ko) ---")
    download_reranker_model()

    print("\n--- 3/3: LLM Model (SOLAR-10.7B-Instruct-GGUF Q4_K_M) ---")
    print("NOTE: The GGUF model is ~6.6GB. Works on CPU or GPU.")
    response = input("Download LLM model now? [y/N]: ").strip().lower()
    if response == "y":
        try:
            download_llm_model()
        except ImportError:
            print("[SKIP] huggingface_hub not installed. Run: pip install huggingface-hub")
    else:
        print("[SKIP] LLM model download skipped.")

    print("\n" + "=" * 60)
    print("Done! Update config.json to use local paths:")
    print('  "embedding": { "model_name": "./models/bge-m3" }')
    print('  "reranker": { "model_name": "./models/bge-reranker-v2-m3-ko" }')
    print('  "llm": { "model_path": "./models/SOLAR-10.7B-Instruct-v1.0-GGUF/solar-10.7b-instruct-v1.0.Q4_K_M.gguf" }')
    print("=" * 60)
