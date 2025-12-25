import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# --- 1. Load Environment File ---
# This ensures that variables in your root .env file (like OPENAI_API_KEY)
# are loaded into the environment (os.environ).
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')


class Settings:
    """
    Application settings loaded directly from environment variables or using defaults.
    """
    
    # --- Data & Chunking Settings ---
    data_dir: str = os.getenv("DATA_DIR", "data/Australian_Privacy_Act.pdf")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 150))
    
    # --- Embedding & FAISS Settings ---
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "hybrid_store/faiss_index")
    bm25_index_dir: str = os.getenv("BM25_INDEX_DIR", "hybrid_store/bm25_index")
    # Setting default to OpenAI's high-performance model for 768 dim compatibility
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", 
        "text-embedding-3-large"
    )
    
    # --- LLM Settings (for your Llama model) ---
    hf_repo_id: str = os.getenv("HF_REPO_ID", "meta-llama/Llama-3.2-3B-Instruct")
    top_k: int = int(os.getenv("TOP_K", 3))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.20))
    threshold: float = float(os.getenv("THRESHOLD", 0.25))
    relative_factor: float = float(os.getenv("RELATIVE_FACTOR", 0.80))
    context_window_size: int = int(os.getenv("CONTEXT_WINDOW_SIZE", 1024))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 1024))

    # --- API Key Settings (Retrieved securely from environment) ---
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    # Generic key for LLM access (e.g., HuggingFace, if needed for Llama)
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")


# Instantiate the settings object
settings = Settings()

# Check and warn if the expected key for OpenAI is not loaded
if 'text-embedding' in settings.embedding_model_name and not settings.openai_api_key:
    print("WARNING: OpenAI embedding model is configured, but OPENAI_API_KEY is missing "
          "or could not be loaded from environment/settings.")