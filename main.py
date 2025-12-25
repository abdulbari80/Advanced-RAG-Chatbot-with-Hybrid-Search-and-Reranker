import os
from langchain_core.documents import Document
from src.rag.hybrid_store import HybridStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.chunker import DocumentChunker
from src.rag.config import settings
from src.rag.logger import get_logger

logger = get_logger(__name__)

PDF_PATH = settings.data_dir

def main():
    """
    Creates hybrid indices and runs a sample query through the RAG pipeline.
    """
    # Step 1: Create Chunks
    logger.info("Chunking starts...")
    chunker = DocumentChunker()
    chunks = chunker.create_chunks(PDF_PATH)
    if not chunks:
        logger.error("No chunks were created. Exiting.")
    else:
        logger.info(f"Total chunks created: {len(chunks)}")
    # Step 2: Create Hybrid Store
    logger.info("Now creating hybrid store...")
    store = HybridStore()
    store.create_store(chunks)

if __name__ == "__main__":
    main()
