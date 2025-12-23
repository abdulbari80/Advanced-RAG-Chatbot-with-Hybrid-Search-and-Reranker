import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag.logger import get_logger
from src.rag.config import settings
from src.rag.exception import (
    EmbeddingModelError, 
    DocumentLoadError, 
    VectorStoreNotInitializedError
)

logger = get_logger(__name__)

class HybridStore:
    """
    Manages both Dense (FAISS) and Sparse (BM25) indices.
    
    Provides functionality to create, persist, and load a hybrid ensemble 
    retriever optimized for legal document retrieval.
    """

    def __init__(self, embedding_model: Optional[Embeddings] = None):
        """Initialize paths and embedding models."""
        self.faiss_dir: Path = Path(settings.faiss_index_dir)
        self.bm25_dir: Path = Path(settings.bm25_index_dir)
        self.bm25_path: Path = self.bm25_dir / "bm25_retriever.pkl"
        
        # Ensure base directories exist (Fixes FileNotFoundError)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Embedding Model
        if embedding_model:
            self.embedding_model = embedding_model
        elif settings.openai_api_key and 'text-embedding' in settings.embedding_model_name:
            logger.info(f"Using OpenAI Embeddings: {settings.embedding_model_name}")
            self.embedding_model = OpenAIEmbeddings(
                model=settings.embedding_model_name, 
                openai_api_key=settings.openai_api_key
            )
            logger.info("OpenAI Embedding model initialized.")
        else:
            logger.info(f"Using HF Embeddings: {settings.embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

    def create_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Creates both FAISS and BM25 indices from chunks. 
        Automatically overrides existing indices if present.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        # Check if index already exists for logging purposes
        if (self.faiss_dir / "index.faiss").exists():
            logger.info("Existing index found. Overriding with updated documents...")

        try:
            documents: List[Document] = []
            for chunk in chunks:
                meta = chunk.get('metadata', {})
                # Professional metadata injection for enhanced Sparse retrieval
                enhanced_text = f"ID: {meta.get('unit_id', '')} Title: {meta.get('unit_title', '')}\n{chunk['text']}"
                documents.append(Document(page_content=enhanced_text, metadata=meta))

            # 1. Build and Persist FAISS (Dense)
            logger.info("Building FAISS vector index...")
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.vector_store.save_local(str(self.faiss_dir))

            # 2. Build and Persist BM25 (Sparse)
            logger.info("Building BM25 keyword index...")
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            
            with open(self.bm25_path, "wb") as file:
                pickle.dump(self.bm25_retriever, file)

            logger.info(f"Hybrid Store successfully built with {len(documents)} documents.")

        except Exception as e:
            logger.exception("Hybrid store creation failed.")
            raise EmbeddingModelError(f"Critical error during indexing: {str(e)}")

    def load_store(self) -> None:
        """Loads both indices from disk into memory."""
        try:
            # Load FAISS
            if not (self.faiss_dir / "index.faiss").exists():
                raise VectorStoreNotInitializedError(f"FAISS index not found at {self.faiss_dir}")
                
            self.vector_store = FAISS.load_local(
                str(self.faiss_dir), 
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Load BM25
            if not self.bm25_path.exists():
                raise DocumentLoadError(f"BM25 pickle missing at {self.bm25_path}")
                
            with open(self.bm25_path, "rb") as file:
                self.bm25_retriever = pickle.load(file)

            logger.info("Hybrid indices loaded successfully.")

        except Exception as e:
            logger.exception("Failed to load hybrid store.")
            raise DocumentLoadError(f"Loading error: {e}")

    def get_ensemble_retriever(
        self, 
        top_k_dense: int = 5, 
        top_k_sparse: int = 5, 
        weight_dense: float = 0.5
    ) -> EnsembleRetriever:
        """
        Constructs and returns a Hybrid EnsembleRetriever.
        """
        if not self.vector_store or not self.bm25_retriever:
            logger.error("Attempted to retrieve without loading stores.")
            raise VectorStoreNotInitializedError()

        # Configure individual retrievers
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k_dense})
        self.bm25_retriever.k = top_k_sparse
        
        weight_sparse = round(1.0 - weight_dense, 2)
        
        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[weight_sparse, weight_dense]
        )