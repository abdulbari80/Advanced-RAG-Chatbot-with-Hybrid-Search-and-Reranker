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
    VectorStoreNotInitializedError,
    RAGBaseException
)

logger = get_logger(__name__)

OPENAI_API_KEY = settings.openai_api_key
EMBEDDING_MODEL_NAME = settings.embedding_model_name

class HybridStore:
    """
    Manages both Dense (FAISS) and Sparse (BM25) indices for the RAG pipeline.
    
    This class orchestrates the creation, persistence, and loading of dual-mode 
    retrievers. It is specifically optimized for legal document retrieval by 
    injecting structural metadata into the sparse index to improve keyword hits.

    Attributes:
        faiss_dir (Path): Directory where FAISS index files are stored.
        bm25_path (Path): File path for the serialized BM25 retriever.
        embedding_model (Embeddings): The model used for generating vector embeddings.
        vector_store (FAISS): The dense vector index.
        bm25_retriever (BM25Retriever): The sparse keyword index.
    """

    def __init__(self, embedding_model: Optional[Embeddings] = None):
        """
        Initializes the storage paths and sets up the embedding model provider.

        Args:
            embedding_model: Optional pre-initialized embedding model. If None, 
                it defaults to OpenAI or HuggingFace based on settings.
        """
        self.faiss_dir: Path = Path(settings.faiss_index_dir)
        self.bm25_dir: Path = Path(settings.bm25_index_dir)
        self.bm25_path: Path = self.bm25_dir / "bm25_retriever.pkl"
        
        # Ensure base directories exist to avoid FileNotFoundError during save
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Embedding Model Provider
        if embedding_model:
            self.embedding_model = embedding_model
        elif OPENAI_API_KEY and 'text-embedding' in EMBEDDING_MODEL_NAME:
            logger.info(f"Using OpenAI Embeddings: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = OpenAIEmbeddings(
                model=EMBEDDING_MODEL_NAME, 
                openai_api_key=OPENAI_API_KEY
            )
        else:
            logger.info(f"Using HuggingFace Embeddings: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

    def create_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Builds and persists both FAISS and BM25 indices from document chunks.

        This method transforms raw dictionary chunks into LangChain Document 
        objects, injecting 'unit_id' and 'unit_title' into the text to boost 
        keyword search relevance for legal citations.

        Args:
            chunks: A list of dictionaries containing 'text' and 'metadata'.

        Raises:
            EmbeddingModelError: If index creation or embedding generation fails.
        """
        if (self.faiss_dir / "index.faiss").exists():
            logger.info("Existing index detected. Overwriting with fresh data...")

        try:
            if not chunks:
                raise ValueError("No chunks provided for indexing.")

            # Convert chunks to LangChain 'Document' objects
            documents: List[Document] = []
            for chunk in chunks:
                meta = chunk.get('metadata', {})
                # ENHANCEMENT: Prepend legal identifiers to text for better BM25 matching
                enhanced_text = (
                    f"ID: {meta.get('unit_id', '')} "
                    f"Title: {meta.get('unit_title', '')}\n"
                    f"{chunk['text']}"
                )
                documents.append(Document(page_content=enhanced_text, metadata=meta))

            # 1. Build and Persist FAISS (Dense/Semantic Search)
            logger.info("Generating embeddings and building FAISS index...")
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.vector_store.save_local(str(self.faiss_dir))

            # 2. Build and Persist BM25 (Sparse/Keyword Search)
            logger.info("Building BM25 keyword index...")
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            
            with open(self.bm25_path, "wb") as file:
                pickle.dump(self.bm25_retriever, file)

            logger.info(f"Hybrid Store successfully built with {len(documents)} documents.")

        except Exception as e:
            logger.exception("Hybrid store creation failed.")
            raise EmbeddingModelError(f"Failed to index documents: {str(e)}")

    def load_store(self) -> None:
        """
        Loads the FAISS and BM25 indices from disk into memory.

        Raises:
            VectorStoreNotInitializedError: If the FAISS index files are missing.
            DocumentLoadError: If the BM25 pickle file is missing or corrupted.
        """
        try:
            # Load FAISS Dense Index
            faiss_file = self.faiss_dir / "index.faiss"
            if not faiss_file.exists():
                raise VectorStoreNotInitializedError(f"FAISS index missing at {self.faiss_dir}")
                
            self.vector_store = FAISS.load_local(
                str(self.faiss_dir), 
                self.embedding_model,
                allow_dangerous_deserialization=True  # Required for loading pickles in FAISS
            )
            
            # Load BM25 Sparse Index
            if not self.bm25_path.exists():
                raise DocumentLoadError(f"BM25 index missing at {self.bm25_path}")
                
            with open(self.bm25_path, "rb") as file:
                self.bm25_retriever = pickle.load(file)

            logger.info("Hybrid search indices successfully loaded into memory.")

        except (VectorStoreNotInitializedError, DocumentLoadError) as e:
            # Re-raise specific exceptions
            raise e
        except Exception as e:
            logger.exception("Unexpected failure during store loading.")
            raise DocumentLoadError(f"Critical failure while loading search indices: {e}")

    def get_ensemble_retriever(
        self, 
        top_k_dense: int = 5, 
        top_k_sparse: int = 5, 
        weight_dense: float = 0.5
    ) -> EnsembleRetriever:
        """
        Combines Dense and Sparse retrievers using Reciprocal Rank Fusion (RRF).

        Args:
            top_k_dense: Number of documents to retrieve from FAISS.
            top_k_sparse: Number of documents to retrieve from BM25.
            weight_dense: The importance (0.0 to 1.0) of the dense results. 
                The sparse weight is calculated as (1.0 - weight_dense).

        Returns:
            EnsembleRetriever: A hybrid retriever object.

        Raises:
            VectorStoreNotInitializedError: If indices are not loaded.
        """
        if not self.vector_store or not self.bm25_retriever:
            logger.error("Retrieval attempted on uninitialized HybridStore.")
            raise VectorStoreNotInitializedError("Must call load_store() or create_store() before retrieving.")

        try:
            # Configure individual retriever parameters
            faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k_dense})
            self.bm25_retriever.k = top_k_sparse
            
            weight_sparse = round(1.0 - weight_dense, 2)
            
            logger.debug(f"Creating Ensemble: Dense weight {weight_dense}, Sparse weight {weight_sparse}")
            
            return EnsembleRetriever(
                retrievers=[self.bm25_retriever, faiss_retriever],
                weights=[weight_sparse, weight_dense]
            )
        except Exception as e:
            logger.error(f"Failed to construct EnsembleRetriever: {e}")
            raise RAGBaseException(f"Error initializing ensemble retrieval logic: {e}")