import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import faiss

# Custom modules
from src.rag.logger import get_logger
from src.rag.exception import (
    VectorStoreNotInitializedError,
    DocumentLoadError,
    EmbeddingModelError,
    RetrievalError
)
from .config import settings

logger = get_logger(__name__)

MODEL_NAME = settings.embedding_model_name
DEFAULT_STORE_DIR = settings.faiss_index_dir


class FaissStore:
    """
    Wrapper around LangChain's FAISS vector store.
    Handles index creation, saving, loading, and similarity search.
    """

    def __init__(self, embedding_model: Optional[HuggingFaceEmbeddings] = None):
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name=MODEL_NAME)
        self.store_dir = DEFAULT_STORE_DIR
        self.vector_store: Optional[FAISS] = None
        os.makedirs(self.store_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self.vector_store is not None

    def create_store(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("No documents provided for FAISS index creation.")

        try:
            logger.info(f"Creating FAISS store with {len(documents)} documents...")

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # FAISS cosine similarity: normalize vectors
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
                normalize_L2=True
            )

            self.vector_store.save_local(self.store_dir)

            logger.info(
                f"FAISS store successfully created and saved at: {self.store_dir} "
                f"with {self.vector_store.index.ntotal} vectors."
            )
            return self.vector_store

        except Exception as e:
            logger.exception("Error while creating FAISS store.")
            raise EmbeddingModelError(f"Failed to create FAISS index: {str(e)}") from e

    def load_store(self) -> FAISS:
        try:
            self.vector_store = FAISS.load_local(
                self.store_dir,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS store loaded from {self.store_dir} with {self.vector_store.index.ntotal} vectors.")
            return self.vector_store

        except FileNotFoundError as e:
            logger.error(f"FAISS store not found at {self.store_dir}.")
            raise DocumentLoadError(f"FAISS index not found at path: {self.store_dir}") from e
        except Exception as e:
            logger.exception("Error while loading FAISS store.")
            raise DocumentLoadError(f"Failed to load FAISS store: {str(e)}") from e

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if not self.is_loaded:
            logger.error("Attempted similarity search before initializing FAISS store.")
            raise VectorStoreNotInitializedError()

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.exception("Error during similarity search.")
            raise RetrievalError(f"Failed similarity search for query '{query}': {str(e)}") from e
