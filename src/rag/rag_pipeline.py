"""
Upgraded RAG Pipeline
- Works with Streamlit Cloud (CPU)
- Uses HuggingFace Inference API Streaming
- Returns session-safe docs
- Clean retrieval + generation pipeline
"""

import os
from typing import List, Generator, Dict
import numpy as np
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain_core.documents import Document

from src.rag.faiss_store import FaissStore
from src.rag.logger import get_logger
from src.rag.exception import VectorStoreNotInitializedError, RetrievalError
from src.rag.config import settings
import streamlit as st

logger = get_logger(__name__)

# load_dotenv()
# HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "").strip('"').strip("'")
HF_API_TOKEN = st.secrets.get("HUGGINGFACE_API_KEY")
if not HF_API_TOKEN:
    raise ValueError("HuggingFace API token not found in environment variables!")
else:
    logger.info("API Key successfully loaded")
# Settings
HF_REPO_ID = settings.hfh_repo_id
TEMPERATURE = settings.llm_temperature
THRESHOLD = settings.threshold
TOP_K = settings.top_k


class RAGPipeline:

    def __init__(self, vector_store=None, temperature=TEMPERATURE, streaming=True):

        self.vector_store = vector_store or FaissStore()
        self.temperature = temperature
        self.streaming = streaming

        # HF Streaming LLM Client
        self.client = InferenceClient(api_key=HF_API_TOKEN)
        self.model_id = HF_REPO_ID

        logger.info("RAGPipeline initialized with HF InferenceClient.")

    # -----------------------------------------------------------
    # Retrieve + Filter
    # -----------------------------------------------------------
    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        # Empty FAISS store → no docs
        if not self.vector_store.is_loaded:
            logger.warning("VectorStoreNotInitializedError: FAISS store empty, returning no documents.")
            return []  # do NOT raise, keep graceful behavior for demo

        try:
            results = self.vector_store.similarity_search(query, k=k)
        except VectorStoreNotInitializedError:
            logger.warning("FAISS store not initialized — returning empty retrieval.")
            return []
        except RetrievalError as e:
            logger.error(f"RetrievalError: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected retrieval error: {e}")
            return []

        if not results:
            return []

        docs, scores = zip(*results)
        scores = np.array(scores)
        top_score = max(scores)
        logger.info(f"Best cosine similarity score: {top_score:.4f}")
        cutoff = max(THRESHOLD, scores.mean() + 0.5 * scores.std())
        
        filtered = [doc for doc, s in results if s >= cutoff]
        return filtered if filtered else [docs[0]]

    # -----------------------------------------------------------
    # Prompt Builder
    # -----------------------------------------------------------
    def build_prompt(self, query: str, docs: List[Document]) -> str:
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            return (
                "You are a helpful assistant.\n"
                "Use the provided context to answer.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\nAnswer:"
            )
        else:
            return f"You are a helpful assistant.\n\nQuestion: {query}\nAnswer:"

    # -----------------------------------------------------------
    # Streaming Generator (Token-by-token)
    # -----------------------------------------------------------
    def generate_streaming(self, query: str, top_k: int = 3, include_docs: bool = True):
        try:
            docs = self.retrieve(query, k=top_k)
        except Exception as e:
            logger.exception(f"Error retrieving documents: {e}")
            docs = []

        prompt = self.build_prompt(query, docs)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=self.temperature,
                stream=True,
            )
        except Exception as e:
            logger.error(f"ChatCompletions streaming failed: {e}")
            yield {"token": "AI Error: Chat Completions API failed.\n"}
            return

        # Stream individual tokens
        for event in response:
            try:
                # Make sure choices exist and is non-empty
                if hasattr(event, "choices") and event.choices:
                    choice = event.choices[0]
                    delta = getattr(choice, "delta", None)

                    # Sometimes delta itself can be None
                    if delta and getattr(delta, "content", None):
                        yield {"token": delta.content}

            except Exception as e:
                logger.exception(f"Streaming token parse error: {e}")

        if include_docs:
            yield {"docs": docs}

    # -----------------------------------------------------------
    # Non-streaming fallback (used internally / batch)
    # -----------------------------------------------------------
    def generate(self, query: str, top_k: int = 3) -> str:
        try:
            docs = self.retrieve(query, k=top_k)
        except Exception as e:
            logger.exception(f"Error retrieving docs: {e}")
            docs = []

        prompt = self.build_prompt(query, docs)

        try:
            result = self.client.text_generation(
                prompt,
                max_new_tokens=256,
                temperature=self.temperature)
            return result
        except Exception as e:
            logger.exception(f"Non-streaming LLM generation error: {e}")
            return "An error occurred while generating the response."

    # -----------------------------------------------------------
    # Legacy user_interaction
    # -----------------------------------------------------------
    def user_interaction(self, user_query: str, top_k=TOP_K, return_context=True):
        docs = self.retrieve(user_query, k=top_k)
        answer = self.generate(user_query, top_k=top_k)

        return (answer, docs) if return_context else (answer, [])
