"""
Retrieval-Augmented Generation (RAG) Pipeline

This module defines the main RAGPipeline class used to:
- Retrieve relevant documents using FAISS
- Build contextual prompts
- Stream LLM outputs token-by-token using HuggingFace Inference API
- Provide a non-streaming fallback generation interface

The pipeline is optimized for:
- Streamlit Cloud deployment (CPU-only)
- Stateless inference
- Safe document handling in session state
"""

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

# dotenv_path = Path(__file__).resolve().parents[2] / ".env"
# load_dotenv(dotenv_path)
# HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "").strip('"').strip("'")

HF_API_TOKEN = st.secrets.get("HUGGINGFACE_API_KEY")
if not HF_API_TOKEN:
    raise ValueError("HuggingFace API token not found in environment variables!")
else:
    logger.info("API Key successfully loaded")

HF_REPO_ID = settings.hf_repo_id
TEMPERATURE = settings.llm_temperature
THRESHOLD = settings.threshold
TOP_K = settings.top_k
MAX_TOKENS = settings.max_tokens


class RAGPipeline:
    """
    Main class implementing the Retrieval-Augmented Generation pipeline.

    This pipeline performs:
    - Retrieval from a FAISS vector store
    - Prompt construction with contextual documents
    - Token-by-token generation via HuggingFace Inference API (streaming)
    - Non-streaming generation for batch use cases

    Parameters
    ----------
    vector_store : FaissStore, optional
        Preloaded FAISS vector store. If None, a new store is created.
    temperature : float, optional
        Sampling temperature for the LLM. Default is from settings.
    streaming : bool, optional
        Whether to enable streaming mode for token output.
    """

    def __init__(self, vector_store=None, temperature=TEMPERATURE, streaming=True):
        self.vector_store = vector_store or FaissStore()
        self.temperature = temperature
        self.streaming = streaming

        # Initialize HuggingFace Inference client
        self.client = InferenceClient(api_key=HF_API_TOKEN)
        self.model_id = HF_REPO_ID

        logger.info("RAGPipeline initialized with HF InferenceClient.")

    # -----------------------------------------------------------
    # Retrieve + Filter
    # -----------------------------------------------------------
    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        """
        Retrieve the top-k documents most relevant to the query from FAISS.

        Retrieval includes an adaptive filtering mechanism based on:
        - Maximum score
        - Dynamic threshold using mean + variance
        - Minimum relevance cutoff

        Parameters
        ----------
        query : str
            User input or question.
        k : int, optional
            Number of top documents to consider. Uses global TOP_K by default.

        Returns
        -------
        List[Document]
            A list of filtered relevant documents. May return an empty list.
        """
        if not self.vector_store.is_loaded:
            logger.warning("Vector store empty — returning no documents.")
            return []

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

        # top_score = max(scores)
        # logger.info(f"Best cosine similarity score: {top_score:.4f}")

        cutoff = max(THRESHOLD, scores.mean() + 0.5 * scores.std())
        filtered = [doc for doc, score in results if score >= cutoff]

        return filtered if filtered else [docs[0]]

    # -----------------------------------------------------------
    # Prompt Builder
    # -----------------------------------------------------------
    def build_prompt(self, query: str, docs: List[Document]) -> str:
        """
        Construct a prompt that includes retrieved context followed by the user query.

        Parameters
        ----------
        query : str
            The user question.
        docs : List[Document]
            List of documents selected by the retriever.

        Returns
        -------
        str
            The generated prompt string ready for LLM inference.
        """
        sys_content = """
        You are a polite senior legal expert specializing in Australian privacy law.
        Your behaviour rules:
        - Begin every answer with: "Thanks for asking!"
        - DO NOT start answers with phrases like "Based on the context" or "The provided context says..."
        - DO NOT mention that you used retrieved documents.
        - Provide clear, concise legal explanations.
        - If the user question is irrelevant to privacy law, say so politely.
        - If the documents do not provide enough information, say so honestly.
        - Always complete your last sentence and do not stop mid-thought.
        """

        # ASSISTANT MESSAGE — contains RAG-retrieved factual material
        if docs:
            context = []
            context = "\n\n".join([d.page_content for d in docs if docs])
            context_message = {
                "role": "assistant",
                "content": f"Here is helpful reference material:\n\n{context}"
            }
        else:
            context_message = {
                "role": "assistant",
                "content": "No reference material was found for this question."
            }

        # USER MESSAGE — contains only the user's question (no formatting!)
        user_message = {"role": "user", "content": query}

        return [
            {"role": "system", "content": sys_content},
            context_message,
            user_message,
        ]

    # -----------------------------------------------------------
    # Streaming Generator (Token-by-token)
    # -----------------------------------------------------------
    def generate_streaming(self, query: str, top_k: int = 3, include_docs: bool = True):
        """
        Generate an answer using token streaming from the HF Inference API.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int, optional
            Number of retrieval documents.
        include_docs : bool, optional
            Whether the generator should emit retrieved docs as the final event.

        Yields
        ------
        dict
            Either {"token": "..."} for each generated token OR
            {"docs": [...]} after generation finishes.
        """
        try:
            docs = self.retrieve(query, k=top_k)
        except Exception as e:
            logger.exception(f"Error retrieving documents: {e}")
            docs = []

        messages = self.build_prompt(query, docs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=self.temperature,
                top_p=0.9,
                stream=True,
                # frequency_penalty=-0.10, # legal matters may require repetions, (+)ve otherwise
                # presence_penalty=-0.20, # keep on topic. for vareity set (+), i.e. 0.6
                # seed=42, # for reproducibility
                # extra_body={"repetition_penalty": 1.1, "top_k": 40} # extra arguments 
            )
        except Exception as e:
            logger.error(f"ChatCompletions streaming failed: {e}")
            yield {"token": "AI Error: Chat Completions API failed.\n"}
            return

        for event in response:
            try:
                if hasattr(event, "choices") and event.choices:
                    choice = event.choices[0]
                    delta = getattr(choice, "delta", None)

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
        """
        Execute non-streaming text generation using the HF Inference API.

        Parameters
        ----------
        query : str
            User query.
        top_k : int, optional
            Number of retrieval documents.

        Returns
        -------
        str
            The generated answer or an error message.
        """
        try:
            docs = self.retrieve(query, k=top_k)
        except Exception as e:
            logger.exception(f"Error retrieving docs: {e}")
            docs = []

        prompt = self.build_prompt(query, docs)

        try:
            result = self.client.text_generation(
                prompt,
                max_new_tokens=MAX_TOKENS,
                temperature=self.temperature,
            )
            return result
        except Exception as e:
            logger.exception(f"Non-streaming LLM generation error: {e}")
            return "An error occurred while generating the response."

    # -----------------------------------------------------------
    # Legacy user_interaction
    # -----------------------------------------------------------
    def user_interaction(self, user_query: str, top_k=TOP_K, return_context=True):
        """
        Legacy helper to retrieve context and produce a final answer
        in a single call.

        Parameters
        ----------
        user_query : str
            The user question.
        top_k : int, optional
            Number of documents to retrieve.
        return_context : bool, optional
            Whether to return the retrieved docs.

        Returns
        -------
        Tuple[str, List[Document]]
            The answer and optionally the documents.
        """
        docs = self.retrieve(user_query, k=top_k)
        answer = self.generate(user_query, top_k=top_k)

        return (answer, docs) if return_context else (answer, [])
