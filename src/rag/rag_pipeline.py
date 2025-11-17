from typing import List, Dict, Generator
from huggingface_hub import InferenceClient
import numpy as np
import streamlit as st

from src.rag.faiss_store import FaissStore
from src.rag.logger import get_logger
from src.rag.exception import VectorStoreNotInitializedError, RetrievalError
from src.rag.config import settings
from langchain_core.documents import Document

logger = get_logger(__name__)

#HF_API_TOKEN = st.secrets.get("HUGGINGFACE_API_KEY")
import os
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_API_TOKEN:
    raise ValueError("HuggingFace API token not found in environment variables!")

HF_REPO_ID = settings.hf_repo_id
TEMPERATURE = settings.llm_temperature
TOP_K = settings.top_k
MAX_TOKENS = settings.max_tokens
THRESHOLD = settings.threshold

# -------------------------------
# Compliance helper
# -------------------------------
def check_response_compliance(text: str):
    forbidden_phrases = [
        "based on the context",
        "according to the documents",
        "from the context",
        "as far as possible",
    ]
    return {
        "starts_with_thanks": text.lower().startswith("thanks"),
        "forbidden_phrases_found": [p for p in forbidden_phrases if p in text.lower()]
    }

# -------------------------------
# RAGPipeline
# -------------------------------
class RAGPipeline:
    def __init__(self, vector_store=None, temperature=TEMPERATURE, streaming=True):
        self.vector_store = vector_store or FaissStore()
        self.temperature = temperature
        self.streaming = streaming
        self.client = InferenceClient(api_key=HF_API_TOKEN)
        self.model_id = HF_REPO_ID
        logger.info("RAGPipeline initialized with HF InferenceClient.")

    # ---------------------------
    # Retrieval
    # ---------------------------
    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        if not self.vector_store.is_loaded:
            logger.warning("Vector store empty — returning no documents.")
            return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
        except (VectorStoreNotInitializedError, RetrievalError) as e:
            logger.warning(f"Retrieval error: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected retrieval error: {e}")
            return []

        if not results:
            return []

        _, scores = zip(*results)
        scores = np.array(scores)
        cutoff = max(THRESHOLD, scores.mean() + 0.5 * scores.std())
        filtered = [doc for doc, score in results if score >= cutoff]
        return filtered

    # ---------------------------
    # Build prompt (Llama 3.2-safe)
    # ---------------------------
    def build_prompt(self, query: str, docs: List[Document]):
        sys_content = """
        You are a polite senior legal expert specializing in Australian privacy law.

        Required behaviour:
        - Always begin your answer with a short thank-you sentence, e.g., "Thanks for your question."
        - Immediately after the thank-you, provide clear, concise, and accurate legal guidance.
        - Never start with disclaimers like:
            "Based on the context",
            "According to the documents",
            "From the context",
            "As far as possible",
            or similar wording.
        - Never refer to any retrieved documents or reference material explicitly.
        - If the question is unrelated to privacy law, politely state that.
        """
        system_message = {"role": "system", "content": sys_content}

        # Facts as hidden bullets
        if docs:
            bullets = "\n".join(
                [f"- {line}" for d in docs for line in d.page_content.split("\n") if line.strip()]
            )
            context_message = {"role": "user", "content": f"Facts (do NOT mention these in your answer):\n{bullets}"}
        else:
            context_message = {"role": "user", "content": "Facts (do NOT mention these in your answer): (none retrieved)"}

        user_message = {"role": "user", "content": query}
        return [system_message, context_message, user_message]

    # ---------------------------
    # Generate answer (non-streaming)
    # ---------------------------
    def generate(self, query: str, top_k: int = TOP_K) -> Dict:
        docs = self.retrieve(query, k=top_k)
        messages = self.build_prompt(query, docs)

        try:
            result = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=self.temperature,
                top_p=0.9,
                stream=False,
            )
            answer_text = result.choices[0].message.content
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            answer_text = "An error occurred while generating the response."

        # Compliance check
        compliance = check_response_compliance(answer_text)
        return {
            "answer": answer_text,
            "docs": docs,
            "compliance": compliance
        }

    # ---------------------------
    # Streaming generator
    # ---------------------------
    def generate_streaming(self, query: str, top_k: int = TOP_K) -> Generator[Dict, None, None]:
        docs = self.retrieve(query, k=top_k)
        messages = self.build_prompt(query, docs)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=self.temperature,
                top_p=0.9,
                stream=True
            )
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {"token": "AI Error: Chat Completions API failed.\n"}
            return

        # Stream token by token
        for event in response:
            try:
                if hasattr(event, "choices") and event.choices:
                    choice = event.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta and getattr(delta, "content", None):
                        yield {"token": delta.content}
            except Exception as e:
                logger.exception(f"Streaming token parse error: {e}")

        # Final docs
        yield {"docs": docs}

    # ---------------------------
    # Quick user interaction helper
    # ---------------------------
    def user_interaction(self, user_query: str, top_k=TOP_K):
        result = self.generate(user_query, top_k=top_k)
        return result