"""
Retrieval-Augmented Generation (RAG) Pipeline

This module defines the main RAGPipeline class used to:
- Retrieve relevant documents using FAISS
- Build contextual prompts
- Stream LLM outputs token-by-token using HuggingFace Inference API
- Provide a non-streaming fallback generation interface
"""

from typing import List, Dict, Any
# langchain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# local imports
from src.rag.hybrid_store import HybridStore
from src.rag.config import settings
from src.rag.logger import get_logger
from src.rag.exception import RetrievalError, RAGBaseException

# Setup logging
logger = get_logger(__name__)

# Constants from config
OPENAI_API_KEY = settings.openai_api_key

class RAGPipeline:
    """
    Orchestrates the RAG lifecycle: Retrieval, Reranking, and Generation.
    
    This class connects the HybridStore with a Cross-Encoder reranker and 
    a GPT-based LLM to provide high-precision answers based on the 
    Privacy Act 1988.
    """
    def __init__(self, hybrid_store: HybridStore):
        """
        Initializes the pipeline with a data store, a reranker, and an LLM.

        Args:
            hybrid_store (HybridStore): The initialized vector and keyword store.
        """
        self.store = hybrid_store
        
        # Initialize the Reranker Model (Cross-Encoder)
        # Higher precision than Bi-Encoders as it scores query-doc pairs directly
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder: {e}")
            raise RAGBaseException("Reranker model initialization failed.")
        
        # Initialize LLM with low temperature for factual consistency
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1, 
            api_key=OPENAI_API_KEY,
            max_tokens=1000,
        )

    def rerank_documents(self, query: str, docs: List[Document], 
                         top_n: int = 5) -> List[Document]:
        """
        Performs semantic reranking on retrieved documents using a Cross-Encoder.

        Args:
            query (str): The user's search query.
            docs (List[Document]): Candidates from the initial hybrid retrieval.
            top_n (int): Number of documents to return after reranking.

        Returns:
            List[Document]: The most relevant documents sorted by relevance score.
        """
        if not docs:
            return []
        
        try:
            # Prepare pairs for CrossEncoder: [[query, doc_text1], [query, doc_text2], ...]
            pairs = [[query, doc.page_content] for doc in docs]
            
            # Compute semantic relevance scores
            scores = self.reranker.predict(pairs)
            
            # Attach scores to metadata for observability and audit trails
            for doc, score in zip(docs, scores):
                doc.metadata['relevance_score'] = float(score)

            # Sort by score descending
            sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in sorted_docs[:top_n]]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return original docs if reranking fails
            return docs[:top_n]

    def build_chain(self):
        """
        Constructs the LangChain Expression Language (LCEL) chain.

        The chain follows a Parallel -> Assign -> Assign flow to ensure that 
        both the generated answer and the source documents are preserved in 
        the final output for transparency (citations).

        Returns:
            RunnableParallel: A reusable LCEL chain.
        
        Raises:
            RetrievalError: If the retriever cannot be constructed.
        """
        try:
            retriever = self.store.get_ensemble_retriever(weight_dense=0.6)
        except Exception as e:
            logger.error(f"Retriever construction failed: {e}")
            raise RetrievalError("Could not build the ensemble retriever.")

        def retrieve_and_rerank(input_data: dict) -> List[Document]:
            """Inner logic for two-stage retrieval."""
            try:
                query = input_data["question"]
                initial_docs = retriever.invoke(query)
                return self.rerank_documents(query, initial_docs, top_n=5)
            except Exception as e:
                logger.error(f"Retrieval/Rerank step failed: {e}")
                return []

        # Define specialized prompt for legal question answering
        template = """
        ### ROLE
        You are a high-precision Legal AI Assistant specializing in the Australian Privacy Act 1988. 

        ### CONTEXT
        {context}

        ### INSTRUCTIONS
        1. START with a thanking note (e.g., "Thanks for asking...").
        2. GROUNDING RULE: Answer ONLY using the provided context. 
        3. CITATION: You MUST refer to the specific Section, APP, or Clause number 
        for every fact you state.
        4. CONFIDENCE THRESHOLD: If the context does not contain enough detail to 
        answer the question accurately, or if the question is outside the scope of 
        the provided legal snippets, acknowledge this humbly. Say: "I'm sorry, but I 
        cannot find a definitive answer in the specific sections of the Privacy Act 
        currently available to me. Please consider rephrasing or asking a more relevant
        question or consulting the full Act."
        5. NO HALLUCINATION: Do not use outside knowledge or make assumptions 
        about legal interpretations not explicitly stated in the context.

        ### FORMAT
        - Concise, professional, and to the point.
        - Use bullet points for lists of requirements or principles.

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # The LCEL Pipeline
        try:
            chain = (
                RunnableParallel({
                    "docs": RunnableLambda(retrieve_and_rerank),
                    "question": lambda x: x["question"]
                })
                .assign(
                    context=lambda x: self._format_docs(x["docs"])
                )
                .assign(
                    answer=(prompt | self.llm | StrOutputParser())
                )
            )
            return chain
        except Exception as e:
            logger.critical(f"Failed to build LCEL chain: {e}")
            raise RAGBaseException("Internal error building the RAG chain.")

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        Flatten a list of documents into a single string for prompt injection.

        Args:
            docs (List[Document]): List of retrieved/reranked documents.

        Returns:
            str: Formatted context string.
        """
        return "\n\n".join(doc.page_content for doc in docs)