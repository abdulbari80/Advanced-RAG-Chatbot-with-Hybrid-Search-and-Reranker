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

from typing import List
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
# Setup logging
logger = get_logger(__name__)
# Constants from config
OPENAI_API_KEY = settings.openai_api_key

class RAGPipeline:
    """
    Constructs RAG pipeline to connect hybrid store and generate responses.
    """
    def __init__(self, hybrid_store: HybridStore):
        """
        Constructor for initializing the hybrid store, reranker, and LLM.
        """
        self.store = hybrid_store
        
        # Initialize the Reranker Model (Cross-Encoder)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano", 
            temperature=0.1, 
            api_key=OPENAI_API_KEY,
            max_tokens=1000,
        )

    def rerank_documents(self, query: str, docs: List[Document], 
                         top_n: int = 5) -> List[Document]:
        """
        Re-ranks among the already retrieved documents for high relevance.
        """
        if not docs:
            return []
        
        # Prepare pairs for CrossEncoder: [[query, doc_text1], [query, doc_text2], ...]
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Get scores
        scores = self.reranker.predict(pairs)
        
        # Attach scores to documents for debugging/logging
        for doc, score in zip(docs, scores):
            doc.metadata['relevance_score'] = float(score)

        # Sort docs by score in descending order
        sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top_n documents
        top_docs = [doc for doc, _ in sorted_docs[:top_n]]
        return top_docs

    def build_chain(self):
        """
        Constructs the LCEL chain. 
        Note: We removed 'query' as a parameter here so the chain stays reusable.
        """
        retriever = self.store.get_ensemble_retriever(weight_dense=0.6)

        # 1. Retrieval + Rerank function
        def retrieve_and_rerank(input_data: dict) -> List[Document]:
            query = input_data["question"]
            initial_docs = retriever.invoke(query)
            return self.rerank_documents(query, initial_docs, top_n=5)

        # 2. Prompt Template
        template = """Start an answer with a thanking note, such as Thanks for asking 
        or in a similar manner. An answer should be based only on the following context:
        {context}
        However, avoid using disclaimer or similar phrases, such as based on the given 
        document or retrieved context or similar phrases. Again, if no context is 
        available, acknowledge that humbly. Do not attempt to answer without context. 
        Instead, politely ask to rephrase the question.
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 3. The Re-structured Chain
        # We use RunnableParallel to keep 'docs' available in the final output
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

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)