# Privacy Act 1988: High-Precision RAG Advisory System
```mermaid
graph TD
    %% Define the Ingestion Flow
    subgraph Ingestion_Pipeline [Ingestion Pipeline]
        Docs[PDF Document] --> Chunking[Section-wise Chunking]
        Chunking --> Embed[Embedding Model]
        Embed --> VDB[(FAISS + BM25 Indices)]
    end

    %% Define the Retrieval Flow
    User([User Query]) --> Rewrite[Query Expansion / Rewriting]
    Rewrite --> Retrieval{Hybrid Search (Ensemble Retrieval)}
    
    VDB <--> Retrieval
    
    Retrieval --> Rerank[Cross-Encoder Reranker]
    Rerank --> Context[Top-K Refined Context]
    
    %% Define the Generation Flow
    Context --> LLM[LLM Generator]
    LLM --> Out([Final Response + Citations])

    %% Professional Styling
    style VDB fill:#16161e,stroke:#333,stroke-width:2px
    style Retrieval fill:#24283b,stroke:#d4a017,stroke-width:2px
    style LLM fill:#24283b,stroke:#01579b,stroke-width:2px
```
An enterprise-grade Retrieval-Augmented Generation (RAG) system specialized in the **Australian Privacy Act 1988**. This system utilizes a **Two-Stage Retrieval** architecture (Bi-Encoder + Cross-Encoder) to provide cited, grounded, and legally-aligned advisory responses.

## 🎯 Engineering Philosophy
Legal documents present unique challenges for standard RAG pipelines: complex hierarchies, interdependent clauses, and the high cost of "hallucinated" advice. This project addresses these via:
* **Structural Integrity:** Legislative-aware chunking that preserves context.
* **Hybrid Search:** Merging semantic (FAISS) and keyword (BM25) search to capture both intent and specific citations.
* **Precision Funneling:** A Cross-Encoder reranking stage to filter out low-confidence context before LLM generation.

## 🛠 Technical Deep Dive
1. Domain-Specific Ingestion (The Chunker)
<p> Unlike naive character splitters, the DocumentChunker utilizes a Hierarchical Regex Strategy to mirror the Act's structure:

Segmentation: Distinguishes between the Main Act and Schedule 1 (Australian Privacy Principles) and other Schedules.

Unit Detection: Locates Sections, APPs, Clauses, and Subsections to ensure embeddings contain complete legal thoughts.

Metadata Injection: Every chunk is enriched with its unit_id and unit_title, enabling the BM25 retriever to provide citations accurately.
</p>
2. Two-Stage Hybrid Retrieval
To solve the "Needle in a Haystack" problem, the system employs a two-stage funnel:

Recall Stage (Hybrid Search): Uses Reciprocal Rank Fusion (RRF) to combine dense vectors from FAISS with sparse keyword scores from BM25.

Precision Stage (Reranking): Uses a Cross-Encoder (ms-marco-MiniLM) to perform a computationally expensive but highly accurate relevance check on the top candidates.

3. Grounded Generation & Guardrails
The LLM is governed by a Modular System Prompt using Markdown delimitation (###) for clear instruction-following. It implements a "Silence over Falsehood" policy:

Confidence Thresholding: If the Reranker returns scores below a specific threshold, the pipeline triggers a "Safe Refusal" rather than guessing an answer.

Strict Grounding: The model is prohibited from using internal knowledge, forcing it to cite the provided legal context.

## 🚀 Deployment & AIOps
* **Optimized Resource Management:** Uses Streamlit's @st.cache_resource to manage memory-intensive models (Reranker and Vector Store) on CPU-bound environments.

* **Observability:** Integrated structured logging captures pipeline latency and retrieval scores for performance auditing.

* **Robust Exception Handling:** A custom exception module ensures the UI fails gracefully and provides actionable logs for developers.

## 🏗 Modular Architecture

The codebase follows a modular design pattern, ensuring that the data ingestion, search indexing, and generation logic are decoupled and independently scalable.

```text
├── src/rag/
│   ├── chunker.py        # Legislative-aware PDF parser and hierarchical splitter
│   ├── hybrid_store.py   # Dual-index management (FAISS & BM25) with RRF
│   ├── rag_pipeline.py   # LCEL orchestration, Reranking, and Grounded Generation
│   ├── exception.py     # Domain-specific custome error handling & hierarchy for AIOps
│   ├── config.py         # Centralized environment and model settings
│   └── logger.py         # Structured telemetry and pipeline logging
├── hybrid_store/
│   ├── bm25_store
│   │    └── bm25_retriever.pkl     # Sparse indices (for keyword search)
│   └── faiss_index
│        ├── index.faiss            # Dense indices (for semantic search)
│        └── index.pkl
│
├── streamlit_app.py      # Streamlit UI with Recall/Precision depth controls
├── README.md             # Description of the project and AIOps steps followed
└── requirements.txt      # Project dependencies
```