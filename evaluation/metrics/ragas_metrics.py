"""
Central registry for RAGAS evaluation metrics.

This ensures:
- consistency across experiments
- easy extension of metrics
- clean separation from evaluation logic
"""

from ragas.metrics import Faithfulness, AnswerRelevancy


def get_default_metrics(llm, embeddings):
    """
    Default metric set for RAG evaluation.

    Args:
        llm:
            RAGAS judge LLM

        embeddings:
            Embedding model for similarity-based metrics

    Returns:
        List of initialized RAGAS metrics
    """

    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        ),
    ]