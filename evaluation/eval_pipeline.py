"""
RAGAS Evaluation Pipeline

Evaluates the Privacy Act 1988 RAG system using RAGAS metrics.

Workflow:
    1. Load evaluation dataset.
    2. Generate answers from the RAG pipeline.
    3. Collect retrieved contexts.
    4. Build a RAGAS-compatible dataset.
    5. Execute evaluation metrics.
    6. Export detailed results.

Run:
    python -m evaluation.eval_pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime

from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, AnswerRelevancy

from langchain_openai import OpenAIEmbeddings

from src.rag.config import settings
from src.rag.hybrid_store import HybridStore
from src.rag.logger import get_logger
from src.rag.rag_pipeline import RAGPipeline


# =============================================================================
# Logging
# =============================================================================

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

OPENAI_API_KEY = settings.openai_api_key

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"

MAX_TOKENS = 2048
TEMPERATURE = 0

CURRENT_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = CURRENT_DIR / "datasets/eval_data_v1.json"

OUTPUT_PATH = Path("evaluation/reports/eval_report.csv")


# =============================================================================
# Dataset Utilities
# =============================================================================

def load_eval_dataset() -> list[dict[str, Any]]:
    """
    Load evaluation samples from the JSON file.

    Returns:
        List of evaluation records.

    Raises:
        FileNotFoundError:
            If eval_set.json cannot be found.
    """
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {EVAL_SET_PATH}"
        )

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_questions_and_references(
    eval_data: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """
    Extract questions and reference answers from the dataset.

    Args:
        eval_data: Raw evaluation records.

    Returns:
        Tuple containing:
            - questions
            - reference answers
    """
    questions = [item["question"] for item in eval_data]
    references = [item["ground_truth"] for item in eval_data]

    return questions, references


# =============================================================================
# RAG Pipeline Utilities
# =============================================================================

def build_rag_chain():
    """
    Build and initialize the RAG chain.

    Returns:
        Configured LangChain runnable chain.
    """
    store = HybridStore()
    store.load_store()

    rag_pipeline = RAGPipeline(hybrid_store=store)

    return rag_pipeline.build_chain()


def generate_responses(
    chain,
    questions: list[str]
) -> tuple[list[str], list[list[str]]]:
    """
    Generate answers and retrieved contexts.

    Args:
        chain:
            RAG chain.

        questions:
            Evaluation questions.

    Returns:
        Tuple containing:
            - generated answers
            - retrieved contexts
    """
    answers: list[str] = []
    contexts: list[list[str]] = []

    logger.info(
        "Executing RAG pipeline to generate responses..."
    )

    for idx, question in enumerate(questions):
        try:
            output = chain.invoke(
                {"question": question}
            )

            answer = output.get("answer", "")
            docs = output.get("docs", []) or []

            if not answer or not docs:
                logger.warning(
                    f"Missing answer/context for question {idx}"
                )

            answers.append(answer)

            contexts.append(
                [doc.page_content for doc in docs]
            )

        except Exception as exc:
            logger.error(
                f"Pipeline failed on question {idx}: {exc}"
            )

            answers.append("ERROR")
            contexts.append([])

    return answers, contexts


# =============================================================================
# RAGAS Dataset Construction
# =============================================================================

def build_ragas_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    references: list[str]
) -> Dataset:
    """
    Build a RAGAS-compatible dataset.

    Args:
        questions:
            User questions.

        answers:
            Generated answers.

        contexts:
            Retrieved document contexts.

        references:
            Ground truth answers.

    Returns:
        Hugging Face Dataset object.
    """
    return Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": references,
        }
    )


# =============================================================================
# Evaluation Configuration
# =============================================================================

def create_llm():
    """
    Create evaluator LLM instance.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    return llm_factory(
        model=LLM_MODEL,
        client=client,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


def create_embeddings():
    """
    Create embedding model instance.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


# =============================================================================
# Evaluation Execution
# =============================================================================

def run_evaluation(dataset: Dataset):
    """
    Execute RAGAS evaluation.

    Args:
        dataset:
            RAGAS evaluation dataset.

    Returns:
        RAGAS evaluation results.
    """
    llm = create_llm()
    embeddings = create_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        ),
    ]

    logger.info("Starting RAGAS evaluation...")

    logger.info(
        f"LLM Model: {LLM_MODEL} | "
        f"Embedding Model: {EMBEDDING_MODEL}"
    )

    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )


# =============================================================================
# Reporting
# =============================================================================
        
def save_results(results) -> None:
    """
    Save evaluation results to CSV and log summary metrics.

    Args:
        results:
            RAGAS evaluation results.
    """
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = results.to_pandas()

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"Evaluation completed. "
        f"Report saved at {OUTPUT_PATH}"
    )

    try:
        numeric_cols = df.select_dtypes(
            include="number"
        )

        logger.info("Evaluation Summary")

        for col in numeric_cols.columns:
            logger.info(
                f"{col}: "
                f"{numeric_cols[col].mean():.4f}"
            )

    except Exception as exc:
        logger.warning(
            f"Unable to generate summary statistics: {exc}"
        )

def save_metadata(
    run_id: str,
    notes: str | None,
    dataset_ver: str = "v1",
    num_questions: int = 4 ,
    metrics: list[str] = ["Faithfulness", "AnswerRelevency"],
) -> Path:
    """
    Save evaluation run metadata as JSON.

    Returns:
        Path to saved metadata.json
    """
    if notes: 
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data_version": dataset_ver,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "framework": "ragas",
            "metrics": metrics,
            "num_questions": num_questions,
            "notes": notes,
        }
    else:
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data_version": dataset_ver,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "framework": "ragas",
            "metrics": metrics,
            "num_questions": num_questions,
        }
    
    metadata_dir = Path("evaluation/experiments")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    file_path = metadata_dir / f"metadata_{dataset_ver}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """
    Execute the complete evaluation workflow.
    """
    eval_data = load_eval_dataset()

    questions, references = (
        extract_questions_and_references(
            eval_data
        )
    )

    logger.info(
        f"Loaded {len(questions)} questions for evaluation."
    )

    chain = build_rag_chain()

    answers, contexts = generate_responses(
        chain=chain,
        questions=questions,
    )

    dataset = build_ragas_dataset(
        questions=questions,
        answers=answers,
        contexts=contexts,
        references=references,
    )

    results = run_evaluation(dataset)

    save_results(results)

    save_metadata("run_2026_06_06_001", 
                  "This is the final run with dataset_v1.")


if __name__ == "__main__":
    main()