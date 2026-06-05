import os
import json
from pathlib import Path

from datasets import Dataset

from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy

from src.rag.logger import get_logger
from src.rag.hybrid_store import HybridStore
from src.rag.rag_pipeline import RAGPipeline 
from src.rag.config import settings

# --------------------
# Setup logging
# --------------------
logger = get_logger(__name__)

# Constants from config
OPENAI_API_KEY = settings.openai_api_key

# --------------------
# Paths
# --------------------
CURRENT_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = CURRENT_DIR / "eval_set.json"


# --------------------
# Load dataset
# --------------------
def load_eval_dataset():
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {EVAL_SET_PATH}"
        )

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------
# Main evaluation
# --------------------
def get_ragas_eval():
    eval_data = load_eval_dataset()
    questions = eval_data["questions"]
    ground_truth = eval_data["answers"]
    
    logger.info(f"Loaded {len(questions)} questions for evaluation.")

    # --------------------
    # Load RAG system
    # --------------------
    store = HybridStore()
    store.load_store()
    rag_pipeline = RAGPipeline(hybrid_store=store)
    chain = rag_pipeline.build_chain()

    answers = []
    contexts = []

    # --------------------
    # Run RAG pipeline
    # --------------------
    logger.info("Executing RAG pipeline to generate responses...")
    for i, q in enumerate(questions):
        try:
            output = chain.invoke({"question": q})
            ans = output.get("answer", "")
            docs = output.get("docs", []) or []
            
            if not ans or not docs:
                logger.warning(f"Empty answer or missing context for question {i}: {q}")
            
            answers.append(ans)
            contexts.append([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Pipeline failed on question {i}: {e}")
            answers.append("ERROR")
            contexts.append([])

    # --------------------
    # Build RAGAS dataset (STRICT format)
    # --------------------
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truth
    })

    # --------------------
    # Configuration
    # --------------------
    client = OpenAI(api_key=OPENAI_API_KEY)
    llm = llm_factory(model="gpt-4o-mini", client=client)
    
    # FIX: Use consistent config variable instead of os.getenv
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", 
        client=client
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings)
    ]

    # --------------------
    # Evaluate
    # --------------------
    logger.info("Starting Ragas evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    # --------------------
    # Save results
    # --------------------
    output_path = Path("evaluation/eval_report.csv")
    
    # FIX: Ensure directory exists before attempting to save
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    
    results.to_pandas().to_csv(output_path, index=False)
    logger.info(f"Evaluation completed. Report saved at {output_path}")


# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    get_ragas_eval() # run this module: python -m evaluation.eval_pipeline