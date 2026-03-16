import os

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openai.vocareum.com/v1" if os.getenv("OPENAI_API_KEY", "").startswith("voc") else None
            )
        )
    # TODO: Create evaluator_embeddings with model test-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openai.vocareum.com/v1" if os.getenv("OPENAI_API_KEY", "").startswith("voc") else None
            )
        )
    # TODO: Define an instance for each metric to evaluate
    bleu = BleuScore()
    context_precision = NonLLMContextPrecisionWithReference()
    response_relevancy = ResponseRelevancy(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm)
    rouge = RougeScore()

    # Build an EvaluationDataset from a single sample
    dataset = EvaluationDataset.from_list([
        {
            "user_input": question,
            "response": answer,
            "reference": answer,
            "retrieved_contexts": contexts,            
            "reference_contexts": contexts,        }
    ])

    # Evaluate the response using the metrics
    result = evaluate(
        dataset,
        metrics=[bleu, context_precision, response_relevancy, faithfulness, rouge],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # Return the evaluation results
    return result.to_pandas().iloc[0].to_dict()

