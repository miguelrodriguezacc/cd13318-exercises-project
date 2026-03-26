#!/usr/bin/env python3
"""Batch evaluation runner for the NASA RAG system.

Reads a QA list from `evaluation_dataset.txt`, runs retrieval + response generation,
and evaluates the outputs using RAGAS (BLEU, relevancy, faithfulness, etc.).

Outputs results to a CSV file.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import rag_client
import llm_client
import ragas_evaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_evaluation_dataset(path: Path) -> List[Dict[str, str]]:
    """Load a simple QA dataset from a text file.

    Expects blocks like:
      Q - question text
      A - answer text
    """
    text = path.read_text(encoding="utf-8")
    items: List[Dict[str, str]] = []

    for block in [b.strip() for b in text.split("Q - ") if b.strip()]:
        if "A -" not in block:
            continue
        question, answer = block.split("A -", 1)
        items.append({"question": question.strip(), "answer": answer.strip()})

    return items


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate the RAG system using evaluation_dataset.txt")
    parser.add_argument("--dataset", default="evaluation_dataset.txt", help="Path to the evaluation dataset file")
    parser.add_argument("--output", default="evaluation_results.csv", help="CSV output path")
    parser.add_argument("--backend", default=None, help="ChromaDB backend key (see discover_chroma_backends)")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), required=False, help="OpenAI API key")
    parser.add_argument("--n-docs", type=int, default=10, help="Number of documents to retrieve for each query")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model for generation")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    args = parser.parse_args()
    
    if not args.openai_key:
        raise SystemExit("OpenAI API key is required (set OPENAI_API_KEY or pass --openai-key)")

    # Discover backends
    backends = rag_client.discover_chroma_backends()
    if not backends:
        raise SystemExit("No ChromaDB backends found. Run the embedding pipeline first.")

    # Choose backend
    backend_key = args.backend or next(iter(backends.keys()))
    if backend_key not in backends:
        raise SystemExit(f"Backend '{backend_key}' not found. Available keys: {list(backends.keys())}")
    backend = backends[backend_key]

    # Initialize RAG system using the same embedding model that created the vectors
    collection, ok, err = rag_client.initialize_rag_system(
        backend["directory"],
        backend["collection_name"],
        openai_api_key=args.openai_key,
        embedding_model=args.embedding_model,
    )
    if not ok:
        raise SystemExit(f"Failed to initialize RAG system: {err}")

    # Load dataset
    items = load_evaluation_dataset(Path(args.dataset))
    if not items:
        raise SystemExit(f"No QA pairs found in {args.dataset}")

    # Prepare output CSV
    fieldnames = [
        "question",
        "reference",
        "prediction",
        "bleu_score",
        "context_precision",
        "response_relevancy",
        "faithfulness",
        "rouge_score",
        "retrieved_docs",
    ]

    total_metrics = {
        "bleu_score": 0.0,
        "context_precision": 0.0,
        "response_relevancy": 0.0,
        "faithfulness": 0.0,
        "rouge_score": 0.0
    }
    count = 0

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            question = item["question"]
            reference = item["answer"]

            # Retrieve documents
            docs_result = rag_client.retrieve_documents(
                collection,
                question,
                n_results=args.n_docs,
            )

            documents = []
            metadatas = []
            if docs_result and docs_result.get("documents"):
                documents = docs_result["documents"][0]
                metadatas = docs_result.get("metadatas", [[]])[0]

            context = rag_client.format_context(documents, metadatas)

            # Generate response
            prediction = llm_client.generate_response(
                args.openai_key,
                question,
                context,
                [],
                model=args.model,
            )

            # Evaluate
            scores = ragas_evaluator.evaluate_response_quality(
                question,
                prediction,
                documents,
                reference=reference,
            )

            # Map known metric keys into stable columns
            def first_present(*keys):
                for k in keys:
                    if k in scores and scores[k] is not None:
                        return scores[k]
                return None
        
            def get_safe_score(*keys):
                for k in keys:
                    val = scores.get(k)
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            continue
                return 0.0

            # Extract metrics for this row, trying multiple possible keys for each metric to handle variations in RAGAS output
            row_metrics = {
                "bleu_score": get_safe_score("bleu_score", "BLEU", "bleu"),
                "context_precision": get_safe_score("non_llm_context_precision_with_reference", "context_precision"),
                "response_relevancy": get_safe_score("response_relevancy", "response_relevancy_score"),
                "faithfulness": get_safe_score("faithfulness", "faithfulness_score"),
                "rouge_score": get_safe_score("rouge_score", "rouge_l", "rouge_1")
            }

            # Totalize metrics for average calculation
            for key in total_metrics:
                total_metrics[key] += row_metrics[key]
            count += 1

            writer.writerow({
                "question": question,
                "reference": reference,
                "prediction": prediction,
                **row_metrics,
                "retrieved_docs": len(documents) if documents else 0,
            })

        if count > 0:
            averages = {k: v / count for k, v in total_metrics.items()}
            
            # Escribir una fila vacía y luego los promedios al final del CSV
            writer.writerow({k: "" for k in fieldnames}) 
            writer.writerow({
                "question": "TOTAL AVERAGE",
                "reference": "-",
                "prediction": "-",
                **averages,
                "retrieved_docs": "-"
            })

            print("\n=== Resumen de Métricas Totales ===")
            for metric, value in averages.items():
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    print(f"\nEvaluation complete. Results written to {args.output}")


if __name__ == "__main__":
    main()
