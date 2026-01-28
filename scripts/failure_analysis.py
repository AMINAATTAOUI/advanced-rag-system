#!/usr/bin/env python3
"""
Failure Analysis for RAG System

Analyzes retrieval and generation failures to identify systematic issues:
- Missing context in retrieval
- Hallucinations in generation
- Poor performance on technical questions
- Out-of-scope handling

Usage:
    python scripts/failure_analysis.py --threshold 0.6
    python scripts/failure_analysis.py --results results/baseline_metrics_100pdfs.json
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Suppress ChromaDB verbose logging
logging.getLogger('chromadb').setLevel(logging.ERROR)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import Reranker
from src.generation.chain import RAGChain
from src.config import settings


class FailureAnalyzer:
    """Analyze systematic failures in RAG system."""
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialize failure analyzer.
        
        Args:
            threshold: Score threshold for considering a result as failure
        """
        self.threshold = threshold
        self.eval_data = self._load_eval_dataset()
        
        # Initialize RAG components
        self.vector_store = VectorStore()
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.load_index()  # Load the BM25 index
        self.reranker = Reranker()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            reranker=self.reranker
        )
        self.rag_chain = RAGChain()
        
        print(f"Failure Analyzer initialized (threshold={threshold})")
    
    def _load_eval_dataset(self) -> List[Dict]:
        """Load evaluation dataset."""
        eval_path = Path(settings.test_data_path) / 'eval_dataset.json'
        
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")
        
        with open(eval_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} evaluation queries")
        return data
    
    def analyze_retrieval_failures(self) -> Dict:
        """Analyze retrieval failures."""
        print("\n" + "="*80)
        print("RETRIEVAL FAILURE ANALYSIS")
        print("="*80)
        
        failures = {
            "missing_context": [],
            "low_relevance": [],
            "wrong_documents": [],
            "technical_queries": [],
            "total_queries": len(self.eval_data)
        }
        
        for item in self.eval_data:
            query = item.get("question", item.get("query", ""))
            expected_docs = item.get("relevant_docs", [])
            
            # Retrieve documents
            results = self.hybrid_retriever.retrieve(query, top_k=10, use_reranking=True)
            
            # Check if expected documents are in results
            retrieved_sources = set()
            for r in results:
                source = r.get("metadata", {}).get("source", "")
                if source:
                    # Extract filename from path
                    source_file = source.split("/")[-1].split("\\")[-1]
                    retrieved_sources.add(source_file)
            
            missing_docs = [doc for doc in expected_docs if doc not in retrieved_sources]
            
            if missing_docs and expected_docs:  # Only count if there are expected docs
                failures["missing_context"].append({
                    "query": query,
                    "missing_docs": missing_docs,
                    "retrieved_count": len(results),
                    "expected_count": len(expected_docs)
                })
            
            # Check relevance scores
            if results:
                avg_score = sum(r.get("score", 0) for r in results) / len(results)
                if avg_score < self.threshold:
                    failures["low_relevance"].append({
                        "query": query,
                        "avg_score": avg_score,
                        "top_score": results[0].get("score", 0) if results else 0
                    })
            
            # Identify technical queries (contain technical terms)
            technical_terms = ["algorithm", "model", "neural", "learning", "optimization", 
                             "architecture", "training", "inference", "embedding"]
            if any(term in query.lower() for term in technical_terms):
                # Calculate precision based on document retrieval
                found_docs = len([doc for doc in expected_docs if doc in retrieved_sources])
                precision = found_docs / len(expected_docs) if expected_docs else 0
                failures["technical_queries"].append({
                    "query": query,
                    "precision": precision,
                    "retrieved": len(results),
                    "expected": len(expected_docs)
                })
        
        return failures
    
    def analyze_generation_failures(self, num_queries: int = 20) -> Dict:
        """Analyze generation failures."""
        print("\n" + "="*80)
        print("GENERATION FAILURE ANALYSIS")
        print("="*80)
        
        failures = {
            "hallucinations": [],
            "irrelevant_answers": [],
            "incomplete_answers": [],
            "out_of_scope": [],
            "total_queries": min(num_queries, len(self.eval_data))
        }
        
        for i, item in enumerate(self.eval_data[:num_queries]):
            query = item.get("question", item.get("query", ""))
            expected_answer = item.get("expected_answer", "")
            
            # Generate answer
            try:
                response = self.rag_chain.query(query)
                answer = response.get("answer", "")
                context = response.get("context", [])
                
                # Check for hallucinations (answer contains info not in context)
                if context:
                    context_text = " ".join([c.get("content", "") for c in context])
                    # Simple heuristic: check if answer has content not in context
                    answer_words = set(answer.lower().split())
                    context_words = set(context_text.lower().split())
                    unique_words = answer_words - context_words
                    
                    if len(unique_words) > len(answer_words) * 0.3:  # >30% unique words
                        failures["hallucinations"].append({
                            "query": query,
                            "answer": answer[:200],
                            "unique_ratio": len(unique_words) / len(answer_words)
                        })
                
                # Check for "I don't know" or similar responses
                uncertain_phrases = ["i don't know", "i'm not sure", "cannot answer", 
                                   "no information", "not enough context"]
                if any(phrase in answer.lower() for phrase in uncertain_phrases):
                    failures["out_of_scope"].append({
                        "query": query,
                        "answer": answer[:200]
                    })
                
                # Check answer length (too short might be incomplete)
                if len(answer.split()) < 10:
                    failures["incomplete_answers"].append({
                        "query": query,
                        "answer": answer,
                        "word_count": len(answer.split())
                    })
                
            except Exception as e:
                print(f"Error generating answer for query {i}: {e}")
        
        return failures
    
    def print_retrieval_report(self, failures: Dict):
        """Print retrieval failure report."""
        print("\n" + "-"*80)
        print("RETRIEVAL FAILURES SUMMARY")
        print("-"*80)
        
        total = failures["total_queries"]
        
        print(f"\nMissing Context: {len(failures['missing_context'])}/{total} queries")
        if failures["missing_context"]:
            print("\nTop 5 examples:")
            for i, f in enumerate(failures["missing_context"][:5], 1):
                print(f"\n{i}. Query: {f['query'][:80]}...")
                print(f"   Missing: {len(f['missing_docs'])} documents")
                print(f"   Retrieved: {f['retrieved_count']}, Expected: {f['expected_count']}")
        
        print(f"\n\nLow Relevance Scores: {len(failures['low_relevance'])}/{total} queries")
        if failures["low_relevance"]:
            print("\nTop 5 examples:")
            for i, f in enumerate(failures["low_relevance"][:5], 1):
                print(f"\n{i}. Query: {f['query'][:80]}...")
                print(f"   Avg Score: {f['avg_score']:.3f}, Top Score: {f['top_score']:.3f}")
        
        print(f"\n\nTechnical Queries: {len(failures['technical_queries'])}/{total} queries")
        if failures["technical_queries"]:
            avg_precision = sum(f["precision"] for f in failures["technical_queries"]) / len(failures["technical_queries"])
            print(f"Average Precision on Technical Queries: {avg_precision:.3f}")
            
            print("\nWorst performing technical queries:")
            sorted_tech = sorted(failures["technical_queries"], key=lambda x: x["precision"])
            for i, f in enumerate(sorted_tech[:5], 1):
                print(f"\n{i}. Query: {f['query'][:80]}...")
                print(f"   Precision: {f['precision']:.3f}")
    
    def print_generation_report(self, failures: Dict):
        """Print generation failure report."""
        print("\n" + "-"*80)
        print("GENERATION FAILURES SUMMARY")
        print("-"*80)
        
        total = failures["total_queries"]
        
        print(f"\nHallucinations: {len(failures['hallucinations'])}/{total} queries")
        if failures["hallucinations"]:
            print("\nTop 3 examples:")
            for i, f in enumerate(failures["hallucinations"][:3], 1):
                print(f"\n{i}. Query: {f['query'][:80]}...")
                print(f"   Unique ratio: {f['unique_ratio']:.1%}")
                print(f"   Answer: {f['answer']}...")
        
        print(f"\n\nOut of Scope: {len(failures['out_of_scope'])}/{total} queries")
        if failures["out_of_scope"]:
            print("\nExamples:")
            for i, f in enumerate(failures["out_of_scope"][:3], 1):
                print(f"\n{i}. Query: {f['query'][:80]}...")
                print(f"   Answer: {f['answer']}...")
        
        print(f"\n\nIncomplete Answers: {len(failures['incomplete_answers'])}/{total} queries")
        if failures["incomplete_answers"]:
            print("\nExamples:")
            for i, f in enumerate(failures["incomplete_answers"][:3], 1):
                print(f"\n{i}. Query: {f['query'][:80]}...")
                print(f"   Word count: {f['word_count']}")
                print(f"   Answer: {f['answer']}")
    
    def generate_recommendations(self, retrieval_failures: Dict, generation_failures: Dict):
        """Generate recommendations based on failure analysis."""
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        # Retrieval recommendations
        missing_pct = len(retrieval_failures["missing_context"]) / retrieval_failures["total_queries"]
        if missing_pct > 0.3:
            recommendations.append({
                "issue": "High missing context rate",
                "severity": "HIGH",
                "suggestion": "Increase retrieval top_k or improve chunking strategy"
            })
        
        low_rel_pct = len(retrieval_failures["low_relevance"]) / retrieval_failures["total_queries"]
        if low_rel_pct > 0.2:
            recommendations.append({
                "issue": "Low relevance scores",
                "severity": "MEDIUM",
                "suggestion": "Adjust score thresholds or improve embedding quality"
            })
        
        if retrieval_failures["technical_queries"]:
            avg_tech_precision = sum(f["precision"] for f in retrieval_failures["technical_queries"]) / len(retrieval_failures["technical_queries"])
            if avg_tech_precision < 0.6:
                recommendations.append({
                    "issue": "Poor performance on technical queries",
                    "severity": "HIGH",
                    "suggestion": "Use domain-specific embeddings or adjust BM25 weights"
                })
        
        # Generation recommendations
        halluc_pct = len(generation_failures["hallucinations"]) / generation_failures["total_queries"]
        if halluc_pct > 0.2:
            recommendations.append({
                "issue": "High hallucination rate",
                "severity": "HIGH",
                "suggestion": "Improve context quality or add stricter generation constraints"
            })
        
        oos_pct = len(generation_failures["out_of_scope"]) / generation_failures["total_queries"]
        if oos_pct > 0.15:
            recommendations.append({
                "issue": "Many out-of-scope responses",
                "severity": "MEDIUM",
                "suggestion": "Improve retrieval recall or expand knowledge base"
            })
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. [{rec['severity']}] {rec['issue']}")
                print(f"   → {rec['suggestion']}")
        else:
            print("\n✓ No critical issues detected!")
        
        return recommendations
    
    def run_full_analysis(self):
        """Run complete failure analysis."""
        print("\n" + "="*80)
        print("COMPREHENSIVE FAILURE ANALYSIS")
        print("="*80)
        print(f"Threshold: {self.threshold}")
        print(f"Dataset: {len(self.eval_data)} queries")
        
        # Analyze retrieval
        retrieval_failures = self.analyze_retrieval_failures()
        self.print_retrieval_report(retrieval_failures)
        
        # Analyze generation
        generation_failures = self.analyze_generation_failures(num_queries=20)
        self.print_generation_report(generation_failures)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(retrieval_failures, generation_failures)
        
        # Save results
        results = {
            "threshold": self.threshold,
            "retrieval_failures": {
                "missing_context_count": len(retrieval_failures["missing_context"]),
                "low_relevance_count": len(retrieval_failures["low_relevance"]),
                "technical_queries_count": len(retrieval_failures["technical_queries"]),
                "total_queries": retrieval_failures["total_queries"]
            },
            "generation_failures": {
                "hallucinations_count": len(generation_failures["hallucinations"]),
                "out_of_scope_count": len(generation_failures["out_of_scope"]),
                "incomplete_count": len(generation_failures["incomplete_answers"]),
                "total_queries": generation_failures["total_queries"]
            },
            "recommendations": recommendations
        }
        
        output_file = Path("results/failure_analysis.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\n✓ Analysis saved to: {output_file}")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze RAG system failures")
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Score threshold for failure detection (default: 0.6)')
    parser.add_argument('--results', type=str,
                       help='Path to baseline results JSON (optional)')
    args = parser.parse_args()
    
    try:
        analyzer = FailureAnalyzer(threshold=args.threshold)
        results = analyzer.run_full_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
