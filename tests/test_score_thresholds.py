#!/usr/bin/env python3
"""
Test different score threshold configurations for hybrid retrieval.

This tests the min_dense_score parameter which filters low-confidence
dense (vector) retrieval results to reduce noise.

Usage:
    python scripts/test_score_thresholds.py
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import Reranker
from tests.test_baseline_metrics import BaselineMetricsCapture
from tests.test_evaluate_retrieval import RetrievalEvaluator


def test_score_thresholds():
    """Test different min_dense_score thresholds."""
    print("\n" + "="*80)
    print("TESTING SCORE THRESHOLDS")
    print("="*80)
    print("\nCurrent best: chunk=512, overlap=100, P@10=0.755")
    print("\nTesting min_dense_score thresholds:")
    print("(Filters low-confidence vector search results to reduce noise)")
    
    configs = [
        {"threshold": 0.3, "name": "threshold=0.3"},
        {"threshold": 0.4, "name": "threshold=0.4"},
        {"threshold": 0.5, "name": "threshold=0.5 (current)"},
        {"threshold": 0.6, "name": "threshold=0.6"},
    ]
    
    results = []
    
    # Initialize components once
    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    bm25_retriever.load_index()
    reranker = Reranker()
    
    for config in configs:
        print(f"\n→ Testing {config['name']}...")
        
        # Create hybrid retriever with custom threshold
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            min_dense_score=config['threshold']
        )
        
        # Create evaluator with custom retriever
        evaluator = RetrievalEvaluator(
            method='hybrid',
            top_k=10,
            precision_k=10,
            recall_k=10
        )
        
        # Replace the hybrid retriever
        evaluator.hybrid_retriever = hybrid_retriever
        
        # Run evaluation
        metrics, _ = evaluator.evaluate(include_reranking_comparison=False)
        precision = metrics.precision_at_k
        
        results.append({
            "config": config['name'],
            "threshold": config['threshold'],
            "precision": precision,
            "change": ((precision - 0.755) / 0.755 * 100)
        })
        
        print(f"  P@10: {precision:.3f} ({results[-1]['change']:+.1f}%)")
    
    # Summary
    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    print(f"{'Config':<25} {'Threshold':<12} {'P@10':<10} {'vs 0.755'}")
    print("-"*80)
    for r in results:
        print(f"{r['config']:<25} {r['threshold']:<12.1f} {r['precision']:<10.3f} {r['change']:+.1f}%")
    
    best = max(results, key=lambda x: x['precision'])
    print(f"\n✓ Best: {best['config']} with P@10={best['precision']:.3f}")
    
    if best['precision'] > 0.755:
        improvement = ((best['precision'] - 0.755) / 0.755 * 100)
        print(f"  Improvement over baseline: +{improvement:.1f}%")
    
    return results


if __name__ == "__main__":
    test_score_thresholds()
