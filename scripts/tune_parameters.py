#!/usr/bin/env python3
"""
Parameter Tuning Script for RAG System

Tests different parameter configurations to optimize precision at 100 PDFs.
Runs one parameter at a time to isolate impact.

Usage:
    python scripts/tune_parameters.py --step 1
    python scripts/tune_parameters.py --step 2
    ...etc
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from tests.test_baseline_metrics import BaselineMetricsCapture


def test_reranker_top_k():
    """Step 1: Test different reranker top_k values."""
    print("\n" + "="*80)
    print("STEP 1: TUNING RERANKER TOP_K")
    print("="*80)
    print("\nBaseline: top_k=5, P@10=0.705")
    print("\nTesting configurations:")
    
    configs = [
        {"top_k": 10, "name": "top_k=10"},
        {"top_k": 15, "name": "top_k=15"},
        {"top_k": 20, "name": "top_k=20"}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n→ Testing {config['name']}...")
        
        # Update settings
        original_top_k = settings.top_k_rerank
        settings.top_k_rerank = config['top_k']
        
        # Run test
        baseline = BaselineMetricsCapture(dataset_size=100)
        metrics = baseline.capture_retrieval_baseline(method='hybrid')
        precision = getattr(metrics, f'precision_at_{baseline.top_k}', metrics.precision_at_k)
        
        results.append({
            "config": config['name'],
            "precision": precision,
            "change": ((precision - 0.705) / 0.705 * 100)
        })
        
        print(f"  P@10: {precision:.3f} ({results[-1]['change']:+.1f}%)")
        
        # Restore
        settings.top_k_rerank = original_top_k
    
    # Summary
    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    for r in results:
        print(f"{r['config']:<20} P@10: {r['precision']:.3f} ({r['change']:+.1f}%)")
    
    best = max(results, key=lambda x: x['precision'])
    print(f"\n✓ Best: {best['config']} with P@10={best['precision']:.3f}")
    
    return results


def test_hybrid_weights():
    """Step 2: Test different hybrid retrieval weights."""
    print("\n" + "="*80)
    print("STEP 2: TUNING HYBRID WEIGHTS")
    print("="*80)
    print("\nBaseline: dense=0.7, sparse=0.3, P@10=0.705")
    print("\nTesting configurations:")
    
    configs = [
        {"dense": 0.6, "sparse": 0.4, "name": "60/40"},
        {"dense": 0.5, "sparse": 0.5, "name": "50/50"},
        {"dense": 0.8, "sparse": 0.2, "name": "80/20"}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n→ Testing {config['name']}...")
        
        # Update settings
        original_dense = settings.dense_weight
        original_sparse = settings.sparse_weight
        settings.dense_weight = config['dense']
        settings.sparse_weight = config['sparse']
        
        # Run test
        baseline = BaselineMetricsCapture(dataset_size=100)
        metrics = baseline.capture_retrieval_baseline(method='hybrid')
        precision = getattr(metrics, f'precision_at_{baseline.top_k}', metrics.precision_at_k)
        
        results.append({
            "config": config['name'],
            "precision": precision,
            "change": ((precision - 0.705) / 0.705 * 100)
        })
        
        print(f"  P@10: {precision:.3f} ({results[-1]['change']:+.1f}%)")
        
        # Restore
        settings.dense_weight = original_dense
        settings.sparse_weight = original_sparse
    
    # Summary
    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    for r in results:
        print(f"{r['config']:<20} P@10: {r['precision']:.3f} ({r['change']:+.1f}%)")
    
    best = max(results, key=lambda x: x['precision'])
    print(f"\n✓ Best: {best['config']} with P@10={best['precision']:.3f}")
    
    return results


def test_retrieval_top_k():
    """Step 3: Test different retrieval top_k values."""
    print("\n" + "="*80)
    print("STEP 3: TUNING RETRIEVAL TOP_K")
    print("="*80)
    print("\nBaseline: top_k=20, P@10=0.705")
    print("\nTesting configurations:")
    
    configs = [
        {"top_k": 30, "name": "top_k=30"},
        {"top_k": 40, "name": "top_k=40"},
        {"top_k": 50, "name": "top_k=50"}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n→ Testing {config['name']}...")
        
        # Update settings
        original_top_k = settings.top_k_retrieval
        settings.top_k_retrieval = config['top_k']
        
        # Run test
        baseline = BaselineMetricsCapture(dataset_size=100)
        metrics = baseline.capture_retrieval_baseline(method='hybrid')
        precision = getattr(metrics, f'precision_at_{baseline.top_k}', metrics.precision_at_k)
        
        results.append({
            "config": config['name'],
            "precision": precision,
            "change": ((precision - 0.705) / 0.705 * 100)
        })
        
        print(f"  P@10: {precision:.3f} ({results[-1]['change']:+.1f}%)")
        
        # Restore
        settings.top_k_retrieval = original_top_k
    
    # Summary
    print("\n" + "-"*80)
    print("RESULTS:")
    print("-"*80)
    for r in results:
        print(f"{r['config']:<20} P@10: {r['precision']:.3f} ({r['change']:+.1f}%)")
    
    best = max(results, key=lambda x: x['precision'])
    print(f"\n✓ Best: {best['config']} with P@10={best['precision']:.3f}")
    
    return results


def test_chunk_overlap():
    """Step 4: Test different chunk overlap values."""
    print("\n" + "="*80)
    print("STEP 4: TUNING CHUNK OVERLAP")
    print("="*80)
    print("\nBaseline: overlap=50, P@10=0.705")
    print("\nNOTE: Requires rebuilding index for each test")
    print("\nConfigurations to test manually:")
    print("  1. overlap=100 → rebuild index → test")
    print("  2. overlap=150 → rebuild index → test")
    print("\nCommands:")
    print("  1. Edit .env: CHUNK_OVERLAP=100")
    print("  2. python scripts/build_index.py")
    print("  3. python tests/test_baseline_metrics.py --dataset_size 100 --save_results")
    print("\n(Skipping automated test - requires index rebuild)")


def test_score_threshold():
    """Step 5: Test adding minimum score threshold."""
    print("\n" + "="*80)
    print("STEP 5: ADDING SCORE THRESHOLD")
    print("="*80)
    print("\nBaseline: no threshold, P@10=0.705")
    print("\nThis requires code changes to add threshold filtering.")
    print("\nRecommended approach:")
    print("  1. Add min_score parameter to hybrid_retriever.py")
    print("  2. Filter results below threshold")
    print("  3. Test thresholds: 0.3, 0.4, 0.5")
    print("\n(Skipping - requires code implementation)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter tuning for RAG system")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], 
                       help="Which tuning step to run")
    args = parser.parse_args()
    
    if not args.step:
        print(__doc__)
        print("\nAvailable steps:")
        print("  1. Reranker top_k (5 → 10, 15, 20)")
        print("  2. Hybrid weights (0.7/0.3 → 0.6/0.4, 0.5/0.5, 0.8/0.2)")
        print("  3. Retrieval top_k (20 → 30, 40, 50)")
        print("  4. Chunk overlap (50 → 100, 150) [manual]")
        print("  5. Score threshold [requires code changes]")
        print("\nRun: python scripts/tune_parameters.py --step 1")
        sys.exit(0)
    
    if args.step == 1:
        test_reranker_top_k()
    elif args.step == 2:
        test_hybrid_weights()
    elif args.step == 3:
        test_retrieval_top_k()
    elif args.step == 4:
        test_chunk_overlap()
    elif args.step == 5:
        test_score_threshold()
