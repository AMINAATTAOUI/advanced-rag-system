#!/usr/bin/env python3
"""
Scalability Baseline Comparison Tool

Compares RAG system performance across different dataset sizes (10, 50, 100 PDFs).
Shows how retrieval and generation metrics change as the system scales.

Usage:
    python tests/compare_scalability_baselines.py --compare
"""

import json
import os


def compare_baselines():
    """Load and compare baseline metrics across different dataset sizes."""
    files = {
        10: "results/baseline_metrics_10pdfs.json",
        50: "results/baseline_metrics_50pdfs.json",
        100: "results/baseline_metrics_100pdfs.json"
    }
    
    baselines = {}
    for size, filepath in files.items():
        if os.path.exists(filepath):
            with open(filepath) as f:
                baselines[size] = json.load(f)
    
    if len(baselines) < 2:
        print(f"[ERROR] Need at least 2 baseline files to compare")
        print(f"Found: {list(baselines.keys())}")
        print("\nRun baseline tests first:")
        print("  python tests/test_baseline_metrics.py --dataset_size 10 --save_results")
        print("  python tests/test_baseline_metrics.py --dataset_size 50 --save_results")
        print("  python tests/test_baseline_metrics.py --dataset_size 100 --save_results")
        return
    
    def get_top_k(dataset_size):
        """Determine adaptive top_k based on dataset size."""
        if dataset_size < 20:
            return 3
        elif dataset_size < 70:
            return 5
        else:
            return 10
    
    sizes = sorted(baselines.keys())
    top_ks = {size: get_top_k(size) for size in sizes}
    
    print("\n" + "="*100)
    print("SCALABILITY COMPARISON: " + " vs ".join([f"{s} PDFs" for s in sizes]))
    print("="*100)
    
    print("\nRETRIEVAL METRICS:")
    print("-"*100)
    header = f"{'Metric':<25}"
    for size in sizes:
        header += f"{f'{size} PDFs':<15}"
    if len(sizes) == 3:
        header += f"{'Δ10→50':<12}{'Δ50→100':<12}{'Δ10→100':<12}"
    else:
        header += f"{'Change':<20}"
    print(header)
    print("-"*100)
    
    def print_metric_row(metric_name, values):
        """Print metric values with percentage changes."""
        row = f"{metric_name:<25}"
        for val in values:
            row += f"{val:<15.3f}"
        
        if len(values) == 3:
            change_10_50 = ((values[1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            change_50_100 = ((values[2] - values[1]) / values[1] * 100) if values[1] != 0 else 0
            change_10_100 = ((values[2] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            
            row += f"{change_10_50:+.1f}%{'':<6}"
            row += f"{change_50_100:+.1f}%{'':<6}"
            row += f"{change_10_100:+.1f}%{'':<6}"
        else:
            change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            row += f"{change:+.1f}%{'':<14}"
        
        print(row)
    
    metric_name = "P@" + "/".join([str(top_ks[s]) for s in sizes])
    values = [baselines[size]['retrieval'][f'precision_at_{top_ks[size]}'] for size in sizes]
    print_metric_row(metric_name, values)
    
    metric_name = "R@" + "/".join([str(top_ks[s]) for s in sizes])
    values = [baselines[size]['retrieval'][f'recall_at_{top_ks[size]}'] for size in sizes]
    print_metric_row(metric_name, values)
    
    values = [baselines[size]['retrieval']['mrr'] for size in sizes]
    print_metric_row("MRR", values)
    
    values = [baselines[size]['retrieval']['avg_latency_ms'] for size in sizes]
    print_metric_row("Avg Latency (ms)", values)
    
    print("\nGENERATION METRICS:")
    print("-"*100)
    print(header)
    print("-"*100)
    
    for metric in ['faithfulness', 'hallucination_rate', 'answer_relevance']:
        values = [baselines[size]['generation'][metric] for size in sizes]
        print_metric_row(metric, values)


if __name__ == "__main__":
    import sys
    
    if "--compare" in sys.argv:
        compare_baselines()
    else:
        print(__doc__)
        print("\nRun with --compare flag to see results:")
        print("  python tests/compare_scalability_baselines.py --compare")
