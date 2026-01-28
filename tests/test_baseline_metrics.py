#!/usr/bin/env python3
"""
Phase 1: Baseline Metrics Capture

Captures baseline performance metrics on current dataset.
Used to establish ground truth before parameter tuning.
Ensures system is stable and metrics are consistent.

Results saved to: results/baseline_metrics_<dataset_size>.json

Usage:
    python tests/test_baseline_metrics.py --dataset_size 10 --save_results
"""

# pylint: disable=too-many-locals,too-many-statements
import sys
import os
import json
import time
import argparse
import warnings
import logging
from typing import Dict, Any
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", message="Failed to send telemetry")
warnings.filterwarnings("ignore", message="resume_download")
logging.getLogger('chromadb').setLevel(logging.ERROR)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import evaluators directly from test modules
import importlib.util

def load_module(name, path):
    """Load module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

test_dir = os.path.dirname(__file__)
retrieval_module = load_module('test_evaluate_retrieval', 
                               os.path.join(test_dir, 'test_evaluate_retrieval.py'))
generation_module = load_module('test_evaluate_generation',
                                os.path.join(test_dir, 'test_evaluate_generation.py'))

RetrievalEvaluator = retrieval_module.RetrievalEvaluator
EvaluationMetrics = retrieval_module.EvaluationMetrics
GenerationEvaluator = generation_module.GenerationEvaluator
GenerationMetrics = generation_module.GenerationMetrics

from src.utils.logger import log


class BaselineMetricsCapture:
    """Capture and store baseline metrics for comparison."""
    
    def __init__(self, dataset_size: int = 10):
        """
        Initialize baseline capture.

        Args:
            dataset_size: Number of documents in dataset (10, 50, 100, etc.)
        """
        self.dataset_size = dataset_size
        self.results_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'results'
        )

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        self.baseline_file = os.path.join(
            self.results_dir,
            f'baseline_metrics_{dataset_size}pdfs.json'
        )

        # Adaptive top_k based on raw PDF count for consistency
        if dataset_size < 20:
            self.top_k = 3
        elif dataset_size < 70:
            self.top_k = 5
        else:
            self.top_k = 10

        log.info(f"BaselineMetricsCapture initialized for {dataset_size} PDFs (top_k={self.top_k})")
    
    def capture_retrieval_baseline(self, method: str = 'hybrid') -> EvaluationMetrics:
        """
        Capture retrieval baseline metrics.

        Args:
            method: Retrieval method ('dense', 'sparse', 'hybrid')

        Returns:
            EvaluationMetrics object
        """
        log.info(f"Capturing retrieval baseline ({method}, top_k={self.top_k})...")

        # Use top_k for both precision and recall evaluation to match retrieval depth
        evaluator = RetrievalEvaluator(method=method, top_k=self.top_k,
                                     precision_k=self.top_k, recall_k=self.top_k)
        metrics, _ = evaluator.evaluate(include_reranking_comparison=False)

        return metrics
    
    def capture_generation_baseline(self, method: str = 'hybrid') -> GenerationMetrics:
        """
        Capture generation baseline metrics.
        
        Args:
            method: Retrieval method ('dense', 'sparse', 'hybrid')
            
        Returns:
            GenerationMetrics object
        """
        log.info(f"Capturing generation baseline ({method})...")
        
        evaluator = GenerationEvaluator(method=method, top_k=5)
        metrics, _ = evaluator.evaluate(num_queries=None)
        
        return metrics
    
    def compile_baseline_report(self) -> Dict[str, Any]:
        """
        Compile comprehensive baseline report.
        
        Returns:
            Dictionary with all baseline metrics
        """
        log.info("Compiling baseline report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": f"{self.dataset_size} PDFs",
            "phase": "Phase 1: Baseline & Scalability Testing",
            "purpose": "Establish ground truth metrics before parameter tuning",
            "retrieval": {},
            "generation": {},
            "system_health": {},
            "notes": []
        }
        
        try:
            # Capture retrieval metrics
            log.info("\n" + "="*70)
            log.info("RETRIEVAL BASELINE CAPTURE")
            log.info("="*70)
            
            retrieval_metrics = self.capture_retrieval_baseline(method='hybrid')
            
            report["retrieval"] = {
                "method": "hybrid",
                f"precision_at_{self.top_k}": float(retrieval_metrics.precision_at_k),
                f"recall_at_{self.top_k}": float(retrieval_metrics.recall_at_k),
                "mrr": float(retrieval_metrics.mrr),
                "context_relevance": float(retrieval_metrics.context_relevance),
                "avg_latency_ms": float(retrieval_metrics.avg_latency_ms),
                "p50_latency_ms": float(retrieval_metrics.latency_p50),
                "p95_latency_ms": float(retrieval_metrics.latency_p95),
                "p99_latency_ms": float(retrieval_metrics.latency_p99),
                "total_queries": retrieval_metrics.total_queries,
                "failed_queries": retrieval_metrics.failed_queries
            }
            
            # Capture generation metrics
            log.info("\n" + "="*70)
            log.info("GENERATION BASELINE CAPTURE")
            log.info("="*70)
            
            generation_metrics = self.capture_generation_baseline(method='hybrid')
            
            report["generation"] = {
                "method": "hybrid",
                "faithfulness": float(generation_metrics.faithfulness),
                "answer_relevance": float(generation_metrics.answer_relevance),
                "context_precision": float(generation_metrics.context_precision),
                "hallucination_rate": float(generation_metrics.hallucination_rate),
                "avg_retrieval_time_ms": float(generation_metrics.avg_retrieval_time_ms),
                "avg_generation_time_ms": float(generation_metrics.avg_generation_time_ms),
                "avg_total_time_ms": float(generation_metrics.avg_total_time_ms),
                "total_queries": generation_metrics.total_queries,
                "failed_queries": generation_metrics.failed_queries
            }
            
            # System health assessment
            report["system_health"] = self._assess_system_health(
                retrieval_metrics,
                generation_metrics
            )
            
            # Add notes
            if retrieval_metrics.failed_queries > 0:
                report["notes"].append(
                    f"Warning: {retrieval_metrics.failed_queries} retrieval queries failed"
                )
            if generation_metrics.failed_queries > 0:
                report["notes"].append(
                    f"Warning: {generation_metrics.failed_queries} generation queries failed"
                )
            
            return report
            
        except Exception as e:
            log.error(f"Error compiling baseline report: {e}")
            raise
    
    def _assess_system_health(
        self,
        retrieval_metrics: EvaluationMetrics,
        generation_metrics: GenerationMetrics
    ) -> Dict[str, Any]:
        """Assess overall system health."""
        return {
            "retrieval_quality": self._assess_retrieval_health(retrieval_metrics),
            "generation_quality": self._assess_generation_health(generation_metrics),
            "overall_status": self._get_overall_status(retrieval_metrics, generation_metrics),
            "ready_for_tuning": self._is_ready_for_tuning(retrieval_metrics, generation_metrics)
        }
    
    def _assess_retrieval_health(self, metrics: EvaluationMetrics) -> str:
        """Assess retrieval health."""
        if metrics.precision_at_k >= 0.8 and metrics.recall_at_k >= 0.7:
            return "EXCELLENT"
        elif metrics.precision_at_k >= 0.6 and metrics.recall_at_k >= 0.5:
            return "GOOD"
        elif metrics.precision_at_k >= 0.4 and metrics.recall_at_k >= 0.3:
            return "FAIR"
        else:
            return "POOR"
    
    def _assess_generation_health(self, metrics: GenerationMetrics) -> str:
        """Assess generation health."""
        if metrics.faithfulness >= 0.8 and metrics.hallucination_rate <= 0.2:
            return "EXCELLENT"
        elif metrics.faithfulness >= 0.6 and metrics.hallucination_rate <= 0.4:
            return "GOOD"
        elif metrics.faithfulness >= 0.4 and metrics.hallucination_rate <= 0.6:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_overall_status(
        self,
        retrieval_metrics: EvaluationMetrics,
        generation_metrics: GenerationMetrics
    ) -> str:
        """Get overall system status."""
        retrieval_status = self._assess_retrieval_health(retrieval_metrics)
        generation_status = self._assess_generation_health(generation_metrics)
        
        if retrieval_status == "EXCELLENT" and generation_status == "EXCELLENT":
            return "PRODUCTION_READY"
        elif retrieval_status in ["EXCELLENT", "GOOD"] and generation_status in ["EXCELLENT", "GOOD"]:
            return "READY_FOR_TUNING"
        else:
            return "NEEDS_INVESTIGATION"
    
    def _is_ready_for_tuning(
        self,
        retrieval_metrics: EvaluationMetrics,
        generation_metrics: GenerationMetrics
    ) -> bool:
        """Check if system is ready for parameter tuning."""
        # System must have at minimum viable metrics to proceed
        return (
            retrieval_metrics.precision_at_k >= 0.5 and
            generation_metrics.faithfulness >= 0.4 and
            retrieval_metrics.failed_queries == 0 and
            generation_metrics.failed_queries == 0
        )
    
    def save_baseline(self, report: Dict[str, Any]):
        """
        Save baseline report to JSON file.
        
        Args:
            report: Baseline report dictionary
        """
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(report, f, indent=2)
            log.info(f"Baseline saved: {self.baseline_file}")
        except Exception as e:
            log.error(f"Error saving baseline: {e}")
            raise
    
    def print_baseline_summary(self, report: Dict[str, Any]):
        """Print baseline summary to console."""
        print("\n" + "="*70)
        print("  PHASE 1: BASELINE METRICS SUMMARY")
        print("="*70)
        print(f"Dataset: {report['dataset_size']} (top_k={self.top_k})")
        print(f"Timestamp: {report['timestamp']}")
        print()
        
        # Retrieval metrics
        print("RETRIEVAL METRICS:")
        print("-" * 70)
        ret = report["retrieval"]
        print(f"  Precision@{self.top_k}:         {ret[f'precision_at_{self.top_k}']:.3f}")
        print(f"  Recall@{self.top_k}:            {ret[f'recall_at_{self.top_k}']:.3f}")
        print(f"  MRR:                 {ret['mrr']:.3f}")
        print(f"  Context Relevance:   {ret['context_relevance']:.3f}")
        print(f"  Avg Latency:         {ret['avg_latency_ms']:.0f}ms (p95: {ret['p95_latency_ms']:.0f}ms)")
        print()
        
        # Generation metrics
        print("GENERATION METRICS:")
        print("-" * 70)
        gen = report["generation"]
        print(f"  Faithfulness:        {gen['faithfulness']:.3f}")
        print(f"  Answer Relevance:    {gen['answer_relevance']:.3f}")
        print(f"  Context Precision:   {gen['context_precision']:.3f}")
        print(f"  Hallucination Rate:  {gen['hallucination_rate']:.3f}")
        print(f"  Avg Total Time:      {gen['avg_total_time_ms']:.0f}ms")
        print()
        
        # System health
        print("SYSTEM HEALTH:")
        print("-" * 70)
        health = report["system_health"]
        print(f"  Retrieval Quality:   {health['retrieval_quality']}")
        print(f"  Generation Quality:  {health['generation_quality']}")
        print(f"  Overall Status:      {health['overall_status']}")
        print(f"  Ready for Tuning:    {'YES' if health['ready_for_tuning'] else 'NO'}")
        print()
        
        if report["notes"]:
            print("NOTES:")
            print("-" * 70)
            for note in report["notes"]:
                print(f"  • {note}")
            print()
        
        print("="*70)
        print(f"Baseline saved to: {self.baseline_file}")
        print("="*70)


def main():
    """Main baseline capture function."""
    parser = argparse.ArgumentParser(description="Capture baseline metrics")
    parser.add_argument('--dataset_size', type=int, default=10,
                       help='Dataset size (10, 50, 100, etc.)')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to JSON file')
    args = parser.parse_args()
    
    try:
        # Capture baseline
        baseline = BaselineMetricsCapture(dataset_size=args.dataset_size)
        report = baseline.compile_baseline_report()
        
        # Save and display
        if args.save_results:
            baseline.save_baseline(report)
        
        baseline.print_baseline_summary(report)
        
        # Exit code based on system status
        if report["system_health"]["ready_for_tuning"]:
            print("\n✓ Baseline captured successfully. System ready for tuning.")
            sys.exit(0)
        else:
            print("\n⚠ Baseline captured but system may need investigation.")
            sys.exit(0)
            
    except Exception as e:
        log.error(f"Baseline capture failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
