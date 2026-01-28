#!/usr/bin/env python3
"""
Retrieval Evaluation 

Tests each retrieval method individually using the evaluation dataset.
Measures Precision@K, Recall@K, MRR, context relevance, and latency.

Usage:
    python tests/test_evaluate_retrieval.py --method dense --top_k 5
    python tests/test_evaluate_retrieval.py --method sparse --top_k 5
    python tests/test_evaluate_retrieval.py --method hybrid --top_k 5
"""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches,unused-import
import sys
import os
import json
import time
import argparse
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Suppress ChromaDB telemetry warnings and verbose logging
warnings.filterwarnings("ignore", message="Failed to send telemetry")
warnings.filterwarnings("ignore", message="resume_download")
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.logger import log


@dataclass
class RetrievalResult:
    """Result of a single retrieval operation."""
    query_id: str
    query: str
    retrieved_docs: List[Dict[str, Any]]
    relevant_docs: List[str]
    latency_ms: float
    method: str


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for retrieval performance."""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    avg_latency_ms: float
    context_relevance: float
    total_queries: int
    precision_k: int
    recall_k: int
    # Enhanced metrics
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    avg_score: float = 0.0
    failed_queries: int = 0


@dataclass
class RerankerImpactMetrics:
    """Metrics comparing retrieval before and after reranking."""
    baseline_precision: float
    reranked_precision: float
    baseline_recall: float
    reranked_recall: float
    baseline_mrr: float
    reranked_mrr: float
    docs_moved_to_top3: int
    avg_position_change: float
    avg_score_improvement: float
    total_queries: int


class RetrievalEvaluator:
    """Evaluates retrieval methods using the evaluation dataset."""

    def __init__(self, method: str, top_k: int = 5,
                 precision_k: int = None, recall_k: int = None, use_adaptive: bool = True):
        self.method = method
        self.top_k = top_k
        self.eval_data = self._load_eval_dataset()

        # If no explicit precision_k/recall_k provided, use top_k for consistency
        self.precision_k = precision_k or top_k
        self.recall_k = recall_k or top_k

        # Only apply adaptive configuration if enabled and no explicit parameters provided
        if use_adaptive and settings.eval_adaptive and precision_k is None and recall_k is None:
            self._auto_configure_metrics()

        self._init_retrieval_components()

        log.debug(
            f"RetrievalEvaluator ready: {method}, top_k={top_k}, "
            f"P@{self.precision_k}, R@{self.recall_k}"
        )

    def _load_eval_dataset(self) -> List[Dict]:
        eval_path = os.path.join(settings.test_data_path, 'eval_dataset.json')
        if not os.path.exists(eval_path):
            raise FileNotFoundError(eval_path)

        encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1']
        for enc in encodings_to_try:
            try:
                with open(eval_path, 'r', encoding=enc) as f:
                    data = json.load(f)
                log.debug(f"Loaded {len(data)} queries")
                return data
            except Exception:
                continue
        raise ValueError("Could not load evaluation dataset")

    def _init_retrieval_components(self):
        if self.method == 'dense':
            self.embedding_model = EmbeddingModel()
            self.vector_store = VectorStore()
        elif self.method == 'sparse':
            self.bm25_retriever = BM25Retriever()
            if not self.bm25_retriever.load_index():
                self.vector_store = VectorStore()
                all_chunks = self._get_all_chunks_from_vector_store()
                if all_chunks:
                    self.bm25_retriever.build_index(all_chunks)
                    self.bm25_retriever.save_index()
        elif self.method == 'hybrid':
            self.embedding_model = EmbeddingModel()
            self.vector_store = VectorStore()
            self.bm25_retriever = BM25Retriever()
            if not self.bm25_retriever.load_index():
                all_chunks = self._get_all_chunks_from_vector_store()
                if all_chunks:
                    self.bm25_retriever.build_index(all_chunks)
                    self.bm25_retriever.save_index()
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_retriever=self.bm25_retriever,
                fusion_strategy='adaptive',
                dense_weight=0.7,
                sparse_weight=0.3,
                min_dense_score=0.5,
                min_sparse_score=0.0
            )
        log.debug(f"{self.method.capitalize()} retrieval ready")

    def _get_all_chunks_from_vector_store(self) -> List:
        from src.data.processor import Chunk
        try:
            results = self.vector_store.collection.get(
                limit=10000,
                include=['documents', 'metadatas', 'ids']
            )
            chunks = []
            for i, doc_id in enumerate(results['ids']):
                chunk = Chunk(
                    content=results['documents'][i],
                    metadata=results['metadatas'][i],
                    chunk_id=doc_id,
                    start_index=0,
                    end_index=len(results['documents'][i]),
                    token_count=len(results['documents'][i].split())
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            log.error(f"Error getting chunks: {e}")
            return []

    def _retrieve_documents(self, query: str) -> Tuple[List[Dict], float]:
        start_time = time.time()
        
        if self.method == 'dense':
            results = self.vector_store.retrieve(query, top_k=self.top_k)
            retrieved_docs = [{'id': r['id'], 'content': r['content'], 
                             'metadata': r['metadata'], 'score': r['score']} for r in results]
        elif self.method == 'sparse':
            results = self.bm25_retriever.retrieve(query, top_k=self.top_k)
            retrieved_docs = [{'id': r['id'], 'content': r['content'],
                             'metadata': r['metadata'], 'score': r['score']} for r in results]
        elif self.method == 'hybrid':
            results = self.hybrid_retriever.retrieve(query, top_k=self.top_k, use_reranking=False)
            retrieved_docs = [{'id': r['id'], 'content': r['content'],
                             'metadata': r['metadata'], 'score': r['score']} for r in results]
        
        latency_ms = (time.time() - start_time) * 1000
        return retrieved_docs, latency_ms

    def _auto_configure_metrics(self):
        all_docs = set()
        for query in self.eval_data:
            if query.get('eval_type') == 'retrieval':
                all_docs.update(query.get('relevant_docs', []))
        
        dataset_size = len(all_docs)
        
        if dataset_size <= settings.eval_small_dataset_threshold:
            self.precision_k = 3
            self.recall_k = 3
            log.info(f"Auto-config: small dataset ({dataset_size} docs) -> P@3, R@3")
        elif dataset_size <= settings.eval_medium_dataset_threshold:
            self.precision_k = 5
            self.recall_k = 5
            log.info(f"Auto-config: medium dataset ({dataset_size} docs) -> P@5, R@5")
        else:
            self.precision_k = 5
            self.recall_k = 10
            log.info(f"Auto-config: large dataset ({dataset_size} docs) -> P@5, R@10")

    def _extract_base_doc(self, doc_id):
        if '_' in doc_id:
            base_doc = '_'.join(doc_id.split('_')[:-1])
            if '\\' in base_doc or '/' in base_doc:
                base_doc = base_doc.split('\\')[-1].split('/')[-1]
            return base_doc
        return doc_id

    def _calculate_metrics(self, results: List[RetrievalResult]) -> EvaluationMetrics:
        precision_scores, recall_scores, reciprocal_ranks = [], [], []
        latencies, relevance_scores, all_scores = [], [], []

        for result in results:
            if result.retrieved_docs:
                scores = [doc.get('score', 0.0) for doc in result.retrieved_docs]
                all_scores.extend(scores)
            retrieved_ids = [doc['id'] for doc in result.retrieved_docs]
            relevant_ids = result.relevant_docs
            
            # Precision@K - count relevant CHUNKS in top K
            retrieved_top_k_ids = retrieved_ids[:self.precision_k]
            retrieved_top_k_base_docs = [self._extract_base_doc(doc_id) for doc_id in retrieved_top_k_ids]
            precision = len([doc for doc in retrieved_top_k_base_docs if doc in relevant_ids]) / float(self.precision_k)
            precision_scores.append(precision)
            
            # Recall@K - count UNIQUE relevant DOCUMENTS in top K (not chunks)
            retrieved_top_k_recall_ids = retrieved_ids[:self.recall_k]
            retrieved_top_k_recall_base_docs = set(self._extract_base_doc(doc_id) for doc_id in retrieved_top_k_recall_ids)
            unique_relevant_retrieved = len(retrieved_top_k_recall_base_docs & set(relevant_ids))
            recall = unique_relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
            recall_scores.append(recall)
            
            # MRR
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved_ids[:max(self.precision_k, self.recall_k)], 1):
                if self._extract_base_doc(doc_id) in relevant_ids:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_ranks.append(reciprocal_rank)
            
            relevance_scores.append(self._calculate_context_relevance(result))
            latencies.append(result.latency_ms)

        return EvaluationMetrics(
            precision_at_k=np.mean(precision_scores),
            recall_at_k=np.mean(recall_scores),
            mrr=np.mean(reciprocal_ranks),
            avg_latency_ms=np.mean(latencies),
            context_relevance=np.mean(relevance_scores),
            total_queries=len(results),
            precision_k=self.precision_k,
            recall_k=self.recall_k,
            latency_p50=np.percentile(latencies, 50) if latencies else 0.0,
            latency_p95=np.percentile(latencies, 95) if latencies else 0.0,
            latency_p99=np.percentile(latencies, 99) if latencies else 0.0,
            min_score=min(all_scores) if all_scores else 0.0,
            max_score=max(all_scores) if all_scores else 0.0,
            avg_score=np.mean(all_scores) if all_scores else 0.0
        )

    def _calculate_context_relevance(self, result: RetrievalResult) -> float:
        query_keywords = set(result.query.lower().split())
        total_relevance = 0.0
        for doc in result.retrieved_docs[:5]:
            doc_content = doc['content'].lower()
            matched = query_keywords & set(doc_content.split())
            relevance = len(matched) / len(query_keywords) if query_keywords else 0.0
            total_relevance += relevance
        return total_relevance / min(5, len(result.retrieved_docs)) if result.retrieved_docs else 0.0

    def evaluate(self, include_reranking_comparison: bool = False) -> Tuple[EvaluationMetrics, Optional['RerankerImpactMetrics']]:
        retrieval_queries = [q for q in self.eval_data if q.get('eval_type') == 'retrieval']
        log.info(f"Evaluating {len(retrieval_queries)} queries ({self.method})...")

        results, failed = [], 0
        eval_start = time.time()

        for i, query_data in enumerate(retrieval_queries):
            try:
                retrieved_docs, latency_ms = self._retrieve_documents(query_data['question'])
                log.debug(f"Q{i+1}/{len(retrieval_queries)}: {latency_ms:.0f}ms")
                
                results.append(RetrievalResult(
                    query_id=query_data['id'],
                    query=query_data['question'],
                    retrieved_docs=retrieved_docs,
                    relevant_docs=query_data['relevant_docs'],
                    latency_ms=latency_ms,
                    method=self.method
                ))
            except Exception as e:
                failed += 1
                log.error(f"Q{i+1} failed: {str(e)[:50]}")

        total_time = time.time() - eval_start
        metrics = self._calculate_metrics(results)
        metrics.failed_queries = failed
        qps = len(results) / total_time if total_time > 0 else 0

        log.info(
            f"Eval complete: {len(results)} queries, {total_time:.1f}s ({qps:.1f} QPS) | "
            f"P@{metrics.precision_k}={metrics.precision_at_k:.3f}, "
            f"R@{metrics.recall_k}={metrics.recall_at_k:.3f}, MRR={metrics.mrr:.3f} | "
            f"Latency: avg={metrics.avg_latency_ms:.0f}ms, p95={metrics.latency_p95:.0f}ms"
        )

        # Compare with reranking if requested and method is hybrid
        reranking_metrics = None
        if include_reranking_comparison and self.method == 'hybrid':
            log.info(f"Computing reranking impact comparison...")
            reranking_metrics = self._evaluate_reranking_impact(retrieval_queries)

        return metrics, reranking_metrics

    def print_results(self, metrics: EvaluationMetrics):
        print("\n" + "="*70)
        print(f"  RETRIEVAL EVALUATION - {self.method.upper()}")
        print("="*70)
        print(f"Method: {self.method} | Top-K: {self.top_k} | Metrics: P@{metrics.precision_k}, R@{metrics.recall_k}")
        print(f"Queries: {metrics.total_queries}" + (f" | Failed: {metrics.failed_queries}" if metrics.failed_queries > 0 else ""))
        print()
        
        print("Retrieval Quality:")
        print(f"  Precision@{metrics.precision_k}: {metrics.precision_at_k:.3f}")
        print(f"  Recall@{metrics.recall_k}: {metrics.recall_at_k:.3f}")
        print(f"  MRR: {metrics.mrr:.3f}")
        print(f"  Context Relevance: {metrics.context_relevance:.3f}")
        print()
        
        print("Performance:")
        print(f"  Avg: {metrics.avg_latency_ms:.1f}ms | P50: {metrics.latency_p50:.1f}ms | "
              f"P95: {metrics.latency_p95:.1f}ms | P99: {metrics.latency_p99:.1f}ms")
        print()
        
        print("Score Distribution:")
        print(f"  Min: {metrics.min_score:.3f} | Avg: {metrics.avg_score:.3f} | Max: {metrics.max_score:.3f}")
        print()

        # Assessment
        assessments = {
            f"Precision@{metrics.precision_k}": self._assess_precision(metrics.precision_at_k),
            f"Recall@{metrics.recall_k}": self._assess_recall(metrics.recall_at_k),
            "Latency (P95)": self._assess_latency(metrics.latency_p95),
            "Relevance": self._assess_relevance(metrics.context_relevance)
        }
        
        print("Assessment:")
        for metric, status in assessments.items():
            print(f"  {metric}: {status}")

    def print_reranking_comparison(self, metrics: 'RerankerImpactMetrics'):
        """Print reranking impact comparison results."""
        print("\n" + "="*70)
        print("  RERANKING IMPACT ANALYSIS (Hybrid Retrieval)")
        print("="*70)
        print()
        
        print("Metrics Comparison (Baseline vs Reranked):")
        print("-" * 70)
        print(f"  {'Metric':<25} {'Baseline':<15} {'Reranked':<15} {'Change':>10}")
        print("-" * 70)
        
        prec_change = metrics.reranked_precision - metrics.baseline_precision
        prec_pct = (prec_change / metrics.baseline_precision * 100) if metrics.baseline_precision > 0 else 0
        print(f"  {'Precision@3':<25} {metrics.baseline_precision:.3f}{'':>10} {metrics.reranked_precision:.3f}{'':>10} "
              f"{prec_pct:+.1f}%")
        
        recall_change = metrics.reranked_recall - metrics.baseline_recall
        recall_pct = (recall_change / metrics.baseline_recall * 100) if metrics.baseline_recall > 0 else 0
        print(f"  {'Recall@3':<25} {metrics.baseline_recall:.3f}{'':>10} {metrics.reranked_recall:.3f}{'':>10} "
              f"{recall_pct:+.1f}%")
        
        mrr_change = metrics.reranked_mrr - metrics.baseline_mrr
        mrr_pct = (mrr_change / metrics.baseline_mrr * 100) if metrics.baseline_mrr > 0 else 0
        print(f"  {'MRR':<25} {metrics.baseline_mrr:.3f}{'':>10} {metrics.reranked_mrr:.3f}{'':>10} "
              f"{mrr_pct:+.1f}%")
        
        print()
        print("Ranking Impact:")
        print("-" * 70)
        print(f"  Relevant docs moved to top-3: {metrics.docs_moved_to_top3}/{metrics.total_queries} queries")
        print(f"  Avg position change: {metrics.avg_position_change:+.2f} positions (- = higher rank)")
        print(f"  Avg score improvement: {metrics.avg_score_improvement:+.4f}")
        print()
        
        # Assessment
        if metrics.docs_moved_to_top3 > 0:
            print(f"[+] Reranking successfully improved ranking for {metrics.docs_moved_to_top3} queries")
        if metrics.avg_position_change > 0:
            print(f"[+] Documents ranked higher after reranking (avg {metrics.avg_position_change:.2f} positions)")
        if metrics.avg_score_improvement > 0:
            print(f"[+] Cross-encoder scores improved by avg {metrics.avg_score_improvement:.4f}")
        print()

    def _assess_precision(self, p): return "[EXCELLENT]" if p >= 0.8 else "[GOOD]" if p >= 0.6 else "[FAIR]" if p >= 0.4 else "[POOR]"
    def _assess_recall(self, r): return "[EXCELLENT]" if r >= 0.9 else "[GOOD]" if r >= 0.7 else "[FAIR]" if r >= 0.5 else "[POOR]"
    def _assess_latency(self, l): return "[FAST]" if l <= 200 else "[ACCEPTABLE]" if l <= 500 else "[SLOW]" if l <= 1000 else "[VERY_SLOW]"
    def _assess_relevance(self, r): return "[HIGH]" if r >= 0.7 else "[MODERATE]" if r >= 0.5 else "[LOW]" if r >= 0.3 else "[VERY_LOW]"

    def _evaluate_reranking_impact(self, retrieval_queries: List[Dict]) -> 'RerankerImpactMetrics':
        """
        Compare retrieval metrics before and after reranking.
        Tracks document ranking improvements and score changes.
        """
        baseline_results, reranked_results = [], []
        position_changes = []
        score_improvements = []
        docs_moved_to_top3 = 0

        for i, query_data in enumerate(retrieval_queries):
            try:
                # Get baseline results (without reranking)
                baseline_docs, _ = self._retrieve_documents_with_reranking(
                    query_data['question'], use_reranking=False
                )
                baseline_results.append(RetrievalResult(
                    query_id=query_data['id'],
                    query=query_data['question'],
                    retrieved_docs=baseline_docs,
                    relevant_docs=query_data['relevant_docs'],
                    latency_ms=0,
                    method=self.method
                ))

                # Get reranked results
                reranked_docs, _ = self._retrieve_documents_with_reranking(
                    query_data['question'], use_reranking=True
                )
                reranked_results.append(RetrievalResult(
                    query_id=query_data['id'],
                    query=query_data['question'],
                    retrieved_docs=reranked_docs,
                    relevant_docs=query_data['relevant_docs'],
                    latency_ms=0,
                    method=self.method
                ))

                # Track position changes for relevant docs
                relevant_doc_ids = set(query_data['relevant_docs'])
                for rel_doc in relevant_doc_ids:
                    baseline_base_doc_ids = [self._extract_base_doc(doc['id']) 
                                            for doc in baseline_docs]
                    reranked_base_doc_ids = [self._extract_base_doc(doc['id']) 
                                            for doc in reranked_docs]
                    
                    if rel_doc in baseline_base_doc_ids:
                        baseline_pos = baseline_base_doc_ids.index(rel_doc) + 1
                        if rel_doc in reranked_base_doc_ids:
                            reranked_pos = reranked_base_doc_ids.index(rel_doc) + 1
                            position_changes.append(baseline_pos - reranked_pos)
                            
                            # Check if moved into top-3
                            if reranked_pos <= 3 and baseline_pos > 3:
                                docs_moved_to_top3 += 1
                        
                        # Track score improvement
                        if baseline_pos - 1 < len(baseline_docs) and reranked_pos - 1 < len(reranked_docs):
                            baseline_score = baseline_docs[baseline_pos - 1].get('score', 0)
                            reranked_score = reranked_docs[reranked_pos - 1].get('score', 0)
                            score_improvements.append(reranked_score - baseline_score)

            except Exception as e:
                log.debug(f"Error in reranking comparison Q{i+1}: {str(e)[:50]}")

        # Calculate metrics
        baseline_metrics = self._calculate_metrics(baseline_results)
        reranked_metrics = self._calculate_metrics(reranked_results)

        return RerankerImpactMetrics(
            baseline_precision=baseline_metrics.precision_at_k,
            reranked_precision=reranked_metrics.precision_at_k,
            baseline_recall=baseline_metrics.recall_at_k,
            reranked_recall=reranked_metrics.recall_at_k,
            baseline_mrr=baseline_metrics.mrr,
            reranked_mrr=reranked_metrics.mrr,
            docs_moved_to_top3=docs_moved_to_top3,
            avg_position_change=np.mean(position_changes) if position_changes else 0.0,
            avg_score_improvement=np.mean(score_improvements) if score_improvements else 0.0,
            total_queries=len(retrieval_queries)
        )

    def _retrieve_documents_with_reranking(self, query: str, use_reranking: bool) -> Tuple[List[Dict], float]:
        """
        Retrieve documents with optional reranking for hybrid method.
        """
        start_time = time.time()

        if self.method == 'hybrid':
            results = self.hybrid_retriever.retrieve(query, top_k=self.top_k, 
                                                     use_reranking=use_reranking)
            retrieved_docs = [{'id': r['id'], 'content': r['content'],
                             'metadata': r['metadata'], 'score': r['score']} for r in results]
        else:
            # For non-hybrid methods, just use regular retrieve
            retrieved_docs, _ = self._retrieve_documents(query)

        latency_ms = (time.time() - start_time) * 1000
        return retrieved_docs, latency_ms



def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods")
    parser.add_argument('--method', choices=['dense', 'sparse', 'hybrid'], required=True)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--precision_k', type=int, default=None)
    parser.add_argument('--recall_k', type=int, default=None)
    parser.add_argument('--with_reranking', action='store_true',
                       help='Compare with reranking impact (hybrid only)')
    args = parser.parse_args()

    try:
        # Disable adaptive configuration if any explicit parameters are provided
        use_adaptive = (args.precision_k is None and args.recall_k is None and
                       not any(arg.startswith('--top_k') for arg in sys.argv[1:]))

        evaluator = RetrievalEvaluator(args.method, args.top_k, args.precision_k, args.recall_k, use_adaptive=use_adaptive)
        metrics, reranking_metrics = evaluator.evaluate(include_reranking_comparison=args.with_reranking)
        evaluator.print_results(metrics)

        if reranking_metrics:
            evaluator.print_reranking_comparison(reranking_metrics)
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
