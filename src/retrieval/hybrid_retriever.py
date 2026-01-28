"""
Hybrid retriever combining dense and sparse retrieval.

Implements multiple fusion strategies:
- Reciprocal Rank Fusion (RRF)
- Score Normalization + Weighted Sum
- Adaptive Fusion

Features:
- Score normalization to handle different scales
- Minimum score thresholds to filter noise
- Adaptive weighting based on result quality
- Logging and debugging
"""

from typing import List, Dict, Optional, Literal
import numpy as np
from src.config import settings
from src.utils.logger import log
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import Reranker


class HybridRetriever:
    """
    Hybrid retrieval combining dense (vector) and sparse (BM25) search.

    Supports multiple fusion strategies:
    - 'rrf': Reciprocal Rank Fusion
    - 'weighted_sum': Normalized score weighted sum
    - 'adaptive': Context-aware adaptive fusion
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        bm25_retriever: BM25Retriever = None,
        reranker: Reranker = None,
        dense_weight: float = None,
        sparse_weight: float = None,
        fusion_strategy: Literal['rrf', 'weighted_sum', 'adaptive'] = 'weighted_sum',
        min_dense_score: float = 0.5,
        min_sparse_score: float = 0.0,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: VectorStore instance
            bm25_retriever: BM25Retriever instance
            reranker: Reranker instance (optional)
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
            fusion_strategy: Strategy for combining results ('rrf', 'weighted_sum', 'adaptive')
            min_dense_score: Minimum score threshold for dense results
            min_sparse_score: Minimum score threshold for sparse results
            rrf_k: RRF constant (default: 60)
        """
        self.vector_store = vector_store or VectorStore()
        
        # Initialize BM25 and load index if not provided
        if bm25_retriever is None:
            self.bm25_retriever = BM25Retriever()
            try:
                self.bm25_retriever.load_index()
                log.debug("BM25 index loaded automatically in HybridRetriever")
            except Exception as e:
                log.warning(f"Failed to load BM25 index: {e}")
        else:
            self.bm25_retriever = bm25_retriever
        
        self.reranker = reranker
        
        self.dense_weight = dense_weight or settings.dense_weight
        self.sparse_weight = sparse_weight or settings.sparse_weight
        self.fusion_strategy = fusion_strategy
        self.min_dense_score = min_dense_score
        self.min_sparse_score = min_sparse_score
        self.rrf_k = rrf_k
        
        # Normalize weights
        total_weight = self.dense_weight + self.sparse_weight
        self.dense_weight /= total_weight
        self.sparse_weight /= total_weight
        
        log.info(
            f"HybridRetriever initialized: "
            f"strategy={fusion_strategy}, "
            f"dense_weight={self.dense_weight:.2f}, "
            f"sparse_weight={self.sparse_weight:.2f}, "
            f"min_dense_score={min_dense_score}, "
            f"min_sparse_score={min_sparse_score}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve documents using hybrid search with production-grade fusion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to apply reranking
            filter_dict: Metadata filters for vector search
            
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or settings.top_k_retrieval
        
        log.debug(f"Hybrid retrieval for query: {query[:50]}...")
        
        # Get results from both retrievers (fetch more for better fusion)
        dense_results = self.vector_store.search(
            query,
            top_k=top_k * 3,  # Get 3x for better coverage
            filter_dict=filter_dict
        )
        
        sparse_results = self.bm25_retriever.search(
            query,
            top_k=top_k * 3
        )
        
        # Apply score thresholds to filter noise
        dense_results = self._filter_by_score(dense_results, self.min_dense_score, 'dense')
        sparse_results = self._filter_by_score(sparse_results, self.min_sparse_score, 'sparse')
        
        log.debug(
            f"After filtering: Dense={len(dense_results)}, Sparse={len(sparse_results)} "
            f"(thresholds: dense≥{self.min_dense_score}, sparse≥{self.min_sparse_score})"
        )
        
        # Choose fusion strategy
        if self.fusion_strategy == 'rrf':
            combined_results = self._reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                top_k=top_k if not use_reranking else top_k * 2
            )
        elif self.fusion_strategy == 'weighted_sum':
            combined_results = self._weighted_sum_fusion(
                dense_results,
                sparse_results,
                top_k=top_k if not use_reranking else top_k * 2
            )
        elif self.fusion_strategy == 'adaptive':
            combined_results = self._adaptive_fusion(
                dense_results,
                sparse_results,
                query,
                top_k=top_k if not use_reranking else top_k * 2
            )
        else:
            log.warning(f"Unknown fusion strategy: {self.fusion_strategy}, falling back to RRF")
            combined_results = self._reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                top_k=top_k if not use_reranking else top_k * 2
            )
        
        # Apply reranking if enabled
        if use_reranking and self.reranker and combined_results:
            log.debug("Applying cross-encoder reranking...")
            combined_results = self.reranker.rerank(
                query,
                combined_results,
                top_k=top_k
            )
        else:
            combined_results = combined_results[:top_k]
        
        log.debug(f"Final results: {len(combined_results)} documents")
        return combined_results
    
    def _filter_by_score(
        self,
        results: List[Dict],
        min_score: float,
        method_name: str
    ) -> List[Dict]:
        """
        Filter results by minimum score threshold.

        Args:
            results: List of results to filter
            min_score: Minimum score threshold
            method_name: Name of method (for logging)

        Returns:
            Filtered results
        """
        if min_score <= 0:
            return results
        
        filtered = [r for r in results if r.get('score', 0) >= min_score]
        
        if len(filtered) < len(results):
            log.debug(
                f"Filtered {method_name}: {len(results)} → {len(filtered)} "
                f"(removed {len(results) - len(filtered)} low-score results)"
            )
        
        return filtered
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score = sum(1 / (k + rank_i))
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of results to return
            k: RRF constant (default: 60)
            
        Returns:
            Fused and ranked results
        """
        # Create a dictionary to store combined scores
        doc_scores = {}
        doc_data = {}
        
        # Process dense results
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            rrf_score = self.dense_weight / (k + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
                doc_data[doc_id] = doc
            
            doc_scores[doc_id] += rrf_score
            
            # Store original dense score
            if "dense_score" not in doc_data[doc_id]:
                doc_data[doc_id]["dense_score"] = doc.get("score", 0)
        
        # Process sparse results
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc["id"]
            rrf_score = self.sparse_weight / (k + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
                doc_data[doc_id] = doc
            
            doc_scores[doc_id] += rrf_score
            
            # Store original sparse score
            if "sparse_score" not in doc_data[doc_id]:
                doc_data[doc_id]["sparse_score"] = doc.get("score", 0)
        
        # Sort by combined score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Create final results
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc = doc_data[doc_id].copy()
            doc["score"] = doc_scores[doc_id]
            doc["fusion_method"] = "rrf"
            results.append(doc)
        
        return results
    
    def _weighted_sum_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Combine results using normalized weighted sum.

        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        doc_scores = {}
        doc_data = {}
        
        # Normalize dense scores
        if dense_results:
            dense_scores = [r.get('score', 0) for r in dense_results]
            dense_min, dense_max = min(dense_scores), max(dense_scores)
            dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
            
            for doc in dense_results:
                doc_id = doc["id"]
                raw_score = doc.get('score', 0)
                norm_score = (raw_score - dense_min) / dense_range
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_data[doc_id] = doc
                
                doc_scores[doc_id] += self.dense_weight * norm_score
                doc_data[doc_id]["dense_score"] = raw_score
                doc_data[doc_id]["dense_norm_score"] = norm_score
        
        # Normalize sparse scores
        if sparse_results:
            sparse_scores = [r.get('score', 0) for r in sparse_results]
            sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
            sparse_range = sparse_max - sparse_min if sparse_max > sparse_min else 1.0
            
            for doc in sparse_results:
                doc_id = doc["id"]
                raw_score = doc.get('score', 0)
                norm_score = (raw_score - sparse_min) / sparse_range
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_data[doc_id] = doc
                
                doc_scores[doc_id] += self.sparse_weight * norm_score
                doc_data[doc_id]["sparse_score"] = raw_score
                doc_data[doc_id]["sparse_norm_score"] = norm_score
        
        # Sort by combined score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Create final results
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc = doc_data[doc_id].copy()
            doc["score"] = doc_scores[doc_id]
            doc["fusion_method"] = "weighted_sum"
            results.append(doc)
        
        log.debug(f"Weighted sum fusion: combined {len(results)} unique documents")
        return results
    
    def _adaptive_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        query: str,
        top_k: int
    ) -> List[Dict]:
        """
        Adaptive fusion that adjusts weights based on result quality.

        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            query: Original query (for context)
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        # Calculate overlap
        dense_ids = set([r['id'] for r in dense_results])
        sparse_ids = set([r['id'] for r in sparse_results])
        overlap = len(dense_ids & sparse_ids)
        total_unique = len(dense_ids | sparse_ids)
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
        
        # Calculate average scores
        avg_dense = np.mean([r.get('score', 0) for r in dense_results]) if dense_results else 0
        avg_sparse = np.mean([r.get('score', 0) for r in sparse_results]) if sparse_results else 0
        
        # Adaptive weight adjustment
        if overlap_ratio > 0.5:
            # High overlap - methods agree, use configured weights
            adaptive_dense_weight = self.dense_weight
            adaptive_sparse_weight = self.sparse_weight
            log.debug(f"High overlap ({overlap_ratio:.2%}) - using configured weights")
        else:
            # Low overlap - trust the method with higher scores
            if avg_dense > avg_sparse * 1.2:  # Dense significantly better
                adaptive_dense_weight = 0.85
                adaptive_sparse_weight = 0.15
                log.debug(f"Low overlap ({overlap_ratio:.2%}), dense stronger - adjusted to 85/15")
            elif avg_sparse > avg_dense * 1.2:  # Sparse significantly better
                adaptive_dense_weight = 0.15
                adaptive_sparse_weight = 0.85
                log.debug(f"Low overlap ({overlap_ratio:.2%}), sparse stronger - adjusted to 15/85")
            else:
                # Similar quality, slightly favor dense (semantic search)
                adaptive_dense_weight = 0.75
                adaptive_sparse_weight = 0.25
                log.debug(f"Low overlap ({overlap_ratio:.2%}), similar quality - adjusted to 75/25")
        
        # Normalize adaptive weights
        total = adaptive_dense_weight + adaptive_sparse_weight
        adaptive_dense_weight /= total
        adaptive_sparse_weight /= total
        
        # Use weighted sum with adaptive weights
        doc_scores = {}
        doc_data = {}
        
        # Normalize and combine dense results
        if dense_results:
            dense_scores = [r.get('score', 0) for r in dense_results]
            dense_min, dense_max = min(dense_scores), max(dense_scores)
            dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
            
            for doc in dense_results:
                doc_id = doc["id"]
                raw_score = doc.get('score', 0)
                norm_score = (raw_score - dense_min) / dense_range
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_data[doc_id] = doc
                
                doc_scores[doc_id] += adaptive_dense_weight * norm_score
                doc_data[doc_id]["dense_score"] = raw_score
        
        # Normalize and combine sparse results
        if sparse_results:
            sparse_scores = [r.get('score', 0) for r in sparse_results]
            sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
            sparse_range = sparse_max - sparse_min if sparse_max > sparse_min else 1.0
            
            for doc in sparse_results:
                doc_id = doc["id"]
                raw_score = doc.get('score', 0)
                norm_score = (raw_score - sparse_min) / sparse_range
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_data[doc_id] = doc
                
                doc_scores[doc_id] += adaptive_sparse_weight * norm_score
                doc_data[doc_id]["sparse_score"] = raw_score
        
        # Sort by combined score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Create final results
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc = doc_data[doc_id].copy()
            doc["score"] = doc_scores[doc_id]
            doc["fusion_method"] = "adaptive"
            doc["adaptive_weights"] = {
                "dense": adaptive_dense_weight,
                "sparse": adaptive_sparse_weight,
                "overlap_ratio": overlap_ratio
            }
            results.append(doc)
        
        log.debug(
            f"Adaptive fusion: overlap={overlap_ratio:.2%}, "
            f"weights=({adaptive_dense_weight:.2f}/{adaptive_sparse_weight:.2f})"
        )
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = None
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve documents and return separate results from each method.
        Useful for analysis and debugging.
        
        Args:
            query: Search query
            top_k: Number of results per method
            
        Returns:
            Dictionary with results from each method
        """
        top_k = top_k or settings.top_k_retrieval
        
        dense_results = self.vector_store.search(query, top_k=top_k)
        sparse_results = self.bm25_retriever.search(query, top_k=top_k)
        hybrid_results = self.retrieve(query, top_k=top_k, use_reranking=False)
        
        return {
            "dense": dense_results,
            "sparse": sparse_results,
            "hybrid": hybrid_results
        }


if __name__ == "__main__":
    # Example usage
    from src.data.processor import Chunk
    
    # Create sample chunks
    sample_chunks = [
        Chunk(
            content="Machine learning is a subset of artificial intelligence that focuses on data.",
            metadata={"source": "doc1.txt"},
            chunk_id="doc1_0",
            start_index=0,
            end_index=100
        ),
        Chunk(
            content="Deep learning uses neural networks with multiple layers for complex tasks.",
            metadata={"source": "doc2.txt"},
            chunk_id="doc2_0",
            start_index=0,
            end_index=100
        ),
        Chunk(
            content="Natural language processing enables computers to understand human language.",
            metadata={"source": "doc3.txt"},
            chunk_id="doc3_0",
            start_index=0,
            end_index=100
        )
    ]
    
    # Initialize retrievers
    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    
    # Build indices
    # vector_store.add_chunks(sample_chunks)
    # bm25_retriever.build_index(sample_chunks)
    
    # Initialize hybrid retriever
    hybrid = HybridRetriever(vector_store, bm25_retriever)
    
    # Retrieve
    # results = hybrid.retrieve("What is machine learning?", top_k=3)
    # print(f"Hybrid retrieval results: {len(results)}")
