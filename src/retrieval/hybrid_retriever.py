"""
Hybrid retriever using LangChain's EnsembleRetriever.

LangChain Components Used:
- langchain_classic.retrievers.EnsembleRetriever  -> Fuses dense + sparse retrievers (RRF built-in)
- langchain_core.documents.Document               -> Standardised document schema

Key LangChain patterns:
  EnsembleRetriever(retrievers=[dense, bm25], weights=[0.7, 0.3])
  ensemble.invoke(query)   → returns fused List[Document]
"""

from typing import List, Dict, Optional, Literal

# ── LangChain EnsembleRetriever ──────────────────────────────────────────
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.config import settings
from src.utils.logger import log
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import Reranker


class HybridRetriever:
    """
    Hybrid retrieval combining dense (vector) and sparse (BM25) search
    via LangChain's EnsembleRetriever with Reciprocal Rank Fusion.

    Backward-compatible search() and retrieve() methods are preserved.
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        bm25_retriever: BM25Retriever = None,
        reranker: Reranker = None,
        dense_weight: float = None,
        sparse_weight: float = None,
        fusion_strategy: Literal['rrf', 'weighted_sum', 'adaptive'] = 'rrf',
        min_dense_score: float = 0.5,
        min_sparse_score: float = 0.0,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store or VectorStore()

        # Initialize BM25 and try loading persisted index
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

        # Normalize weights
        dw = dense_weight or settings.dense_weight
        sw = sparse_weight or settings.sparse_weight
        total = dw + sw
        self.dense_weight = dw / total
        self.sparse_weight = sw / total

        self.fusion_strategy = fusion_strategy
        self.min_dense_score = min_dense_score
        self.min_sparse_score = min_sparse_score
        self.rrf_k = rrf_k

        # ── Build the LangChain EnsembleRetriever ────────────────
        self._build_ensemble()

        log.info(
            f"HybridRetriever initialized (LangChain EnsembleRetriever): "
            f"dense_weight={self.dense_weight:.2f}, "
            f"sparse_weight={self.sparse_weight:.2f}"
        )

    # ── internal: assemble the EnsembleRetriever ─────────────────────
    def _build_ensemble(self):
        """Create a LangChain EnsembleRetriever from dense + sparse."""
        retrievers = []
        weights = []

        # Dense retriever from Chroma
        dense_ret = self.vector_store.as_retriever(
            search_kwargs={"k": settings.top_k_retrieval * 3}
        )
        retrievers.append(dense_ret)
        weights.append(self.dense_weight)

        # Sparse BM25 retriever
        if self.bm25_retriever.lc_retriever is not None:
            retrievers.append(self.bm25_retriever.as_retriever())
            weights.append(self.sparse_weight)
        else:
            log.warning("BM25 retriever not ready – EnsembleRetriever uses dense only")

        # ── LangChain EnsembleRetriever (uses RRF internally) ────
        self.ensemble = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
        )

    # ── main retrieval method (backward-compatible) ──────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve documents using LangChain EnsembleRetriever + optional reranking.
        Returns backward-compatible list of dicts.
        """
        top_k = top_k or settings.top_k_retrieval
        log.debug(f"Hybrid retrieval for query: {query[:50]}...")

        # ── LangChain EnsembleRetriever.invoke ───────────────────
        lc_docs: List[Document] = self.ensemble.invoke(query)

        # Convert to legacy dict format
        combined_results = []
        for doc in lc_docs:
            # BM25 docs have chunk_id in metadata; Chroma dense docs don't.
            # Reconstruct the ID from source + chunk_index when chunk_id is absent.
            doc_id = doc.metadata.get("chunk_id", "")
            if not doc_id:
                source = doc.metadata.get("source", "")
                chunk_idx = doc.metadata.get("chunk_index", 0)
                doc_id = f"{source}_{chunk_idx}" if source else ""

            combined_results.append({
                "id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 1.0),
                "fusion_method": "ensemble_rrf",
            })

        # Apply reranking if enabled
        if use_reranking and self.reranker and combined_results:
            log.debug("Applying cross-encoder reranking...")
            combined_results = self.reranker.rerank(query, combined_results, top_k=top_k)
        else:
            combined_results = combined_results[:top_k]

        log.debug(f"Final results: {len(combined_results)} documents")
        return combined_results

    # ── LangChain native access ──────────────────────────────────────
    def as_retriever(self):
        """
        Return the underlying LangChain EnsembleRetriever for use in LCEL chains.

        Usage:
            retriever = hybrid.as_retriever()
            docs = retriever.invoke("query")
        """
        return self.ensemble

    # ── convenience: separate results for debugging ──────────────────
    def retrieve_with_scores(self, query: str, top_k: int = None) -> Dict[str, List[Dict]]:
        """Return separate results from each retrieval method."""
        top_k = top_k or settings.top_k_retrieval
        dense_results = self.vector_store.search(query, top_k=top_k)
        sparse_results = self.bm25_retriever.search(query, top_k=top_k)
        hybrid_results = self.retrieve(query, top_k=top_k, use_reranking=False)
        return {
            "dense": dense_results,
            "sparse": sparse_results,
            "hybrid": hybrid_results,
        }


if __name__ == "__main__":
    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    hybrid = HybridRetriever(vector_store, bm25_retriever)

    # LangChain native:
    # ensemble = hybrid.as_retriever()
    # docs = ensemble.invoke("What is machine learning?")
    # for d in docs: print(d.page_content[:80])
