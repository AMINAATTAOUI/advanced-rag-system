"""
Cross-encoder reranker using LangChain's document compressor pipeline.

LangChain Components Used:
- langchain_classic.retrievers.document_compressors.CrossEncoderReranker
    -> Wraps a cross-encoder model for reranking
- langchain_community.cross_encoders.HuggingFaceCrossEncoder
    -> Loads the HuggingFace cross-encoder model
- langchain_classic.retrievers.ContextualCompressionRetriever
    -> Wraps any retriever + compressor into a single retriever

Key LangChain patterns:
  compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(...), top_n=10)
  ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
"""

from typing import List, Dict

# ── LangChain Cross-Encoder Reranker ────────────────────────────────────
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

from src.config import settings
from src.utils.logger import log


class Reranker:
    """
    Cross-encoder reranker managed through LangChain's CrossEncoderReranker
    and ContextualCompressionRetriever.

    Exposes both LangChain-native interface (as_compressor / wrap_retriever)
    and backward-compatible rerank(query, documents, top_k) method.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.embedding_device

        log.info(f"Loading LangChain CrossEncoder reranker: {self.model_name}")

        try:
            # ── LangChain HuggingFaceCrossEncoder ────────────────
            self.cross_encoder = HuggingFaceCrossEncoder(
                model_name=self.model_name,
            )

            # ── LangChain CrossEncoderReranker (document compressor) ─
            self.compressor = CrossEncoderReranker(
                model=self.cross_encoder,
                top_n=settings.top_k_rerank,
            )

            log.info("LangChain CrossEncoder reranker loaded successfully")
        except Exception as e:
            log.error(f"Error loading reranker model: {e}")
            raise

    # ── backward-compatible rerank method ────────────────────────────
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
    ) -> List[Dict]:
        """
        Rerank documents using LangChain CrossEncoderReranker.
        Accepts and returns legacy dict format for backward compatibility.
        """
        if not documents:
            log.warning("No documents to rerank")
            return []

        top_k = top_k or settings.top_k_rerank
        self.compressor.top_n = top_k

        log.debug(f"Reranking {len(documents)} documents with LangChain CrossEncoder")

        # Convert dicts → LangChain Documents
        lc_docs = [
            Document(
                page_content=doc["content"],
                metadata={**doc.get("metadata", {}),
                          "chunk_id": doc.get("id", ""),
                          "original_score": doc.get("score", 0.0)},
            )
            for doc in documents
        ]

        # ── LangChain CrossEncoderReranker.compress_documents ────
        reranked_docs: List[Document] = self.compressor.compress_documents(
            lc_docs, query
        )

        # Convert back to legacy dict format
        results = []
        for doc in reranked_docs:
            relevance_score = doc.metadata.get("relevance_score", 0.0)
            results.append({
                "id": doc.metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(relevance_score),
                "rerank_score": float(relevance_score),
                "original_score": doc.metadata.get("original_score", 0.0),
            })

        log.debug(f"Reranking complete. Returning top {len(results)} documents")
        return results

    # ── score a single pair ──────────────────────────────────────────
    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        result = self.rerank(query, [{"content": document, "id": "single", "metadata": {}}], top_k=1)
        return result[0]["rerank_score"] if result else 0.0

    # ── LangChain native: wrap any retriever with reranking ──────────
    def wrap_retriever(self, base_retriever) -> ContextualCompressionRetriever:
        """
        Wrap a LangChain retriever with this reranker using ContextualCompressionRetriever.

        Usage:
            reranked_retriever = reranker.wrap_retriever(ensemble_retriever)
            docs = reranked_retriever.invoke("query")  # already reranked
        """
        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever,
        )

    # ── expose the compressor for direct chain usage ─────────────────
    def as_compressor(self):
        """Return the underlying LangChain CrossEncoderReranker compressor."""
        return self.compressor


if __name__ == "__main__":
    reranker = Reranker()

    sample_docs = [
        {"content": "Machine learning is a subset of artificial intelligence.",
         "score": 0.8, "id": "doc1", "metadata": {}},
        {"content": "Deep learning uses neural networks.",
         "score": 0.75, "id": "doc2", "metadata": {}},
        {"content": "Python is a programming language.",
         "score": 0.7, "id": "doc3", "metadata": {}},
    ]

    query = "What is machine learning?"
    reranked = reranker.rerank(query, sample_docs, top_k=3)
    for doc in reranked:
        print(f"  ID: {doc['id']}, Rerank: {doc['rerank_score']:.4f}")

    # Pure LangChain path:
    # reranked_retriever = reranker.wrap_retriever(some_retriever)
    # docs = reranked_retriever.invoke("What is ML?")
