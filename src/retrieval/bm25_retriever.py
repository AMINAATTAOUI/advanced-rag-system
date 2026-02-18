"""
BM25 sparse retrieval using LangChain's BM25Retriever.

LangChain Components Used:
- langchain_community.retrievers.BM25Retriever  → Sparse keyword retrieval
- langchain_core.documents.Document              → Standardised document schema

Key LangChain patterns:
  BM25Retriever.from_documents()  → build index from Documents
  retriever.invoke(query)          → run retrieval (returns List[Document])
"""

from typing import List, Dict
import pickle
from pathlib import Path

# ── LangChain BM25 Retriever ────────────────────────────────────────────
from langchain_community.retrievers import BM25Retriever as LangChainBM25
from langchain_core.documents import Document

from src.config import settings
from src.utils.logger import log
from src.data.processor import Chunk


class BM25Retriever:
    """
    BM25-based sparse retrieval managed through LangChain's BM25Retriever.

    Exposes both LangChain-native interface (.as_retriever / .invoke)
    and backward-compatible methods (build_index, search, save_index, load_index).
    """

    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or Path(settings.data_processed_path) / "bm25_index.pkl"
        self.lc_retriever: LangChainBM25 = None  # LangChain retriever
        self.documents: List[str] = []
        self.document_ids: List[str] = []
        self.metadatas: List[dict] = []

        log.info("LangChain BM25Retriever initialized")

    # ── build from Chunk dataclass objects (backward-compat) ─────────
    def build_index(self, chunks: List[Chunk]):
        """Build BM25 index from Chunk objects using LangChain BM25Retriever."""
        if not chunks:
            log.warning("No chunks provided to build index")
            return

        log.info(f"Building LangChain BM25 index from {len(chunks)} chunks")

        # Store for persistence & backward-compat search
        self.documents = [chunk.content for chunk in chunks]
        self.document_ids = [chunk.chunk_id for chunk in chunks]
        self.metadatas = [chunk.metadata for chunk in chunks]

        # Convert to LangChain Documents
        lc_docs = [
            Document(page_content=chunk.content,
                     metadata={**chunk.metadata, "chunk_id": chunk.chunk_id})
            for chunk in chunks
        ]

        # ── LangChain BM25Retriever.from_documents ──────────────
        self.lc_retriever = LangChainBM25.from_documents(
            lc_docs,
            k=settings.top_k_retrieval,
        )

        log.info("LangChain BM25 index built successfully")

    # ── build directly from LangChain Documents ─────────────────────
    def build_index_from_documents(self, documents: List[Document], k: int = None):
        """
        Build BM25 index from LangChain Document objects (pure LC path).

        Usage:
            bm25.build_index_from_documents(text_splitter.split_documents(docs))
        """
        k = k or settings.top_k_retrieval
        self.lc_retriever = LangChainBM25.from_documents(documents, k=k)
        self.documents = [d.page_content for d in documents]
        self.document_ids = [d.metadata.get("chunk_id", f"doc_{i}") for i, d in enumerate(documents)]
        self.metadatas = [d.metadata for d in documents]
        log.info(f"Built BM25 index from {len(documents)} LangChain Documents")

    # ── search (backward-compatible dict output) ────────────────────
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search using LangChain BM25, returning legacy dict format."""
        if self.lc_retriever is None:
            log.error("BM25 index not built. Call build_index() first.")
            return []

        top_k = top_k or settings.top_k_retrieval
        self.lc_retriever.k = top_k

        # ── LangChain retriever invoke ───────────────────────────
        lc_results: List[Document] = self.lc_retriever.invoke(query)

        results = []
        for doc in lc_results:
            results.append({
                "id": doc.metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0,  # BM25Retriever doesn't expose raw scores
                "retrieval_method": "bm25",
            })

        log.debug(f"BM25 search returned {len(results)} results")
        return results

    def retrieve(self, query: str, top_k: int = None,
                 use_reranking: bool = False) -> List[Dict]:
        """Backward-compatible retrieve wrapper."""
        return self.search(query, top_k=top_k)

    # ── LangChain retriever interface ────────────────────────────────
    def as_retriever(self):
        """Return the underlying LangChain BM25Retriever for use in chains."""
        if self.lc_retriever is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        return self.lc_retriever

    # ── persistence ──────────────────────────────────────────────────
    def save_index(self):
        """Save BM25 index data to disk for later reload."""
        if self.lc_retriever is None:
            log.warning("No index to save")
            return

        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            index_data = {
                "documents": self.documents,
                "document_ids": self.document_ids,
                "metadatas": self.metadatas,
            }
            with open(self.persist_path, "wb") as f:
                pickle.dump(index_data, f)
            log.info(f"BM25 index saved to {self.persist_path}")
        except Exception as e:
            log.error(f"Error saving BM25 index: {e}")

    def load_index(self) -> bool:
        """Load BM25 index from disk and rebuild the LangChain retriever."""
        if not Path(self.persist_path).exists():
            log.warning(f"BM25 index file not found: {self.persist_path}")
            return False

        try:
            with open(self.persist_path, "rb") as f:
                index_data = pickle.load(f)

            self.documents = index_data["documents"]
            self.document_ids = index_data["document_ids"]
            self.metadatas = index_data["metadatas"]

            # Rebuild LangChain BM25Retriever from stored documents
            lc_docs = [
                Document(page_content=doc,
                         metadata={**meta, "chunk_id": doc_id})
                for doc, doc_id, meta in zip(self.documents, self.document_ids, self.metadatas)
            ]
            self.lc_retriever = LangChainBM25.from_documents(
                lc_docs, k=settings.top_k_retrieval
            )

            log.info(f"BM25 index loaded from {self.persist_path}")
            return True
        except Exception as e:
            log.error(f"Error loading BM25 index: {e}")
            return False

    # ── stats ────────────────────────────────────────────────────────
    @property
    def bm25(self):
        """Backward-compat: return vectorizer if available."""
        if self.lc_retriever is not None:
            return getattr(self.lc_retriever, "vectorizer", True)
        return None

    def get_stats(self) -> Dict:
        if self.lc_retriever is None:
            return {"indexed": False}
        return {
            "indexed": True,
            "total_documents": len(self.documents),
            "avg_doc_length": sum(len(d.split()) for d in self.documents) / max(len(self.documents), 1),
            "persist_path": str(self.persist_path),
        }


if __name__ == "__main__":
    from src.data.processor import Chunk

    sample_chunks = [
        Chunk(content="Machine learning is a subset of artificial intelligence.",
              metadata={"source": "doc1.txt"}, chunk_id="doc1_0",
              start_index=0, end_index=100),
        Chunk(content="Deep learning uses neural networks with multiple layers.",
              metadata={"source": "doc2.txt"}, chunk_id="doc2_0",
              start_index=0, end_index=100),
    ]

    retriever = BM25Retriever()
    retriever.build_index(sample_chunks)

    results = retriever.search("machine learning artificial intelligence", top_k=3)
    for r in results:
        print(f"  {r['id']}: {r['content'][:60]}...")

    # Pure LangChain path:
    # lc_ret = retriever.as_retriever()
    # docs = lc_ret.invoke("machine learning")
    # for d in docs: print(d.page_content)
