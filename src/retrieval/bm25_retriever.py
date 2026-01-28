"""
BM25 sparse retrieval implementation.
Provides keyword-based search to complement dense retrieval.
"""

from typing import List, Dict
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.config import settings
from src.utils.logger import log
from src.data.processor import Chunk


class BM25Retriever:
    """BM25-based sparse retrieval for keyword search."""
    
    def __init__(self, persist_path: str = None):
        """
        Initialize BM25 retriever.
        
        Args:
            persist_path: Path to save/load BM25 index
        """
        self.persist_path = persist_path or Path(settings.data_processed_path) / "bm25_index.pkl"
        self.bm25 = None
        self.documents = []
        self.document_ids = []
        self.metadatas = []
        
        log.info("BM25Retriever initialized")
    
    def build_index(self, chunks: List[Chunk]):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of Chunk objects
        """
        if not chunks:
            log.warning("No chunks provided to build index")
            return
        
        log.info(f"Building BM25 index from {len(chunks)} chunks")
        
        # Store documents and metadata
        self.documents = [chunk.content for chunk in chunks]
        self.document_ids = [chunk.chunk_id for chunk in chunks]
        self.metadatas = [chunk.metadata for chunk in chunks]
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
        log.info("BM25 index built successfully")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with BM25 scores
        """
        if self.bm25 is None:
            log.error("BM25 index not built. Call build_index() first.")
            return []
        
        top_k = top_k or settings.top_k_retrieval
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append({
                    "id": self.document_ids[idx],
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(scores[idx]),
                    "retrieval_method": "bm25"
                })
        
        log.debug(f"BM25 search returned {len(results)} results")
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = False
    ) -> List[Dict]:
        """
        Retrieve documents (wrapper around search for API compatibility).
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Ignored for BM25Retriever (no reranker)
            
        Returns:
            List of retrieved documents with scores
        """
        return self.search(query, top_k=top_k)
    
    def save_index(self):
        """Save BM25 index to disk."""
        if self.bm25 is None:
            log.warning("No index to save")
            return
        
        try:
            # Create directory if it doesn't exist
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index data
            index_data = {
                "bm25": self.bm25,
                "documents": self.documents,
                "document_ids": self.document_ids,
                "metadatas": self.metadatas
            }
            
            with open(self.persist_path, "wb") as f:
                pickle.dump(index_data, f)
            
            log.info(f"BM25 index saved to {self.persist_path}")
            
        except Exception as e:
            log.error(f"Error saving BM25 index: {e}")
    
    def load_index(self) -> bool:
        """
        Load BM25 index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.persist_path.exists():
            log.warning(f"BM25 index file not found: {self.persist_path}")
            return False
        
        try:
            with open(self.persist_path, "rb") as f:
                index_data = pickle.load(f)
            
            self.bm25 = index_data["bm25"]
            self.documents = index_data["documents"]
            self.document_ids = index_data["document_ids"]
            self.metadatas = index_data["metadatas"]
            
            log.info(f"BM25 index loaded from {self.persist_path}")
            return True
            
        except Exception as e:
            log.error(f"Error loading BM25 index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the BM25 index."""
        if self.bm25 is None:
            return {"indexed": False}
        
        return {
            "indexed": True,
            "total_documents": len(self.documents),
            "avg_doc_length": sum(len(doc.split()) for doc in self.documents) / len(self.documents),
            "persist_path": str(self.persist_path)
        }


if __name__ == "__main__":
    # Example usage
    from src.data.processor import Chunk
    
    # Create sample chunks
    sample_chunks = [
        Chunk(
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "doc1.txt"},
            chunk_id="doc1_0",
            start_index=0,
            end_index=100
        ),
        Chunk(
            content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "doc2.txt"},
            chunk_id="doc2_0",
            start_index=0,
            end_index=100
        ),
        Chunk(
            content="Natural language processing helps computers understand human language.",
            metadata={"source": "doc3.txt"},
            chunk_id="doc3_0",
            start_index=0,
            end_index=100
        )
    ]
    
    # Initialize and build index
    retriever = BM25Retriever()
    retriever.build_index(sample_chunks)
    
    # Search
    results = retriever.search("machine learning artificial intelligence", top_k=3)
    print(f"BM25 search results:")
    for result in results:
        print(f"  Score: {result['score']:.4f} - {result['content'][:50]}...")
    
    # Get stats
    stats = retriever.get_stats()
    print(f"\nBM25 stats: {stats}")
