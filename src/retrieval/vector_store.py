"""
Vector store implementation using ChromaDB.
Handles storage and retrieval of document embeddings.
"""

from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import settings
from src.utils.logger import log
from src.retrieval.embeddings import EmbeddingModel
from src.data.processor import Chunk


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: EmbeddingModel = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Embedding model instance
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.vector_db_path
        self.embedding_model = embedding_model or EmbeddingModel()
        
        log.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            log.info(f"Collection '{self.collection_name}' ready. Count: {self.collection.count()}")
        except Exception as e:
            log.error(f"Error initializing collection: {e}")
            raise
    
    def add_chunks(self, chunks: List[Chunk], show_progress: bool = True):
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects
            show_progress: Whether to show progress
        """
        if not chunks:
            log.warning("No chunks to add")
            return
        
        log.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Prepare data
        documents = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        log.info("Generating embeddings...")
        embeddings = self.embedding_model.encode_documents(
            documents,
            show_progress=show_progress
        )
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            try:
                self.collection.add(
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end].tolist(),
                    ids=ids[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                log.debug(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            except Exception as e:
                log.error(f"Error adding batch: {e}")
                continue
        
        log.info(f"Successfully added chunks. Total count: {self.collection.count()}")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of search results with scores
        """
        top_k = top_k or settings.top_k_retrieval
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_queries(query)
        
        # Search
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "distance": results["distances"][0][i]
                })
            
            log.debug(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            log.error(f"Error searching: {e}")
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = False,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve documents (wrapper around search for API compatibility).
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Ignored for VectorStore (no reranker)
            filter_dict: Metadata filters
            
        Returns:
            List of retrieved documents with scores
        """
        return self.search(query, top_k=top_k, filter_dict=filter_dict)
    
    def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of search results
        """
        top_k = top_k or settings.top_k_retrieval
        
        try:
            # Convert embedding to list if it's a numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            log.error(f"Error searching by embedding: {e}")
            return []
    
    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        """
        Get documents by their IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents
        """
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            for i in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
            
            return formatted_results
            
        except Exception as e:
            log.error(f"Error getting by IDs: {e}")
            return []
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            log.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            log.error(f"Error deleting collection: {e}")
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)."""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        log.info(f"Reset collection: {self.collection_name}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_dimension": self.embedding_model.get_embedding_dimension(),
            "persist_directory": self.persist_directory
        }


if __name__ == "__main__":
    # Example usage
    from src.data.processor import Chunk
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Create sample chunks (with token_count field)
    sample_chunks = [
        Chunk(
            content="This is the first document about machine learning.",
            metadata={"source": "doc1.txt"},
            chunk_id="doc1_0",
            start_index=0,
            end_index=52,
            token_count=10
        ),
        Chunk(
            content="This is the second document about deep learning.",
            metadata={"source": "doc2.txt"},
            chunk_id="doc2_0",
            start_index=0,
            end_index=50,
            token_count=10
        )
    ]
    
    # Add chunks
    # vector_store.add_chunks(sample_chunks)
    
    # Search
    # results = vector_store.search("machine learning", top_k=5)
    # print(f"Search results: {results}")
    
    # Get stats
    stats = vector_store.get_stats()
    print(f"Vector store stats: {stats}")
