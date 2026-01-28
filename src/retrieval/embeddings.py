"""
Embedding model for converting text to dense vectors.
Uses sentence-transformers for high-quality embeddings.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.utils.logger import log


class EmbeddingModel:
    """Wrapper for sentence-transformer embedding models."""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        
        log.info(f"Loading embedding model: {self.model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            log.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            log.error(f"Error loading embedding model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            log.debug(f"Encoded {len(texts)} texts into embeddings")
            return embeddings
            
        except Exception as e:
            log.error(f"Error encoding texts: {e}")
            raise
    
    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        Encode queries (optimized for search).
        
        Args:
            queries: Single query or list of queries
            
        Returns:
            Numpy array of query embeddings
        """
        # Some models have special query encoding
        # For now, use the same as regular encoding
        return self.encode(queries, normalize=True)
    
    def encode_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode documents (optimized for indexing).
        
        Args:
            documents: List of document texts
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of document embeddings
        """
        return self.encode(documents, show_progress=show_progress, normalize=True)
    
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize if not already normalized
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Compute dot product (cosine similarity for normalized vectors)
        similarity = np.dot(embeddings1, embeddings2.T)
        
        return similarity
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Allow calling the model directly."""
        return self.encode(texts)


if __name__ == "__main__":
    # Example usage
    model = EmbeddingModel()
    
    # Encode single text
    text = "This is a sample sentence."
    embedding = model.encode(text)
    print(f"Embedding shape: {embedding.shape}")
    
    # Encode multiple texts
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = model.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute similarity
    query_emb = model.encode("Sample query")
    similarities = model.similarity(query_emb, embeddings)
    print(f"Similarities: {similarities}")
