"""
Embedding model using LangChain's HuggingFaceEmbeddings.

LangChain Components Used:
- langchain_huggingface.HuggingFaceEmbeddings  → Wraps sentence-transformers
  models with the LangChain Embeddings interface, compatible with
  Chroma.from_documents(), vectorstore.as_retriever(), etc.
"""

from typing import List, Union
import numpy as np

# ── LangChain Embeddings ────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.utils.logger import log


class EmbeddingModel:
    """
    Wrapper around LangChain's HuggingFaceEmbeddings.
    
    Provides the same API as before but backed by LangChain,
    so the embedding object can be passed directly to:
        Chroma.from_documents(documents, embedding=model.lc_embeddings)
        Chroma(embedding_function=model.lc_embeddings)
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None
    ):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        
        log.info(f"Loading embedding model via LangChain HuggingFaceEmbeddings: "
                 f"{self.model_name} on {self.device}")

        # ── LangChain HuggingFaceEmbeddings ─────────────────────────
        self.lc_embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": self.batch_size,
            },
        )
        
        # Determine embedding dimension from a test embed
        test_vec = self.lc_embeddings.embed_query("test")
        self.embedding_dim = len(test_vec)
        log.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    # ── encode texts ─────────────────────────────────────────────────
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode text(s) into embeddings using LangChain embed_documents."""
        if isinstance(texts, str):
            texts = [texts]
        
        # LangChain embed_documents returns List[List[float]]
        embeddings = self.lc_embeddings.embed_documents(texts)
        return np.array(embeddings)

    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """Encode queries using LangChain embed_query."""
        if isinstance(queries, str):
            queries = [queries]
        vecs = [self.lc_embeddings.embed_query(q) for q in queries]
        return np.array(vecs)

    def encode_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Encode documents using LangChain embed_documents."""
        embeddings = self.lc_embeddings.embed_documents(documents)
        return np.array(embeddings)

    # ── similarity ───────────────────────────────────────────────────
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        e1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        e2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.dot(e1, e2.T)
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.encode(texts)


if __name__ == "__main__":
    model = EmbeddingModel()
    
    text = "This is a sample sentence."
    embedding = model.encode(text)
    print(f"Embedding shape: {embedding.shape}")
    
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = model.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    query_emb = model.encode_queries("Sample query")
    similarities = model.similarity(query_emb, embeddings)
    print(f"Similarities: {similarities}")
