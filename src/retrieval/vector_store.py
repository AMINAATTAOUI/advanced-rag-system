"""
Vector store using LangChain's Chroma integration.

LangChain Components Used:
- langchain_chroma.Chroma                   → Managed vector store wrapping ChromaDB
- langchain_core.documents.Document          → Standardised document schema
- HuggingFaceEmbeddings (via EmbeddingModel) → Embedding function passed to Chroma

Key LangChain patterns:
  Chroma.from_documents()   → index Documents in one call
  vectorstore.as_retriever() → convert to a LangChain Retriever for use in chains
  similarity_search_with_score()  → retrieval with relevance scores
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

# ── LangChain Chroma VectorStore ────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.utils.logger import log
from src.retrieval.embeddings import EmbeddingModel
from src.data.processor import Chunk


class VectorStore:
    """
    ChromaDB vector store managed through LangChain's Chroma wrapper.
    
    Exposes both LangChain-native methods (as_retriever, from_documents)
    and backward-compatible methods (add_chunks, search).
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: EmbeddingModel = None
    ):
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.vector_db_path
        self.embedding_model = embedding_model or EmbeddingModel()
        
        log.info(f"Initializing LangChain Chroma at {self.persist_directory}")

        # ── LangChain Chroma vectorstore ─────────────────────────────
        # Uses the HuggingFaceEmbeddings instance from our EmbeddingModel
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model.lc_embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        
        # Keep a reference to the underlying collection for stats
        self.collection = self.vectorstore._collection
        
        log.info(
            f"LangChain Chroma collection '{self.collection_name}' ready. "
            f"Count: {self.collection.count()}"
        )

    # ── index Chunk objects (backward-compatible) ────────────────────
    def add_chunks(self, chunks: List[Chunk], show_progress: bool = True):
        """Add Chunk objects by converting them to LangChain Documents first."""
        if not chunks:
            log.warning("No chunks to add")
            return
        
        log.info(f"Adding {len(chunks)} chunks to LangChain Chroma vectorstore")

        # Convert Chunk dataclass → LangChain Document
        lc_docs = []
        ids = []
        for chunk in chunks:
            lc_docs.append(
                Document(page_content=chunk.content, metadata=chunk.metadata)
            )
            ids.append(chunk.chunk_id)

        # ── Chroma.add_documents handles embedding + storage ─────
        batch_size = 100
        for i in range(0, len(lc_docs), batch_size):
            batch_docs = lc_docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            try:
                self.vectorstore.add_documents(batch_docs, ids=batch_ids)
                log.debug(f"Added batch {i // batch_size + 1}")
            except Exception as e:
                log.error(f"Error adding batch: {e}")

        log.info(f"Successfully added chunks. Total count: {self.collection.count()}")

    # ── index LangChain Documents directly ───────────────────────────
    def add_documents(self, documents: List[Document], ids: List[str] = None):
        """
        Add LangChain Document objects directly (pure LangChain path).
        
        Usage:
            chunks = text_splitter.split_documents(docs)
            vector_store.add_documents(chunks)
        """
        self.vectorstore.add_documents(documents, ids=ids)
        log.info(f"Added {len(documents)} LangChain Documents. "
                 f"Total: {self.collection.count()}")

    # ── search ───────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using Chroma's query API to retrieve results WITH their IDs.
        
        LangChain's similarity_search_with_score does not return doc IDs,
        so we use the underlying collection.query() and wrap the results.
        """
        top_k = top_k or settings.top_k_retrieval

        try:
            # Embed the query via LangChain embeddings
            query_embedding = self.embedding_model.lc_embeddings.embed_query(query)

            # Use the underlying Chroma collection to get IDs + distances
            raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            for i in range(len(raw["ids"][0])):
                doc_id = raw["ids"][0][i]
                distance = raw["distances"][0][i]
                formatted.append({
                    "id": doc_id,
                    "content": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "score": 1 - distance,  # cosine distance -> similarity
                    "distance": distance,
                })

            log.debug(f"Found {len(formatted)} results for query")
            return formatted

        except Exception as e:
            log.error(f"Error searching: {e}")
            return []

    def retrieve(self, query: str, top_k: int = None,
                 use_reranking: bool = False,
                 filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Backward-compatible retrieve wrapper."""
        return self.search(query, top_k=top_k, filter_dict=filter_dict)

    # ── LangChain Retriever ──────────────────────────────────────────
    def as_retriever(self, **kwargs):
        """
        Convert to a LangChain Retriever for use in LCEL chains.
        
        Usage:
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            docs = retriever.invoke("query")
        """
        return self.vectorstore.as_retriever(**kwargs)

    # ── class method: build from documents in one shot ───────────────
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding_model: EmbeddingModel = None,
        collection_name: str = None,
        persist_directory: str = None,
    ) -> "VectorStore":
        """
        Create a VectorStore from LangChain Documents in one call.
        
        Usage:
            vs = VectorStore.from_documents(chunks, embedding_model)
        """
        emb = embedding_model or EmbeddingModel()
        col = collection_name or settings.collection_name
        pdir = persist_directory or settings.vector_db_path

        instance = cls.__new__(cls)
        instance.collection_name = col
        instance.persist_directory = pdir
        instance.embedding_model = emb

        instance.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=emb.lc_embeddings,
            collection_name=col,
            persist_directory=pdir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        instance.collection = instance.vectorstore._collection
        log.info(f"Created VectorStore from {len(documents)} documents")
        return instance

    # ── search by embedding ──────────────────────────────────────────
    def search_by_embedding(
        self,
        embedding: List[float],
        top_k: int = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        top_k = top_k or settings.top_k_retrieval
        try:
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            results = self.vectorstore.similarity_search_by_vector(
                embedding, k=top_k, filter=filter_dict
            )
            return [
                {"id": doc.metadata.get("chunk_id", ""),
                 "content": doc.page_content,
                 "metadata": doc.metadata,
                 "score": 1.0}
                for doc in results
            ]
        except Exception as e:
            log.error(f"Error searching by embedding: {e}")
            return []
    
    # ── utility methods ──────────────────────────────────────────────
    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        try:
            raw = self.collection.get(ids=ids, include=["documents", "metadatas"])
            return [
                {"id": raw["ids"][i], "content": raw["documents"][i],
                 "metadata": raw["metadatas"][i]}
                for i in range(len(raw["ids"]))
            ]
        except Exception as e:
            log.error(f"Error getting by IDs: {e}")
            return []
    
    def delete_collection(self):
        try:
            self.vectorstore.delete_collection()
            log.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            log.error(f"Error deleting collection: {e}")
    
    def reset_collection(self):
        self.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model.lc_embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.collection = self.vectorstore._collection
        log.info(f"Reset collection: {self.collection_name}")
    
    def get_stats(self) -> Dict:
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_dimension": self.embedding_model.get_embedding_dimension(),
            "persist_directory": self.persist_directory,
        }


if __name__ == "__main__":
    vs = VectorStore()
    stats = vs.get_stats()
    print(f"Vector store stats: {stats}")
    
    # Pure LangChain path:
    # retriever = vs.as_retriever(search_kwargs={"k": 5})
    # docs = retriever.invoke("What is machine learning?")
