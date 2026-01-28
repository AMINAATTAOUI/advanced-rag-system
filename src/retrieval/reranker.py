"""
Cross-encoder reranker for improving retrieval quality.
Reranks initial retrieval results using a more powerful model.
"""

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.config import settings
from src.utils.logger import log


class Reranker:
    """Cross-encoder based reranker for improving retrieval results."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.embedding_device
        
        log.info(f"Loading reranker model: {self.model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            log.info("Reranker model loaded successfully")
        except Exception as e:
            log.error(f"Error loading reranker model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document dictionaries with 'content' field
            top_k: Number of top results to return
            
        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            log.warning("No documents to rerank")
            return []
        
        top_k = top_k or settings.top_k_rerank
        
        log.debug(f"Reranking {len(documents)} documents")
        
        # Prepare query-document pairs
        pairs = [(query, doc["content"]) for doc in documents]
        
        # Compute relevance scores
        scores = self._compute_scores(pairs)
        
        # Add reranking scores to documents and update main score
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            doc["original_score"] = doc.get("score", 0.0)
            # Update main score to rerank score so chunks are properly ordered
            doc["score"] = float(score)
        
        # Sort by reranking score
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top-k
        result = reranked_docs[:top_k]
        
        log.debug(f"Reranking complete. Returning top {len(result)} documents")
        return result
    
    def _compute_scores(self, pairs: List[tuple]) -> List[float]:
        """
        Compute relevance scores for query-document pairs.
        
        Args:
            pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        scores = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get scores
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                
                # Handle single item case
                if len(batch_pairs) == 1:
                    batch_scores = [batch_scores.item()]
                else:
                    batch_scores = batch_scores.tolist()
                
                scores.extend(batch_scores)
        
        return scores
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: Search query
            document: Document text
            
        Returns:
            Relevance score
        """
        scores = self._compute_scores([(query, document)])
        return scores[0]


if __name__ == "__main__":
    # Example usage
    reranker = Reranker()
    
    # Sample documents
    sample_docs = [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "score": 0.8,
            "id": "doc1"
        },
        {
            "content": "Deep learning uses neural networks.",
            "score": 0.75,
            "id": "doc2"
        },
        {
            "content": "Python is a programming language.",
            "score": 0.7,
            "id": "doc3"
        }
    ]
    
    # Rerank
    query = "What is machine learning?"
    reranked = reranker.rerank(query, sample_docs, top_k=3)
    
    print("Reranked results:")
    for doc in reranked:
        print(f"  ID: {doc['id']}, Rerank Score: {doc['rerank_score']:.4f}, "
              f"Original Score: {doc['original_score']:.4f}")
