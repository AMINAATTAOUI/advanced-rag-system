"""
RAG chain orchestration.
Combines retrieval and generation into a complete RAG pipeline.
"""

from typing import List, Dict, Optional, Iterator
from src.config import settings
from src.utils.logger import log
from src.utils.cache import cached
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMClient
from src.generation.prompts import PromptTemplates


class RAGChain:
    """
    Complete RAG chain combining retrieval and generation.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever = None,
        llm_client: LLMClient = None,
        prompt_templates: PromptTemplates = None
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: HybridRetriever instance
            llm_client: LLMClient instance
            prompt_templates: PromptTemplates instance
        """
        # Initialize retriever with reranker if not provided
        if retriever is None:
            try:
                reranker = Reranker()
                self.retriever = HybridRetriever(reranker=reranker)
                log.info("RAG Chain initialized with reranker")
            except Exception as e:
                log.warning(f"Failed to initialize reranker: {e}. Proceeding without reranking.")
                self.retriever = HybridRetriever()
        else:
            self.retriever = retriever
        
        self.llm_client = llm_client or LLMClient()
        self.prompt_templates = prompt_templates or PromptTemplates()
        
        log.info("RAG Chain initialized")
    
    def query(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        include_sources: bool = True,
        temperature: float = None,
        system_prompt_type: str = "default"
    ) -> Dict:
        """
        Execute RAG query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_reranking: Whether to use reranking
            include_sources: Whether to include source information
            temperature: LLM temperature
            system_prompt_type: Type of system prompt to use
            
        Returns:
            Dictionary with answer and metadata
        """
        log.info(f"Processing query: {query[:100]}...")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            use_reranking=use_reranking
        )
        
        if not retrieved_docs:
            log.warning("No documents retrieved")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "retrieved_docs": []
            }
        
        log.debug(f"Retrieved {len(retrieved_docs)} documents")
        
        # Format prompt
        if include_sources:
            prompt = self.prompt_templates.format_qa_with_sources_prompt(
                query=query,
                context=retrieved_docs
            )
        else:
            context_texts = [doc["content"] for doc in retrieved_docs]
            prompt = self.prompt_templates.format_qa_prompt(
                query=query,
                context=context_texts
            )
        
        # Get system prompt
        system_prompt = self.prompt_templates.get_system_prompt(system_prompt_type)
        
        # Generate answer
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        log.info("Answer generated successfully")
        
        # Prepare sources
        sources = []
        for i, doc in enumerate(retrieved_docs):
            sources.append({
                "index": i + 1,
                "content": doc["content"][:200] + "...",  # Truncate for brevity
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "score": doc.get("score", 0),
                "rerank_score": doc.get("rerank_score")
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "query": query
        }
    
    def query_stream(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        temperature: float = None,
        system_prompt_type: str = "default"
    ) -> Iterator[Dict]:
        """
        Execute RAG query with streaming response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_reranking: Whether to use reranking
            temperature: LLM temperature
            system_prompt_type: Type of system prompt to use
            
        Yields:
            Dictionaries with answer chunks and metadata
        """
        log.info(f"Processing streaming query: {query[:100]}...")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            use_reranking=use_reranking
        )
        
        if not retrieved_docs:
            yield {
                "type": "answer",
                "content": "I couldn't find any relevant information to answer your question.",
                "done": True
            }
            return
        
        # Yield sources first
        yield {
            "type": "sources",
            "content": retrieved_docs,
            "done": False
        }
        
        # Format prompt
        context_texts = [doc["content"] for doc in retrieved_docs]
        prompt = self.prompt_templates.format_qa_prompt(
            query=query,
            context=context_texts
        )
        
        # Get system prompt
        system_prompt = self.prompt_templates.get_system_prompt(system_prompt_type)
        
        # Stream answer
        for chunk in self.llm_client.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        ):
            yield {
                "type": "answer",
                "content": chunk,
                "done": False
            }
        
        # Signal completion
        yield {
            "type": "answer",
            "content": "",
            "done": True
        }
    
    def multi_query_retrieve(
        self,
        query: str,
        num_queries: int = 3,
        top_k: int = None
    ) -> List[Dict]:
        """
        Generate multiple query variations and retrieve documents.
        
        Args:
            query: Original query
            num_queries: Number of query variations to generate
            top_k: Number of documents to retrieve per query
            
        Returns:
            Deduplicated list of retrieved documents
        """
        log.info(f"Multi-query retrieval for: {query[:100]}...")
        
        # Generate query variations
        multi_query_prompt = self.prompt_templates.format_multi_query_prompt(query)
        variations_text = self.llm_client.generate(
            prompt=multi_query_prompt,
            temperature=0.7
        )
        
        # Parse variations
        variations = [line.strip() for line in variations_text.split("\n") if line.strip()]
        variations = [query] + variations[:num_queries - 1]  # Include original query
        
        log.debug(f"Generated {len(variations)} query variations")
        
        # Retrieve for each variation
        all_docs = {}
        for var_query in variations:
            docs = self.retriever.retrieve(
                query=var_query,
                top_k=top_k,
                use_reranking=False
            )
            
            # Deduplicate by ID
            for doc in docs:
                doc_id = doc["id"]
                if doc_id not in all_docs or doc["score"] > all_docs[doc_id]["score"]:
                    all_docs[doc_id] = doc
        
        # Sort by score
        result = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        
        log.info(f"Multi-query retrieval returned {len(result)} unique documents")
        return result
    
    def summarize_documents(
        self,
        document_ids: List[str] = None,
        max_docs: int = 5
    ) -> str:
        """
        Summarize multiple documents.
        
        Args:
            document_ids: List of document IDs to summarize
            max_docs: Maximum number of documents to summarize
            
        Returns:
            Summary text
        """
        # Get documents
        if document_ids:
            docs = self.retriever.vector_store.get_by_ids(document_ids[:max_docs])
        else:
            log.warning("No document IDs provided for summarization")
            return "No documents to summarize."
        
        if not docs:
            return "No documents found."
        
        # Format prompt
        doc_texts = [doc["content"] for doc in docs]
        prompt = self.prompt_templates.format_summarization_prompt(doc_texts)
        
        # Generate summary
        summary = self.llm_client.generate(
            prompt=prompt,
            temperature=0.3
        )
        
        return summary


if __name__ == "__main__":
    # Example usage
    chain = RAGChain()
    
    # Simple query
    # result = chain.query("What is machine learning?")
    # print(f"Answer: {result['answer']}")
    # print(f"Sources: {len(result['sources'])}")
    
    # Streaming query
    # print("Streaming answer:")
    # for chunk in chain.query_stream("Explain deep learning"):
    #     if chunk["type"] == "answer":
    #         print(chunk["content"], end="", flush=True)
