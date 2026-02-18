"""
RAG chain orchestration using LangChain Expression Language (LCEL).

LangChain Components Used:
- langchain_core.runnables.RunnablePassthrough  → Pass-through input in LCEL pipelines
- langchain_core.output_parsers.StrOutputParser → Parse LLM output to string
- ChatPromptTemplate                            → Prompt construction (from prompts.py)
- ChatHuggingFace / ChatOllama                  -> LLM (from llm.py)
- EnsembleRetriever                             → Hybrid retrieval (from hybrid_retriever.py)
- ContextualCompressionRetriever                → Reranking wrapper (from reranker.py)

Key LangChain LCEL pattern:
  chain = (
      {"context": retriever | format_docs, "query": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )
  answer = chain.invoke("What is machine learning?")
"""

from typing import List, Dict, Optional, Iterator

# ── LangChain LCEL components ───────────────────────────────────────────
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import settings
from src.utils.logger import log
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMClient
from src.generation.prompts import PromptTemplates


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved LangChain Documents into numbered context string."""
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}\n"
        f"Source: {doc.metadata.get('source', 'Unknown')}"
        for i, doc in enumerate(docs)
    )


class RAGChain:
    """
    Complete RAG chain built with LangChain Expression Language (LCEL).
    
    The core LCEL chain:
        {"context": retriever | format_docs, "query": RunnablePassthrough(),
         "system_prompt": lambda _: system_prompt}
        | ChatPromptTemplate
        | ChatOllama
        | StrOutputParser()
    """

    def __init__(
        self,
        retriever: HybridRetriever = None,
        llm_client: LLMClient = None,
        prompt_templates: PromptTemplates = None,
    ):
        # Initialize retriever with reranker
        if retriever is None:
            try:
                reranker = Reranker()
                self.retriever = HybridRetriever(reranker=reranker)
                log.info("RAG Chain initialized with reranker")
            except Exception as e:
                log.warning(f"Failed to initialize reranker: {e}. Proceeding without.")
                self.retriever = HybridRetriever()
        else:
            self.retriever = retriever

        self.llm_client = llm_client or LLMClient()
        self.prompt_templates = prompt_templates or PromptTemplates()

        # ── Build LCEL chain ─────────────────────────────────────
        self._build_lcel_chain()

        log.info("RAG Chain initialized (LCEL pipeline)")

    def _build_lcel_chain(self):
        """Build the core LCEL pipeline."""
        # LangChain retriever from HybridRetriever
        lc_retriever = self.retriever.as_retriever()

        # ── LCEL: retriever → format → prompt → LLM → parse ─────
        self.lcel_chain = (
            {
                "context": lc_retriever | RunnableLambda(_format_docs),
                "query": RunnablePassthrough(),
                "system_prompt": RunnableLambda(
                    lambda _: PromptTemplates.SYSTEM_PROMPT
                ),
            }
            | PromptTemplates.QA_WITH_SOURCES_PROMPT
            | self.llm_client.llm
            | StrOutputParser()
        )

    # ── main query method (backward-compatible) ─────────────────────
    def query(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        include_sources: bool = True,
        temperature: float = None,
        system_prompt_type: str = "default",
    ) -> Dict:
        """Execute RAG query, returning backward-compatible dict."""
        log.info(f"Processing query: {query[:100]}...")

        # Retrieve documents (using backward-compatible method for score info)
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            use_reranking=use_reranking,
        )

        if not retrieved_docs:
            log.warning("No documents retrieved")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "retrieved_docs": [],
            }

        log.debug(f"Retrieved {len(retrieved_docs)} documents")

        # Format prompt
        if include_sources:
            prompt = self.prompt_templates.format_qa_with_sources_prompt(
                query=query, context=retrieved_docs
            )
        else:
            context_texts = [doc["content"] for doc in retrieved_docs]
            prompt = self.prompt_templates.format_qa_prompt(
                query=query, context=context_texts
            )

        system_prompt = self.prompt_templates.get_system_prompt(system_prompt_type)

        # ── LangChain LLM invoke ─────────────────────────────────
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        log.info("Answer generated successfully")

        sources = [
            {
                "index": i + 1,
                "content": doc["content"][:200] + "...",
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "score": doc.get("score", 0),
                "rerank_score": doc.get("rerank_score"),
            }
            for i, doc in enumerate(retrieved_docs)
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "query": query,
        }

    # ── pure LCEL invocation ─────────────────────────────────────────
    def invoke(self, query: str) -> str:
        """
        Run the full LCEL chain in one call.
        
        Usage:
            answer = rag_chain.invoke("What is attention?")
        """
        return self.lcel_chain.invoke(query)

    # ── streaming ────────────────────────────────────────────────────
    def query_stream(
        self,
        query: str,
        top_k: int = None,
        use_reranking: bool = True,
        temperature: float = None,
        system_prompt_type: str = "default",
    ) -> Iterator[Dict]:
        """Stream RAG answer chunks (backward-compatible)."""
        log.info(f"Processing streaming query: {query[:100]}...")

        retrieved_docs = self.retriever.retrieve(
            query=query, top_k=top_k, use_reranking=use_reranking
        )

        if not retrieved_docs:
            yield {
                "type": "answer",
                "content": "I couldn't find any relevant information to answer your question.",
                "done": True,
            }
            return

        yield {"type": "sources", "content": retrieved_docs, "done": False}

        context_texts = [doc["content"] for doc in retrieved_docs]
        prompt = self.prompt_templates.format_qa_prompt(query=query, context=context_texts)
        system_prompt = self.prompt_templates.get_system_prompt(system_prompt_type)

        # ── LangChain streaming ──────────────────────────────────
        for chunk in self.llm_client.generate_stream(
            prompt=prompt, system_prompt=system_prompt, temperature=temperature
        ):
            yield {"type": "answer", "content": chunk, "done": False}

        yield {"type": "answer", "content": "", "done": True}

    # ── multi-query retrieval ────────────────────────────────────────
    def multi_query_retrieve(
        self, query: str, num_queries: int = 3, top_k: int = None
    ) -> List[Dict]:
        """Generate multiple query variations and retrieve documents."""
        log.info(f"Multi-query retrieval for: {query[:100]}...")

        multi_query_prompt = self.prompt_templates.format_multi_query_prompt(query)
        variations_text = self.llm_client.generate(prompt=multi_query_prompt, temperature=0.7)

        variations = [line.strip() for line in variations_text.split("\n") if line.strip()]
        variations = [query] + variations[: num_queries - 1]

        all_docs = {}
        for var_query in variations:
            docs = self.retriever.retrieve(query=var_query, top_k=top_k, use_reranking=False)
            for doc in docs:
                doc_id = doc["id"]
                if doc_id not in all_docs or doc["score"] > all_docs[doc_id]["score"]:
                    all_docs[doc_id] = doc

        result = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        log.info(f"Multi-query retrieval returned {len(result)} unique documents")
        return result

    # ── summarization ────────────────────────────────────────────────
    def summarize_documents(self, document_ids: List[str] = None, max_docs: int = 5) -> str:
        if not document_ids:
            return "No documents to summarize."
        docs = self.retriever.vector_store.get_by_ids(document_ids[:max_docs])
        if not docs:
            return "No documents found."
        doc_texts = [doc["content"] for doc in docs]
        prompt = self.prompt_templates.format_summarization_prompt(doc_texts)
        return self.llm_client.generate(prompt=prompt, temperature=0.3)

    # ── expose LCEL chain for external composition ───────────────────
    def as_chain(self):
        """Return the LCEL chain for composition with other runnables."""
        return self.lcel_chain


if __name__ == "__main__":
    chain = RAGChain()
    # Backward-compatible:
    # result = chain.query("What is machine learning?")
    # Pure LCEL:
    # answer = chain.invoke("What is machine learning?")
