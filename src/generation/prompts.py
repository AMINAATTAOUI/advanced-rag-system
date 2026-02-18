"""
Prompt templates for RAG system using LangChain's ChatPromptTemplate.

LangChain Components Used:
- langchain_core.prompts.ChatPromptTemplate   → Structured prompt construction
- langchain_core.prompts.MessagesPlaceholder  → Dynamic message insertion

Key LangChain patterns:
  prompt = ChatPromptTemplate.from_messages([
      ("system", "..."),
      ("human", "{context}\n{query}"),
  ])
  chain = prompt | llm | StrOutputParser()
"""

from typing import List, Dict

# ── LangChain Prompt Templates ──────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate


class PromptTemplates:
    """Collection of LangChain ChatPromptTemplates for the RAG system."""

    # ── System prompts (plain strings, injected into templates) ──────
    SYSTEM_PROMPT = (
        "You are a helpful AI assistant that answers questions based on the provided context.\n"
        "Your responses should be:\n"
        "- Accurate and based solely on the given context\n"
        "- Clear and well-structured\n"
        "- Honest about limitations (if context doesn't contain the answer)\n"
        "- Professional and informative\n\n"
        "If the context doesn't contain enough information to answer the question, say so clearly."
    )

    CONCISE_SYSTEM_PROMPT = (
        "You are a precise AI assistant that provides concise, faithful answers "
        "based strictly on the provided context.\n\n"
        "Your responses must:\n"
        "- Be brief and to-the-point\n"
        "- Quote or closely paraphrase the context\n"
        "- Never add information not in the context\n"
        "- Cite sources using [1], [2], etc.\n"
        "- State clearly if the context doesn't contain the answer\n\n"
        "Prioritize accuracy and faithfulness over comprehensiveness."
    )

    RESEARCH_ASSISTANT_PROMPT = (
        "You are an expert research assistant specializing in academic papers and scientific literature.\n"
        "Your role is to:\n"
        "- Analyze and synthesize information from research papers\n"
        "- Provide accurate citations and references\n"
        "- Explain complex concepts clearly\n"
        "- Identify key findings and methodologies\n"
        "- Point out limitations or gaps in the research when relevant\n\n"
        "Always base your answers on the provided context and cite specific papers when possible."
    )

    # ── LangChain ChatPromptTemplates ────────────────────────────────

    QA_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human",
         "Context information from relevant documents:\n{context}\n\n"
         "Based on the above context, please answer the following question:\n"
         "Question: {query}\n\n"
         "Instructions:\n"
         "- Use only the information provided in the context\n"
         "- If the context doesn't contain the answer, state that clearly\n"
         "- Cite specific sources when possible\n"
         "- Be concise but comprehensive\n\n"
         "Answer:"),
    ])

    QA_WITH_SOURCES_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human",
         "Context information from relevant documents (ranked by relevance):\n{context}\n\n"
         "Question: {query}\n\n"
         "Instructions:\n"
         "- PRIORITIZE the TOP-RANKED sources [1], [2], etc. - they are most relevant\n"
         "- Answer CONCISELY using ONLY information from the context above\n"
         "- Start with the most relevant information from the highest-ranked sources\n"
         "- Quote or paraphrase directly from the sources\n"
         "- Use citations [1], [2], etc. to reference sources\n"
         "- If the context doesn't fully answer the question, state what's missing\n"
         "- Do NOT add information not present in the context\n\n"
         "Answer:"),
    ])

    SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes documents accurately."),
        ("human",
         "Please provide a comprehensive summary of the following documents:\n\n"
         "{context}\n\n"
         "Summary should include:\n"
         "- Main topics and themes\n"
         "- Key findings or arguments\n"
         "- Important details and examples\n"
         "- Overall conclusions\n\n"
         "Summary:"),
    ])

    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant helping to improve search queries."),
        ("human",
         "Given the original question, generate 3 alternative versions that capture "
         "different aspects or phrasings of the same information need.\n\n"
         "Original question: {query}\n\n"
         "Generate 3 alternative questions (one per line):"),
    ])

    QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human",
         "Given the following query, suggest related terms and concepts that could "
         "help find relevant information:\n\n"
         "Query: {query}\n\nRelated terms and concepts:"),
    ])

    # ── Evaluation prompts ───────────────────────────────────────────
    RELEVANCE_EVAL_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluation assistant."),
        ("human",
         "Evaluate if the following context is relevant to answer the question.\n\n"
         "Question: {query}\nContext: {context}\n\n"
         "Is this context relevant? Answer with 'Yes' or 'No' and provide a brief explanation.\n\n"
         "Evaluation:"),
    ])

    FAITHFULNESS_EVAL_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluation assistant."),
        ("human",
         "Evaluate if the answer is faithful to the provided context.\n\n"
         "Context: {context}\nAnswer: {answer}\n\n"
         "Is the answer faithful? Answer with 'Yes' or 'No' and explain any discrepancies.\n\n"
         "Evaluation:"),
    ])

    # ── formatting helpers (backward-compatible) ─────────────────────

    @staticmethod
    def format_qa_prompt(query: str, context: List[str]) -> str:
        """Format a QA prompt string (backward-compatible)."""
        context_text = "\n\n".join(
            [f"[Document {i+1}]\n{ctx}" for i, ctx in enumerate(context)]
        )
        messages = PromptTemplates.QA_PROMPT.format_messages(
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            query=query,
            context=context_text,
        )
        # Return only the human message content for legacy callers
        return messages[-1].content

    @staticmethod
    def format_qa_with_sources_prompt(query: str, context: List[Dict]) -> str:
        """Format a QA prompt with sources (backward-compatible)."""
        context_text = "\n\n".join([
            f"[{i+1}] {ctx['content']}\n"
            f"Source: {ctx.get('metadata', {}).get('source', 'Unknown')}\n"
            f"Relevance Score: {ctx.get('score', 'N/A')}"
            for i, ctx in enumerate(context)
        ])
        messages = PromptTemplates.QA_WITH_SOURCES_PROMPT.format_messages(
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            query=query,
            context=context_text,
        )
        return messages[-1].content

    @staticmethod
    def format_summarization_prompt(documents: List[str]) -> str:
        context_text = "\n\n---\n\n".join(documents)
        messages = PromptTemplates.SUMMARIZATION_PROMPT.format_messages(context=context_text)
        return messages[-1].content

    @staticmethod
    def format_multi_query_prompt(query: str) -> str:
        messages = PromptTemplates.MULTI_QUERY_PROMPT.format_messages(query=query)
        return messages[-1].content

    @staticmethod
    def format_query_expansion_prompt(query: str) -> str:
        messages = PromptTemplates.QUERY_EXPANSION_PROMPT.format_messages(query=query)
        return messages[-1].content

    @staticmethod
    def get_system_prompt(prompt_type: str = "default") -> str:
        prompts = {
            "default": PromptTemplates.SYSTEM_PROMPT,
            "research": PromptTemplates.RESEARCH_ASSISTANT_PROMPT,
            "concise": PromptTemplates.CONCISE_SYSTEM_PROMPT,
        }
        return prompts.get(prompt_type, PromptTemplates.SYSTEM_PROMPT)


if __name__ == "__main__":
    query = "What is machine learning?"
    context = [
        "Machine learning is a subset of artificial intelligence.",
        "It focuses on building systems that learn from data.",
    ]
    prompt = PromptTemplates.format_qa_prompt(query, context)
    print("QA Prompt:\n", prompt)

    # Pure LangChain path:
    # chain = PromptTemplates.QA_PROMPT | llm | StrOutputParser()
    # result = chain.invoke({"system_prompt": "...", "query": "...", "context": "..."})
