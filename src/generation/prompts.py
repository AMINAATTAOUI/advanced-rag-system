"""
Prompt templates for RAG system.
Contains various prompt templates for different use cases.
"""

from typing import List, Dict
from string import Template


class PromptTemplates:
    """Collection of prompt templates for RAG."""
    
    # System prompts
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.
Your responses should be:
- Accurate and based solely on the given context
- Clear and well-structured
- Honest about limitations (if context doesn't contain the answer)
- Professional and informative

If the context doesn't contain enough information to answer the question, say so clearly."""

    CONCISE_SYSTEM_PROMPT = """You are a precise AI assistant that provides concise, faithful answers based strictly on the provided context.

Your responses must:
- Be brief and to-the-point
- Quote or closely paraphrase the context
- Never add information not in the context
- Cite sources using [1], [2], etc.
- State clearly if the context doesn't contain the answer

Prioritize accuracy and faithfulness over comprehensiveness."""

    RESEARCH_ASSISTANT_PROMPT = """You are an expert research assistant specializing in academic papers and scientific literature.
Your role is to:
- Analyze and synthesize information from research papers
- Provide accurate citations and references
- Explain complex concepts clearly
- Identify key findings and methodologies
- Point out limitations or gaps in the research when relevant

Always base your answers on the provided context and cite specific papers when possible."""

    # RAG prompts
    QA_TEMPLATE = Template("""Context information from relevant documents:
$context

Based on the above context, please answer the following question:
Question: $query

Instructions:
- Use only the information provided in the context
- If the context doesn't contain the answer, state that clearly
- Cite specific sources when possible
- Be concise but comprehensive

Answer:""")

    QA_WITH_SOURCES_TEMPLATE = Template("""Context information from relevant documents (ranked by relevance):
$context

Question: $query

Instructions:
- PRIORITIZE the TOP-RANKED sources [1], [2], etc. - they are most relevant
- Answer CONCISELY using ONLY information from the context above
- Start with the most relevant information from the highest-ranked sources
- Quote or paraphrase directly from the sources
- Use citations [1], [2], etc. to reference sources
- If the context doesn't fully answer the question, state what's missing
- Do NOT add information not present in the context

Answer:""")

    SUMMARIZATION_TEMPLATE = Template("""Please provide a comprehensive summary of the following documents:

$context

Summary should include:
- Main topics and themes
- Key findings or arguments
- Important details and examples
- Overall conclusions

Summary:""")

    MULTI_QUERY_TEMPLATE = Template("""You are an AI assistant helping to improve search queries.
Given the original question, generate 3 alternative versions that capture different aspects or phrasings of the same information need.

Original question: $query

Generate 3 alternative questions (one per line):""")

    QUERY_EXPANSION_TEMPLATE = Template("""Given the following query, suggest related terms and concepts that could help find relevant information:

Query: $query

Related terms and concepts:""")

    # Evaluation prompts
    RELEVANCE_EVAL_TEMPLATE = Template("""Evaluate if the following context is relevant to answer the question.

Question: $query
Context: $context

Is this context relevant? Answer with 'Yes' or 'No' and provide a brief explanation.

Evaluation:""")

    FAITHFULNESS_EVAL_TEMPLATE = Template("""Evaluate if the answer is faithful to the provided context (i.e., doesn't contain information not present in the context).

Context: $context
Answer: $answer

Is the answer faithful to the context? Answer with 'Yes' or 'No' and explain any discrepancies.

Evaluation:""")

    @staticmethod
    def format_qa_prompt(query: str, context: List[str]) -> str:
        """
        Format a QA prompt with query and context.
        
        Args:
            query: User query
            context: List of context strings
            
        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join([
            f"[Document {i+1}]\n{ctx}" 
            for i, ctx in enumerate(context)
        ])
        
        return PromptTemplates.QA_TEMPLATE.substitute(
            query=query,
            context=context_text
        )
    
    @staticmethod
    def format_qa_with_sources_prompt(
        query: str,
        context: List[Dict]
    ) -> str:
        """
        Format a QA prompt with sources.

        Args:
            query: User query
            context: List of context dictionaries with 'content' and 'metadata'

        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join([
            f"[{i+1}] {ctx['content']}\nSource: {ctx.get('metadata', {}).get('source', 'Unknown')}\nRelevance Score: {ctx.get('score', 'N/A')}"
            for i, ctx in enumerate(context)
        ])

        return PromptTemplates.QA_WITH_SOURCES_TEMPLATE.substitute(
            query=query,
            context=context_text
        )
    
    @staticmethod
    def format_summarization_prompt(documents: List[str]) -> str:
        """
        Format a summarization prompt.
        
        Args:
            documents: List of document texts
            
        Returns:
            Formatted prompt
        """
        context_text = "\n\n---\n\n".join(documents)
        
        return PromptTemplates.SUMMARIZATION_TEMPLATE.substitute(
            context=context_text
        )
    
    @staticmethod
    def format_multi_query_prompt(query: str) -> str:
        """
        Format a multi-query generation prompt.
        
        Args:
            query: Original query
            
        Returns:
            Formatted prompt
        """
        return PromptTemplates.MULTI_QUERY_TEMPLATE.substitute(query=query)
    
    @staticmethod
    def format_query_expansion_prompt(query: str) -> str:
        """
        Format a query expansion prompt.
        
        Args:
            query: Original query
            
        Returns:
            Formatted prompt
        """
        return PromptTemplates.QUERY_EXPANSION_TEMPLATE.substitute(query=query)
    
    @staticmethod
    def get_system_prompt(prompt_type: str = "default") -> str:
        """
        Get a system prompt by type.
        
        Args:
            prompt_type: Type of system prompt ('default', 'research', 'concise')
            
        Returns:
            System prompt string
        """
        prompts = {
            "default": PromptTemplates.SYSTEM_PROMPT,
            "research": PromptTemplates.RESEARCH_ASSISTANT_PROMPT,
            "concise": PromptTemplates.CONCISE_SYSTEM_PROMPT
        }
        
        return prompts.get(prompt_type, PromptTemplates.SYSTEM_PROMPT)


if __name__ == "__main__":
    # Example usage
    query = "What is machine learning?"
    context = [
        "Machine learning is a subset of artificial intelligence.",
        "It focuses on building systems that learn from data."
    ]
    
    # Format QA prompt
    prompt = PromptTemplates.format_qa_prompt(query, context)
    print("QA Prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Format multi-query prompt
    multi_query_prompt = PromptTemplates.format_multi_query_prompt(query)
    print("Multi-Query Prompt:")
    print(multi_query_prompt)
